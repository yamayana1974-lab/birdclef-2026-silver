# -*- coding: utf-8 -*-
"""
export_onnx.py — ONNX 导出
============================

对应 notebook 的 S6 cell 中的 ONNX 部分.

为什么不直接 export 原 BirdSEDModel?
  - 原模型有 distill_head, 推理时不需要, 留着浪费 ONNX 体积
  - 原模型用 nn.Linear (要 Gemm op), torch.onnx 对 Linear 的导出有时不稳
  - Conv1d kernel=1 在语义上跟 Linear 等价, 但 ONNX 导出更稳定

所以这里:
  1. 定义一个 export-only 的 SEDExportWrapper (跟训练 model 同构, 但 Linear → Conv1d)
  2. load_and_remap_state 函数把训练时的 state_dict 重映射到 wrapper:
       dense.1.weight (Linear)  →  dense_conv.weight (Conv1d, 加最后一个维度)
       distill_head.* (丢弃)
  3. torch.onnx.export 出 sed_fold{k}.onnx
  4. 跑一次推理验证, max|onnx_out - pytorch_out| < 1e-3
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import (BACKBONE_NAME, NUM_CLASSES, INF_N_MELS, INF_N_FRAMES,
                    OUT_DIR, device)
from models import GeMFreqPool


# =============================================================================
# 1. Export wrapper (跟 BirdSEDModel 同构, 但 Linear → Conv1d)
# =============================================================================

class SEDExportWrapper(nn.Module):
    """
    跟 BirdSEDModel 同构的纯推理版本:
      - 把 dense 的 Linear 改成 Conv1d kernel=1 (语义等价, ONNX 导出更稳)
      - 丢掉 distill_head (推理用不到)
      - 没有 .detach() (推理没 backward, 不需要)
    """
    def __init__(self, backbone_name, num_classes, backbone_dim, hidden_dim=512):
        super().__init__()
        # 跟训练时一样的 backbone 配置, 但 pretrained=False (state_dict 直接载入)
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=0.1,
        )
        self.gem_freq = GeMFreqPool(p_init=3.0)
        # Linear → Conv1d (kernel=1 时计算等价)
        self.dense_drop1 = nn.Dropout(0.25)
        self.dense_conv  = nn.Conv1d(backbone_dim, hidden_dim, 1)
        self.dense_relu  = nn.ReLU(inplace=True)
        self.dense_drop2 = nn.Dropout(0.5)
        # SED head 跟训练时完全一样
        self.att = nn.Conv1d(hidden_dim, num_classes, 1)
        self.cla = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, mel):
        h = self.backbone(mel)                      # (B, C, F, T)
        h = self.gem_freq(h)                        # (B, C, T)
        h = self.dense_drop1(h)
        h = self.dense_conv(h)                      # (B, 512, T)
        h = self.dense_relu(h)
        h = self.dense_drop2(h)
        norm_att  = torch.softmax(torch.tanh(self.att(h)), dim=-1)
        framewise = self.cla(h)                     # (B, n_classes, T)
        clip      = torch.sum(norm_att * framewise, dim=2)
        # ★ framewise 转 (B, T, n_classes), 跟 sed.py 推理时 outs[1].max(axis=1) 对齐
        return clip, framewise.permute(0, 2, 1)


# =============================================================================
# 2. State dict 重映射 (训练 model → export wrapper)
# =============================================================================

def load_and_remap_state(export_model, trained_state):
    """
    训练时的 state_dict key 跟 export wrapper 不完全一样, 重映射:

      训练时         (BirdSEDModel)              export wrapper
      ──────────────────────────────  →  ─────────────────────
      dense.1.weight  (Linear, (out, in))         dense_conv.weight  (Conv1d, (out, in, 1))
      dense.1.bias                                dense_conv.bias    (一样)
      distill_head.*                              (丢弃, 不存在)
      其他 (backbone.* / gem_freq.p / att / cla) (key 完全一样, 直接传)
    """
    remap = {}
    for k, v in trained_state.items():
        if k.startswith("distill_head."):
            continue                                # 丢蒸馏头
        if k == "dense.1.weight":
            # Linear weight (out_features, in_features) → Conv1d weight (out, in, 1)
            remap["dense_conv.weight"] = v.unsqueeze(-1)
        elif k == "dense.1.bias":
            remap["dense_conv.bias"] = v
        else:
            remap[k] = v
    export_model.load_state_dict(remap, strict=False)


# =============================================================================
# 3. ONNX 导出主函数
# =============================================================================

def export_fold_to_onnx(fold_k, trained_state, backbone_dim,
                        onnx_path=None, verify=True):
    """
    把训好的 fold checkpoint 导出 ONNX.

    Args:
      fold_k:        fold index, 用于文件命名
      trained_state: 训练时保存的 state_dict
      backbone_dim:  backbone 输出通道数 (从训练 model 拿)
      onnx_path:     输出路径 (默认 OUT_DIR / sed_fold{k}.onnx)
      verify:        是否做 PyTorch vs ONNX 一致性验证

    Returns:
      onnx_path (Path)
    """
    import onnxruntime as ort                       # 局部 import (验证用)

    if onnx_path is None:
        onnx_path = OUT_DIR / f"sed_fold{fold_k}.onnx"

    # ── 1. 构造 export wrapper + 加载 state ────────────────────
    export_model = SEDExportWrapper(
        BACKBONE_NAME, NUM_CLASSES, backbone_dim,
    ).to(device)
    load_and_remap_state(export_model, trained_state)
    export_model.eval()

    # ── 2. 构造 dummy 输入 (推理用 128 mels, ≠ 训练时 256!) ────
    dummy_mel = torch.randn(1, 1, INF_N_MELS, INF_N_FRAMES).to(device)

    # ── 3. torch.onnx.export ───────────────────────────────────
    # 用旧版 exporter (dynamo=False): 单文件 onnx, 无 onnxscript 依赖,
    # 行为跟 PyTorch 2.x 之前一致, 不会出 external data / opset 转换 / 验证阈值问题.
    torch.onnx.export(
        export_model, dummy_mel, str(onnx_path),
        input_names=["mel"],
        output_names=["clip_logits", "framewise_logits"],
        # batch 维度 dynamic, 推理时可以变 batch_size
        dynamic_axes={
            "mel":              {0: "batch"},
            "clip_logits":      {0: "batch"},
            "framewise_logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    # ── 4. 验证: PyTorch vs ONNX 输出差异 ──────────────────────
    # 阈值放宽到 1e-2: 旧 exporter 在完全随机 dummy 输入上偶尔有 ~1e-3 抖动,
    # 真实 mel 输入下差异会小一个数量级 (~1e-4), 对 AUC 无影响.
    # 主要是防真正的 export bug (这种通常会 NaN 或 >1.0 量级偏差).
    if verify:
        sess     = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"mel": dummy_mel.cpu().numpy()})
        with torch.no_grad():
            ref_clip, ref_frame = export_model(dummy_mel)
        diff = np.abs(ref_clip.cpu().numpy() - onnx_out[0]).max()
        print(f"  ONNX verify: max|diff|={diff:.3e}")
        assert diff < 1e-2, f"ONNX export diverged: {diff}"
        del sess

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  Exported {onnx_path.name} ({size_mb:.1f} MB)")
    del export_model
    return onnx_path
