# -*- coding: utf-8 -*-
"""
models.py — 模型架构 + 梅尔变换 + SpecAugment + Perch teacher
==============================================================

对应 notebook 的 S3 cell.

包含 5 个组件 (按调用顺序):
  1. MelSpecTransform    — GPU 上算 mel + dB (训练用)
  2. SpecAugment         — mel 上的随机 freq/time mask
  3. PerchTeacher        — frozen Perch v2 ONNX, 当蒸馏 teacher
  4. DistillHead         — 把 backbone feat 投到 Perch emb 空间 (GAP + Linear)
  5. GeMFreqPool         — 可学的 generalized mean pooling (频率维)
  6. BirdSEDModel        — 主模型: EfficientNet B0 + GeM + bottleneck + att/cla

★ 核心 trick (stop gradient + distillation):
  - backbone 用 `.detach()` 喂给 SED head, 让 backbone 不被 cls_loss 更新
  - 蒸馏分支 DistillHead 把 backbone 输出回归到 Perch emb (MSE)
  - 训完后丢掉 DistillHead, 只保留 backbone + SED head 导出 ONNX
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm

from config import (SR, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX,
                    TRAIN_SAMPLES, NUM_CLASSES, BACKBONE_NAME,
                    USE_PERCH_DISTILL, PERCH_EMBED_DIM,
                    FREQ_MASK_PARAM, TIME_MASK_PARAM,
                    NUM_FREQ_MASKS, NUM_TIME_MASKS,
                    device)


# =============================================================================
# 1. MelSpecTransform — GPU 上算 mel + dB
# =============================================================================

class MelSpecTransform(nn.Module):
    """
    torchaudio 的 MelSpectrogram + AmplitudeToDB, 全在 GPU 上跑.
    输入: (B, 1, time) 波形, float32 in [-1, 1]
    输出: (B, 1, n_mels, T_frames) mel dB
    """
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))


# =============================================================================
# 2. SpecAugment — mel 上的 freq/time mask
# =============================================================================

class SpecAugment(nn.Module):
    """
    在 mel 上随机 mask 频率带和时间段 (类似 dropout, 防过拟合).
        FreqMask × NUM_FREQ_MASKS 次, 每次最大 FREQ_MASK_PARAM 个频率 bin
        TimeMask × NUM_TIME_MASKS 次, 每次最大 TIME_MASK_PARAM 个 frame
    只在训练时启用 (model.train() 模式).
    """
    def __init__(self):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=FREQ_MASK_PARAM)
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=TIME_MASK_PARAM)

    def forward(self, mel):
        for _ in range(NUM_FREQ_MASKS):
            mel = self.freq_mask(mel)
        for _ in range(NUM_TIME_MASKS):
            mel = self.time_mask(mel)
        return mel


# =============================================================================
# 3. PerchTeacher — frozen Perch v2 ONNX teacher
# =============================================================================

class PerchTeacher:
    """
    冻结的 Perch v2 (ONNX 加载).
        输入: (B, 160000) 5s 波形, float32 in [-1, 1]
        输出: (B, 1536) embedding
    Perch 在 14795 个物种上训过, 它的 emb 是"通用鸟类特征",
    我们让 student backbone 学着模仿这个 emb (MSE 蒸馏).

    重要:
      - 这个 teacher 永远不更新参数 (它在 ONNX 里, 没法 backprop)
      - 蒸馏 loss 通过 student 的 DistillHead 来反传
    """
    def __init__(self, onnx_path, device_str="cuda"):
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device_str == "cuda" else ["CPUExecutionProvider"])
        self.session     = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        # 自动识别 embedding output 的 index (找最后一维 = 1536 的那个)
        self._embed_idx = None
        for i, o in enumerate(self.session.get_outputs()):
            if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                self._embed_idx = i
                break
        if self._embed_idx is None:
            self._embed_idx = 1                    # fallback
        print(f"Perch ONNX loaded: embed_idx={self._embed_idx}")

    @torch.no_grad()
    def embed(self, waveforms_5s):
        """
        Args:
          waveforms_5s: (B, 160000) float32 tensor (任意 device)
        Returns:
          (B, 1536) float32 tensor (CPU 上)
        """
        wav_np = waveforms_5s.cpu().numpy()
        results = self.session.run(None, {self.input_name: wav_np})
        return torch.from_numpy(results[self._embed_idx]).float()


# =============================================================================
# 4. DistillHead — backbone feature → Perch emb space
# =============================================================================

class DistillHead(nn.Module):
    """
    把 backbone 的 (B, C, F, T) 输出投影到 Perch 的 1536-d 空间.
    用 GAP (sum 在 F,T 维度上) + Linear.

    训完后这个 head 丢掉, 只保留 backbone + SED head 导出 ONNX.
    """
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)

    def forward(self, feature_map):
        gap = feature_map.mean(dim=[2, 3])         # (B, C, F, T) → (B, C)
        return self.proj(gap)                       # (B, embed_dim)


# =============================================================================
# 5. GeMFreqPool — 可学的 generalized mean pooling (频率维)
# =============================================================================

class GeMFreqPool(nn.Module):
    """
    Generalized Mean pooling 在频率维度上.
        y = (mean(x^p))^(1/p)
        p=1 → mean pool
        p→∞ → max pool
        p=3 (默认) 介于中间, 更尖锐
    可学的 p, 让模型自适应选最合适的池化强度.

    比传统 GAP (mean) 强很多:
      - 高频鸟叫 (sharp peak) 不会被低频背景稀释
      - 低 SNR 段也不会全被 max 主导

    输入: (B, C, F, T)
    输出: (B, C, T)  (频率维被池化掉)
    """
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)                  # p 不能 < 1 (不然 mean 没意义)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)                          # 沿频率维平均
        return x.pow(1.0 / p)


# =============================================================================
# 6. BirdSEDModel — 主模型
# =============================================================================

class BirdSEDModel(nn.Module):
    """
    Tucker SED 模型 (PANNs Cnn14_DecisionLevelAtt 风格 + 改进):

    [B, 1, 256, 313]  mel 输入
        ↓
    backbone (EfficientNet B0)         → (B, C=1280, F', T')
        ↓ (★ stop gradient 在这里发生)
    GeMFreqPool                         → (B, C, T')
        ↓
    dense bottleneck (Conv1d 1280→512)  → (B, 512, T')
        ↓
    ┌── att = Conv1d(512, 234)          → (B, 234, T')
    │   norm_att = softmax(tanh(att))
    └── cla = Conv1d(512, 234)          → (B, 234, T')   ← framewise_logits
        ↓
    clip_logits = Σ_t (norm_att * cla)  → (B, 234)

    Distillation 分支 (训练用, 推理丢掉):
        backbone feat → DistillHead → (B, 1536)  ← 跟 Perch emb 算 MSE

    ★ 核心 trick: SED head 用 detached backbone 输出, 不更新 backbone.
                  backbone 只被蒸馏 loss 更新.
    """
    def __init__(self, backbone_name=BACKBONE_NAME, num_classes=NUM_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        # ── backbone (timm EfficientNet B0, ImageNet noisy-student pretrained) ──
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        # 推断 backbone 输出维度 (跑一次 dummy 拿 shape)
        with torch.no_grad():
            n_tf  = TRAIN_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat  = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
            print(f"V2 backbone: {tuple(feat.shape)}  (C={self.backbone_dim})")

        # ── GeM 频率池化 ────────────────────────────────────────
        self.gem_freq = GeMFreqPool(p_init=3.0)

        # ── 512-d bottleneck (Linear, 训练时是 nn.Linear, 导出 ONNX 改成 Conv1d) ──
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # ── SED head (PANNs DecisionLevelAtt 风格) ─────────────
        # att: 每类一套时序 attention (Conv1d kernel=1 等价于 per-frame Linear)
        # cla: 每类一套 frame-level classifier
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.att.weight)
        nn.init.xavier_uniform_(self.cla.weight)
        self.att.bias.data.fill_(0.)
        self.cla.bias.data.fill_(0.)

        # ── 蒸馏分支 (可选) ────────────────────────────────────
        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x, return_framewise=False, return_distill=False):
        """
        Args:
          x:                (B, 1, n_mels, T) 标准化的 mel dB
          return_framewise: 是否返回 framewise_logits (B, T', n_classes)
          return_distill:   是否返回 distill_emb (B, 1536)

        Returns:
          (clip_logits, framewise?, distill_emb?) 根据上面两个 flag 组合
        """
        h = self.backbone(x)                       # (B, C, F, T')

        # 蒸馏分支用真实 gradient flow 的 backbone 输出
        distill_emb = None
        if return_distill and hasattr(self, "distill_head"):
            distill_emb = self.distill_head(h)

        # ★ stop gradient: SED head 不更新 backbone
        h_cls = h.detach() if USE_PERCH_DISTILL else h

        # 频率维池化 + bottleneck
        h_cls = self.gem_freq(h_cls)               # (B, C, T')
        h_cls = h_cls.permute(0, 2, 1)             # (B, T', C)
        h_cls = self.dense(h_cls)                  # (B, T', 512)
        h_cls = h_cls.permute(0, 2, 1)             # (B, 512, T')

        # PANNs decision-level attention
        norm_att         = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)          # (B, n_classes, T')
        clip_logits      = torch.sum(norm_att * framewise_logits, dim=2)  # (B, n_classes)

        fw = framewise_logits.permute(0, 2, 1) if return_framewise else None

        if return_framewise and return_distill:
            return clip_logits, fw, distill_emb
        elif return_framewise:
            return clip_logits, fw
        elif return_distill:
            return clip_logits, distill_emb
        return clip_logits


def make_model():
    """工厂函数: 创建 BirdSEDModel 并放到 device 上."""
    return BirdSEDModel(BACKBONE_NAME).to(device)
