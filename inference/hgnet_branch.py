# -*- coding: utf-8 -*-
"""
hgnet_branch.py — HGNetV2-B0 4-fold ensemble 推理 + 与 0.949 方案融合
======================================================================

把 hgnet_model 训出的 HGNetV2-B0 baseline (Kaggle LB 0.888) 作为
**附加分支**, 以 0.05 的权重融合进 0.949 方案 (birdclef-2026-v6_0.949.ipynb)
最终产出的 submission.csv.

设计原则:
  - 跟 0.949 notebook 的 row_id / 类别列保持完全一致:
    * 每个 60s 文件切 12 个 5s 窗口
    * row_id 格式: "{stem}_{(w+1)*5}"
    * 234 个 primary_label, 顺序由 sample_submission.csv 决定
  - 4-fold HGNet 模型 prob 平均, 时间维 Gaussian σ=0.65 平滑
  - 自包含: 模型类直接内联在本文件, 不依赖训练脚本 (避免它的路径副作用)
  - 推理只读: 不写任何中间状态, 只产出 submission_hgnet.csv

用法 (两种):
=============================================================================

▍A. Kaggle Notebook 末尾追加 (最常见用法):
-----------------------------------------------------------------------------
    # 把本文件作为 dataset 或 utility 加到 0.949 notebook
    from hgnet_branch import run_hgnet_inference, blend_with_hgnet

    # 1) 跑 4-fold HGNet 推理 → submission_hgnet.csv
    run_hgnet_inference(
        test_paths     = test_paths,                       # 跟 notebook 已有的 test_paths 一致
        ckpt_dir       = "/kaggle/input/birdclef-hgnetv2-b0-ckpts",
        primary_labels = PRIMARY_LABELS,
        out_csv        = "submission_hgnet.csv",
    )

    # 2) 把 HGNet 以 0.05 权重 rank-blend 进 0.949 的最终 submission
    blend_with_hgnet(
        base_csv  = "submission.csv",                      # 0.949 notebook 输出的最终 csv
        hgnet_csv = "submission_hgnet.csv",
        out_csv   = "submission.csv",                      # 覆盖原 submission
        w_base    = 0.95,
        w_hgnet   = 0.05,
    )

▍B. 本地命令行用法:
-----------------------------------------------------------------------------
    # 1) 先跑 HGNet 推理
    python hgnet_branch.py infer \
        --ckpt_dir /path/to/repo/hgnet_model \
        --test_dir ./data/birdclef-2026/test_soundscapes \
        --sample_sub ./data/birdclef-2026/sample_submission.csv \
        --out submission_hgnet.csv

    # 2) 再跟 0.949 的输出做 0.95/0.05 融合
    python hgnet_branch.py blend \
        --base submission_949.csv \
        --hgnet submission_hgnet.csv \
        --out submission.csv \
        --w_base 0.95 --w_hgnet 0.05

=============================================================================
"""
from __future__ import annotations

import argparse
import gc
import math
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

import torch
from torch import nn
import torchaudio
from torchvision.transforms import v2 as tvt_v2

import timm


# =============================================================================
# 1. 训练时的超参 (必须跟 birdclef_2026_hgnetv2_b0_baseline_training.py 一致!)
# =============================================================================
import os as _os

SR              = 32_000
WINDOW_SEC      = 5
WINDOW_SAMPLES  = SR * WINDOW_SEC                  # 160_000
N_WINDOWS       = 12                                # 60s / 5s
N_FOLDS         = 4
N_CLASSES       = 234

# backbone 可通过环境变量 ``BIRDCLEF_HGNET_BACKBONE`` 切换 (例如换成
# ``hgnetv2_b4.ssld_stage2_ft_in1k``). 默认保持 b0 以兼容老的 ckpt.
MODEL_NAME      = _os.environ.get(
    "BIRDCLEF_HGNET_BACKBONE", "hgnetv2_b0.ssld_stage2_ft_in1k",
)
HEAD_DROPOUT    = 0.5
LSE_TEMPERATURE = 1.0

MEL_SPECTROGRAM_PARAMS = dict(
    sample_rate=32_000,
    n_fft=2048,
    win_length=626,
    hop_length=313,
    f_min=20,
    n_mels=256,
    power=2.0,
    center=True,
    pad_mode="reflect",
    norm="slaney",
    mel_scale="htk",
)
LMS_SHAPE = (256, 256)
TOP_DB    = 80.0


# =============================================================================
# 2. 模型类 (内联, 避免依赖训练脚本里的 ROOT/Path.cwd() 副作用)
# =============================================================================

class LogMelSpectrogramTransform(nn.Module):
    """波形 (B, T) → log-mel "图像" (B, 1, 256, 256). 每条样本 min-max 归一化."""

    def __init__(self, mel_params: tp.Dict, top_db: float,
                 lms_shape: tp.Tuple[int, int] = (256, 256)):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_params)
        self.db            = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
        self.resize        = tvt_v2.Resize(size=lms_shape)

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_transform(wave)
        lms      = self.db(mel_spec)
        lms      = self.resize(lms)

        B        = lms.shape[0]
        lms_flat = lms.reshape(B, -1)
        lms_min  = lms_flat.min(dim=1)[0][:, None, None]
        lms_max  = lms_flat.max(dim=1)[0][:, None, None]
        lms      = (lms - lms_min) / (lms_max - lms_min + 1e-7)
        return lms[:, None, :, :]


class LSEPooling(nn.Module):
    """LSE pool: y = T * (logsumexp(x/T, dim) - log(N))"""

    def __init__(self, pool_axis: int = -1, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.pool_axis   = pool_axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temperature * (
            torch.logsumexp(x / self.temperature, axis=self.pool_axis)
            - math.log(x.shape[self.pool_axis])
        )


class LSEHead(nn.Module):
    """频率轴 mean → 时间轴 LSE pool → (B, N_CLS)."""

    def __init__(self, num_features: int, num_classes: int,
                 dropout: float = 0.2, lse_temperature: float = 1.0):
        super().__init__()
        self.lse_pool = LSEPooling(pool_axis=1, temperature=lse_temperature)
        self.cls_fc   = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, C, Freq, Time)
        h = torch.mean(h, axis=2)               # (B, C, Time)
        h = h.transpose(1, 2)                   # (B, Time, C)
        timewise_logits = self.cls_fc(h)        # (B, Time, N_CLS)
        return self.lse_pool(timewise_logits)   # (B, N_CLS)


class LSEModel(nn.Module):
    """timm backbone + LSEHead. 兼容 HGNetV2 / RepViT / EfficientNet 等 backbone."""

    def __init__(self, model_name: str = MODEL_NAME,
                 num_classes: int = N_CLASSES,
                 head_dropout: float = HEAD_DROPOUT,
                 lse_temperature: float = LSE_TEMPERATURE,
                 drop_path_rate: float = 0.0):
        super().__init__()
        # 部分 timm backbone (例如 RepViT) 不接受 drop_path_rate kwarg.
        # 先带着 drop_path_rate 试一次, 抛 TypeError 就回退到不带.
        common_kwargs = dict(
            pretrained=False, in_chans=1,
            global_pool="", num_classes=0,
        )
        try:
            self.backbone = timm.create_model(
                model_name, drop_path_rate=drop_path_rate, **common_kwargs,
            )
        except TypeError as e:
            if "drop_path_rate" not in str(e):
                raise
            self.backbone = timm.create_model(model_name, **common_kwargs)
        # dummy forward 拿真实通道数 (有些 backbone num_features 跟输出不一致)
        with torch.no_grad():
            dummy        = torch.randn(1, 1, 256, 256)
            num_features = self.backbone(dummy).shape[1]
        self.head = LSEHead(num_features, num_classes, head_dropout, lse_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)   # (B, C, Freq, Time)
        return self.head(h)    # (B, N_CLS)


# =============================================================================
# 3. 推理 helper
# =============================================================================

def _find_ckpts(ckpt_dir: Path, ext: str = "pt") -> tp.List[Path]:
    """按 fold 索引顺序找 4 个权重文件.

    Args:
        ckpt_dir: 权重所在目录
        ext: 'pt' 或 'onnx', 决定找 .pt 还是 .onnx
    """
    ckpt_dir = Path(ckpt_dir)
    paths    = []
    for k in range(N_FOLDS):
        cands = (list(ckpt_dir.rglob(f"best_model_fold{k}.{ext}"))
                 + list(ckpt_dir.rglob(f"*fold{k}*.{ext}")))
        cands = [p for p in cands if p.is_file()]
        if not cands:
            raise FileNotFoundError(
                f"fold {k} ckpt not found under {ckpt_dir} "
                f"(expect 'best_model_fold{k}.{ext}' or '*fold{k}*.{ext}')"
            )
        # 同名多个 → 取最新修改时间的
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        paths.append(cands[0])
    return paths


def _detect_backend(ckpt_dir: tp.Union[str, Path]) -> str:
    """检测 ckpt_dir 里是 .pt 还是 .onnx, 优先 .onnx (CPU 更快).

    返回 'onnx' / 'pt'.
    """
    p = Path(ckpt_dir)
    if list(p.rglob(f"best_model_fold0.onnx")) or list(p.rglob("*fold0*.onnx")):
        return "onnx"
    if list(p.rglob(f"best_model_fold0.pt")) or list(p.rglob("*fold0*.pt")):
        return "pt"
    raise FileNotFoundError(
        f"既没找到 .onnx 也没找到 .pt 在 {p} 下. "
        f"请确认 ckpt 目录里有 best_model_fold{{0..3}}.pt 或 best_model_fold{{0..3}}.onnx"
    )


def load_hgnet_models(ckpt_dir: tp.Union[str, Path],
                      device: torch.device,
                      model_name: tp.Optional[str] = None,
                      ) -> tp.Tuple[tp.List[nn.Module], LogMelSpectrogramTransform]:
    """加载 4 折权重 + 共享一个 mel transform.

    Args:
        ckpt_dir:   存放 best_model_fold{0..3}.pt 的目录
        device:     torch device
        model_name: timm backbone 名 (默认 None = 用 module 顶部的 MODEL_NAME).
                    传入这个参数可以避免 reload module 切 backbone, 方便多路 ensemble.
    """
    ckpt_paths = _find_ckpts(Path(ckpt_dir), ext="pt")
    backbone   = model_name or MODEL_NAME
    print(f"ckpts found ({backbone}):")
    for p in ckpt_paths:
        print(f"  {p}")

    models = []
    for p in ckpt_paths:
        m     = LSEModel(model_name=backbone)
        # PyTorch 2.4+ 默认 weights_only=True 会报错, 显式关掉
        try:
            state = torch.load(str(p), map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(str(p), map_location=device)
        m.load_state_dict(state, strict=True)
        m.eval().to(device)
        models.append(m)

    lms = LogMelSpectrogramTransform(
        MEL_SPECTROGRAM_PARAMS, TOP_DB, LMS_SHAPE,
    ).eval().to(device)
    return models, lms


def load_hgnet_onnx_sessions(ckpt_dir: tp.Union[str, Path],
                             intra_op_threads: int = 0,
                             ) -> tp.Tuple[tp.List, LogMelSpectrogramTransform]:
    """加载 4 折 ONNX 模型 (CPU) + 共享一个 mel transform.

    比 PyTorch CPU 推理快 2-3 倍, 适合 Kaggle CPU only 提交.

    Args:
        ckpt_dir:         存放 best_model_fold{0..3}.onnx 的目录
        intra_op_threads: ORT intra-op 线程数, 0 = 让 ORT 自己决定
                          (Kaggle 通常 4 核, 0 一般会用满)
    """
    import onnxruntime as ort

    onnx_paths = _find_ckpts(Path(ckpt_dir), ext="onnx")
    print("onnx ckpts found:")
    for p in onnx_paths:
        print(f"  {p}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads > 0:
        sess_options.intra_op_num_threads = intra_op_threads

    sessions = []
    for p in onnx_paths:
        sess = ort.InferenceSession(
            str(p), sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        sessions.append(sess)

    # log-mel 留在 CPU torch 上算, 跟 PyTorch 后端走一样的 transform.
    lms = LogMelSpectrogramTransform(
        MEL_SPECTROGRAM_PARAMS, TOP_DB, LMS_SHAPE,
    ).eval()
    return sessions, lms


def _read_60s_wav(path: Path, target_sr: int = SR) -> np.ndarray:
    """读 60s wav, 不足 0-pad, 多余截断, 立体声转 mono. 返回 (60*SR,) float32."""
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != target_sr:
        y = librosa.resample(y, orig_sr=sr0, target_sr=target_sr)
    n = 60 * target_sr
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]
    return y.astype(np.float32)


# =============================================================================
# 4. 主推理函数
# =============================================================================

@torch.no_grad()
def run_hgnet_inference(
    test_paths     : tp.Sequence[tp.Union[str, Path]],
    ckpt_dir       : tp.Union[str, Path],
    primary_labels : tp.Sequence[str],
    out_csv        : tp.Union[str, Path] = "submission_hgnet.csv",
    device         : tp.Optional[torch.device] = None,
    smooth_sigma   : float = 0.65,
    verbose        : bool  = True,
    model_name     : tp.Optional[str] = None,
    backend        : tp.Optional[str] = None,
) -> pd.DataFrame:
    """
    跑 4-fold ensemble 推理, 输出与 0.949 row_id 格式对齐的 csv.

    后端自动检测:
      - ckpt_dir 里有 .onnx 文件 -> 走 ONNXRuntime CPU (推荐, CPU 上快 2-3x);
      - 否则走 PyTorch (.pt). 也可以用 ``backend='pt' / 'onnx'`` 强制指定.

    Args:
      test_paths:     60s .ogg/.wav 文件路径列表
      ckpt_dir:       存放 best_model_fold{0..3}.pt 或 .onnx 的目录
      primary_labels: 234 个 primary_label, 顺序必须跟 sample_submission.csv 一致
      out_csv:        输出 csv 路径
      device:         None = 自动选 cuda / cpu (仅 .pt 后端用; ONNX 走 CPU)
      smooth_sigma:   时间维 Gaussian 平滑, <=0 时跳过
      verbose:        是否打印进度
      model_name:     timm backbone 名 (仅 .pt 后端用; ONNX 已经把架构烘进图)
      backend:        'pt' / 'onnx' / None (None=自动检测)

    Returns:
      sub_df: DataFrame, columns = ['row_id', *primary_labels],
              shape = (len(test_paths) * 12, 235), prob ∈ [0, 1]
    """
    assert len(primary_labels) == N_CLASSES, \
        f"primary_labels 应该是 {N_CLASSES} 类, 实际 {len(primary_labels)}"

    if backend is None:
        backend = _detect_backend(ckpt_dir)
    assert backend in {"pt", "onnx"}, f"backend must be 'pt' / 'onnx', got {backend}"

    if backend == "onnx":
        sessions, lms = load_hgnet_onnx_sessions(ckpt_dir)
        infer_device  = "cpu (onnxruntime)"
        n_folds       = len(sessions)
        # 取第一个 session 的输入名, 4 折 ONNX 用同样的 export 流程, 名字必相同
        input_name    = sessions[0].get_inputs()[0].name
        output_name   = sessions[0].get_outputs()[0].name
    else:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models, lms   = load_hgnet_models(ckpt_dir, device, model_name=model_name)
        infer_device  = str(device)
        n_folds       = len(models)

    print(f"backend: {backend}  |  device: {infer_device}  |  "
          f"{n_folds} folds loaded ({model_name or MODEL_NAME})")

    rows  : tp.List[str]        = []
    preds : tp.List[np.ndarray] = []

    n_total = len(test_paths)
    for i, path in enumerate(test_paths, 1):
        path = Path(path)
        y    = _read_60s_wav(path, SR)
        # (12, 160000) → tensor on CPU (ONNX 后端) 或 device (PyTorch 后端)
        chunks = torch.from_numpy(y.reshape(N_WINDOWS, WINDOW_SAMPLES))
        if backend == "pt":
            chunks = chunks.to(device)
        mel = lms(chunks)                                # (12, 1, 256, 256)

        # 4-fold ensemble in prob space (sigmoid 后平均)
        if backend == "onnx":
            mel_np   = mel.numpy().astype(np.float32, copy=False)
            prob_sum = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
            for sess in sessions:
                logits = sess.run([output_name], {input_name: mel_np})[0]  # (12, N_CLS)
                prob_sum += 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
            prob = prob_sum / float(n_folds)
        else:
            prob_sum = torch.zeros(N_WINDOWS, N_CLASSES, device=device)
            for m in models:
                prob_sum += torch.sigmoid(m(mel))
            prob = (prob_sum / n_folds).cpu().numpy().astype(np.float32)

        # 时间维 Gaussian 平滑
        if smooth_sigma and smooth_sigma > 0 and N_WINDOWS > 1:
            prob = gaussian_filter1d(prob, sigma=smooth_sigma, axis=0,
                                     mode="nearest").astype(np.float32)
        prob = np.clip(prob, 0.0, 1.0)

        stem = path.stem
        rows.extend([f"{stem}_{(w + 1) * WINDOW_SEC}" for w in range(N_WINDOWS)])
        preds.append(prob)

        if verbose and (i == 1 or i % 50 == 0 or i == n_total):
            print(f"HGNet[{backend}]: {i}/{n_total}")

    arr = np.concatenate(preds, axis=0)
    sub = pd.DataFrame(arr, columns=list(primary_labels))
    sub.insert(0, "row_id", rows)
    sub.to_csv(out_csv, index=False)
    print(f"HGNet inference done. Saved → {out_csv}  shape={sub.shape}")

    # 释放内存
    if backend == "onnx":
        del sessions
    else:
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del lms
    gc.collect()

    return sub


# =============================================================================
# 4b. 多 backbone ONNX ensemble 推理 (mel 只算一次, 喂给所有 session)
# =============================================================================

@torch.no_grad()
def run_onnx_ensemble_inference(
    test_paths     : tp.Sequence[tp.Union[str, Path]],
    onnx_dirs      : tp.Sequence[tp.Union[str, Path]],
    primary_labels : tp.Sequence[str],
    out_csv        : tp.Union[str, Path] = "submission_hgnet.csv",
    dir_weights    : tp.Optional[tp.Sequence[float]] = None,
    smooth_sigma   : float = 0.65,
    intra_op_threads: int  = 0,
    verbose        : bool  = True,
) -> pd.DataFrame:
    """多 backbone ONNX ensemble: 每个 dir 含 ``best_model_fold{0..3}.onnx``.

    所有 backbone 都是同一个 ``LSEModel`` 导出 (输入 (B,1,256,256), 输出
    (B,234)), mel transform 在 ONNX 图外, 因此 **mel 只算一次** 喂给全部
    session, N 个 backbone × M 折只增加 ONNX 前向开销, 不重复读音频/算 mel.

    融合策略:
      * 每个 dir 内部 folds 先在 prob 空间求均值 → per-backbone prob;
      * 再按 ``dir_weights`` 跨 backbone 加权平均 (默认等权), 这样 fold 数
        不同的 backbone 也是等话语权.

    Args:
        test_paths:      60s .ogg/.wav 路径列表
        onnx_dirs:       目录列表, 每个含 best_model_fold{0..3}.onnx
        primary_labels:  234 个 label, 顺序跟 sample_submission.csv 一致
        out_csv:         输出 csv
        dir_weights:     跨 backbone 权重 (默认 None=等权), 长度需等于 onnx_dirs
        smooth_sigma:    时间维 Gaussian 平滑, <=0 跳过
        intra_op_threads: ORT intra-op 线程数, 0=让 ORT 自己决定
        verbose:         打印进度

    Returns:
        sub_df: columns = ['row_id', *primary_labels],
                shape = (len(test_paths)*12, 235), prob ∈ [0, 1]
    """
    import onnxruntime as ort

    assert len(primary_labels) == N_CLASSES, \
        f"primary_labels 应该是 {N_CLASSES} 类, 实际 {len(primary_labels)}"
    onnx_dirs = [Path(d) for d in onnx_dirs]
    if not onnx_dirs:
        raise ValueError("onnx_dirs 不能为空")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads > 0:
        so.intra_op_num_threads = intra_op_threads

    # 每个 backbone 一组 sessions
    groups: tp.List[dict] = []
    for d in onnx_dirs:
        ckpt_paths = _find_ckpts(d, ext="onnx")
        sessions   = []
        for p in ckpt_paths:
            sess = ort.InferenceSession(
                str(p), sess_options=so, providers=["CPUExecutionProvider"],
            )
            sessions.append(
                (sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name)
            )
        groups.append({"name": d.name, "sessions": sessions})
        print(f"[hgnet-ens] {d.name}: {len(sessions)} folds")

    n_groups = len(groups)
    if dir_weights is None:
        dir_weights = [1.0 / n_groups] * n_groups
    else:
        assert len(dir_weights) == n_groups, \
            f"dir_weights 长度 {len(dir_weights)} != backbone 数 {n_groups}"
        s = float(sum(dir_weights))
        assert s > 0, "dir_weights 和必须 > 0"
        dir_weights = [float(w) / s for w in dir_weights]
    print(f"[hgnet-ens] {n_groups} backbones={[g['name'] for g in groups]}  "
          f"weights={[round(w, 4) for w in dir_weights]}")

    # log-mel 留在 CPU torch 上算, 跟单 backbone ONNX 后端走一样的 transform
    lms = LogMelSpectrogramTransform(
        MEL_SPECTROGRAM_PARAMS, TOP_DB, LMS_SHAPE,
    ).eval()

    rows : tp.List[str]        = []
    preds: tp.List[np.ndarray] = []
    n_total = len(test_paths)

    for i, path in enumerate(test_paths, 1):
        path   = Path(path)
        y      = _read_60s_wav(path, SR)
        chunks = torch.from_numpy(y.reshape(N_WINDOWS, WINDOW_SAMPLES))
        mel_np = lms(chunks).numpy().astype(np.float32, copy=False)  # (12,1,256,256)

        prob = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
        for g, gw in zip(groups, dir_weights):
            grp_sum = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
            for sess, in_name, out_name in g["sessions"]:
                logits   = sess.run([out_name], {in_name: mel_np})[0]
                grp_sum += 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
            prob += gw * (grp_sum / float(len(g["sessions"])))

        if smooth_sigma and smooth_sigma > 0 and N_WINDOWS > 1:
            prob = gaussian_filter1d(
                prob, sigma=smooth_sigma, axis=0, mode="nearest",
            ).astype(np.float32)
        prob = np.clip(prob, 0.0, 1.0)

        stem = path.stem
        rows.extend([f"{stem}_{(wi + 1) * WINDOW_SEC}" for wi in range(N_WINDOWS)])
        preds.append(prob)

        if verbose and (i == 1 or i % 50 == 0 or i == n_total):
            print(f"[hgnet-ens]: {i}/{n_total}")

    arr = np.concatenate(preds, axis=0)
    sub = pd.DataFrame(arr, columns=list(primary_labels))
    sub.insert(0, "row_id", rows)
    sub.to_csv(out_csv, index=False)
    print(f"[hgnet-ens] done. Saved → {out_csv}  shape={sub.shape}")

    del groups
    gc.collect()
    return sub


# =============================================================================
# 5. 融合函数
# =============================================================================

def blend_with_hgnet(
    base_csv   : tp.Union[str, Path],
    hgnet_csv  : tp.Union[str, Path],
    out_csv    : tp.Union[str, Path] = "submission.csv",
    w_base     : float = 0.95,
    w_hgnet    : float = 0.05,
    mode       : str   = "rank",                       # "rank" or "prob"
) -> pd.DataFrame:
    """
    把 HGNet 的 csv 跟 0.949 的最终 csv 做 2-way 融合.

    Args:
      base_csv:  0.949 方案输出的 submission.csv (主分支)
      hgnet_csv: run_hgnet_inference 输出的 submission_hgnet.csv
      out_csv:   融合后的 csv (可以跟 base_csv 同名, 覆盖原文件)
      w_base:    主分支权重 (默认 0.95)
      w_hgnet:   HGNet 权重 (默认 0.05)
      mode:      "rank" = rank-percentile 加权 (推荐, 抗 scale 差异);
                 "prob" = 直接 prob 线性加权

    Returns:
      sub: 融合后的 DataFrame
    """
    assert mode in {"rank", "prob"}, f"mode must be 'rank' or 'prob', got {mode}"
    assert w_base + w_hgnet > 0, "权重和必须 > 0"

    base_csv, hgnet_csv = Path(base_csv), Path(hgnet_csv)
    df_base  = pd.read_csv(base_csv)
    df_hgnet = pd.read_csv(hgnet_csv)

    # 1) 按 base 的 row_id 顺序对齐 hgnet
    if "row_id" not in df_base.columns or "row_id" not in df_hgnet.columns:
        raise ValueError("两个 csv 都必须包含 'row_id' 列")
    missing = set(df_base["row_id"]) - set(df_hgnet["row_id"])
    if missing:
        # Dry-run / Save&Run All 阶段 hidden test 没挂载,
        # base 走 sample_submission stub (3 行), HGNet 走 train_soundscapes
        # dry-run (240 行), row_id 完全不重叠. 这种情况直接退化成只用 base,
        # 让 commit 阶段能产出 submission.csv. 真打分时 row_id 都对齐 sample,
        # 不会进这里.
        addon_rows = set(df_hgnet["row_id"])
        base_rows  = set(df_base["row_id"])
        if base_rows.isdisjoint(addon_rows):
            print(
                f"[blend_with_hgnet] WARN: row_id 与 hgnet 完全不重叠 "
                f"(base={len(base_rows)}, hgnet={len(addon_rows)}). "
                f"Dry-run 期间正常, 真打分时不会出现. 直接写出 base."
            )
            df_base.to_csv(out_csv, index=False)
            return df_base
        raise ValueError(
            f"hgnet_csv 缺少 {len(missing)} 个 row_id (例: {list(missing)[:3]})"
        )
    df_hgnet = (df_hgnet.set_index("row_id")
                          .loc[df_base["row_id"]]
                          .reset_index())

    # 2) 取出 234 个类别列 (跳过 row_id)
    cols = [c for c in df_base.columns if c != "row_id"]
    assert len(cols) == N_CLASSES, \
        f"base_csv 应有 {N_CLASSES} 类别列, 实际 {len(cols)}"
    assert all(c in df_hgnet.columns for c in cols), \
        "hgnet_csv 与 base_csv 类别列不匹配"

    eps    = 1e-5
    p_base  = np.clip(df_base[cols].to_numpy(np.float32),  eps, 1.0 - eps)
    p_hgnet = np.clip(df_hgnet[cols].to_numpy(np.float32), eps, 1.0 - eps)

    # 3) rank 或 prob 加权融合
    wb, wh = float(w_base), float(w_hgnet)
    s      = wb + wh
    wb, wh = wb / s, wh / s

    if mode == "rank":
        r_base  = pd.DataFrame(p_base).rank(axis=0, pct=True).to_numpy(np.float32)
        r_hgnet = pd.DataFrame(p_hgnet).rank(axis=0, pct=True).to_numpy(np.float32)
        pred    = wb * r_base + wh * r_hgnet
    else:
        pred    = wb * p_base + wh * p_hgnet

    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)

    # 4) 写出
    sub = df_base.copy()
    sub[cols] = pred
    sub.to_csv(out_csv, index=False)
    print(f"Blend done. base={wb:.2f} hgnet={wh:.2f} mode={mode}  "
          f"→ {out_csv}  shape={sub.shape}")
    return sub


# =============================================================================
# 6. CLI 入口 (本地调试用)
# =============================================================================

def _cli():
    parser = argparse.ArgumentParser(description="HGNet 4-fold ensemble 推理 + 与 0.949 融合")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # infer 子命令
    p_inf = sub.add_parser("infer", help="跑 HGNet 4-fold 推理")
    p_inf.add_argument("--ckpt_dir",   required=True, help="存放 best_model_fold{0..3}.pt 的目录")
    p_inf.add_argument("--test_dir",   required=True, help="60s .ogg 文件所在目录")
    p_inf.add_argument("--sample_sub", required=True,
                        help="sample_submission.csv 路径 (用来拿 primary_labels 顺序)")
    p_inf.add_argument("--out",        default="submission_hgnet.csv", help="输出 csv 路径")
    p_inf.add_argument("--smooth_sigma", type=float, default=0.65)
    p_inf.add_argument("--device",     default=None, help="'cuda' / 'cpu' / None")

    # blend 子命令
    p_bld = sub.add_parser("blend", help="融合 0.949 最终 csv + HGNet csv")
    p_bld.add_argument("--base",    required=True, help="0.949 输出的 submission.csv")
    p_bld.add_argument("--hgnet",   required=True, help="HGNet 输出的 submission_hgnet.csv")
    p_bld.add_argument("--out",     default="submission.csv")
    p_bld.add_argument("--w_base",  type=float, default=0.95)
    p_bld.add_argument("--w_hgnet", type=float, default=0.05)
    p_bld.add_argument("--mode",    default="rank", choices=["rank", "prob"])

    args = parser.parse_args()

    if args.cmd == "infer":
        sample_sub = pd.read_csv(args.sample_sub)
        primary_labels = [c for c in sample_sub.columns if c != "row_id"]
        # 找所有 .ogg 文件
        test_paths = sorted(Path(args.test_dir).glob("*.ogg"))
        if not test_paths:
            test_paths = sorted(Path(args.test_dir).glob("*.wav"))
        if not test_paths:
            raise FileNotFoundError(f"No .ogg/.wav under {args.test_dir}")
        print(f"Found {len(test_paths)} test files in {args.test_dir}")

        device = torch.device(args.device) if args.device else None
        run_hgnet_inference(
            test_paths=test_paths,
            ckpt_dir=args.ckpt_dir,
            primary_labels=primary_labels,
            out_csv=args.out,
            device=device,
            smooth_sigma=args.smooth_sigma,
        )

    elif args.cmd == "blend":
        blend_with_hgnet(
            base_csv=args.base,
            hgnet_csv=args.hgnet,
            out_csv=args.out,
            w_base=args.w_base,
            w_hgnet=args.w_hgnet,
            mode=args.mode,
        )


if __name__ == "__main__":
    _cli()
