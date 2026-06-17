# -*- coding: utf-8 -*-
"""sed_pipeline.py — Distilled SED multi-fold ONNX → ``submission_sed.csv``.

对应 cell_07 第 1326-1417 行: 找 ``sed_fold*.onnx``, 跑每个 fold 输出
``0.5 * sigmoid(clip_logits) + 0.5 * sigmoid(frame_max)``, 多 fold 求 mean,
最后 sigma=0.65 高斯平滑.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

from .config import (
    FILE_SAMPLES, INPUT_ROOT, N_WINDOWS, SED_CFG, SR, WINDOW_SAMPLES, WINDOW_SEC,
)


def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


def find_sed_dir() -> Path:
    """优先级:
    1. ``BIRDCLEF_SED_DIR``  显式指定包含 ``sed_fold*.onnx`` 的目录
    2. ``INPUT_ROOT.rglob('sed_fold0.onnx')``  自动探测 (sorted, 拿第一个)
    """
    env_dir = os.environ.get("BIRDCLEF_SED_DIR")
    if env_dir:
        p = Path(env_dir)
        if (p / "sed_fold0.onnx").exists():
            return p
        # 容忍传成上一级目录的写法
        hits = sorted(p.rglob("sed_fold0.onnx")) if p.exists() else []
        if hits:
            return hits[0].parent
        raise FileNotFoundError(
            f"BIRDCLEF_SED_DIR={env_dir} but sed_fold0.onnx not found inside."
        )

    hits = sorted(INPUT_ROOT.rglob("sed_fold0.onnx")) if INPUT_ROOT.exists() else []
    if not hits:
        raise FileNotFoundError(
            "sed_fold0.onnx not found. Set BIRDCLEF_SED_DIR or attach the SED ONNX dataset."
        )
    return hits[0].parent


def make_sed_session(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads     = 4
    so.inter_op_num_threads     = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def audio_to_mel(chunks: np.ndarray) -> np.ndarray:
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR,
            n_fft   = SED_CFG["n_fft"],
            hop_length = SED_CFG["hop"],
            n_mels  = SED_CFG["n_mels"],
            fmin    = SED_CFG["fmin"],
            fmax    = SED_CFG["fmax"],
            power=2.0,
        )
        s = librosa.power_to_db(s, top_db=SED_CFG["top_db"])
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def file_to_sed_chunks(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    ends   = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC
    return chunks, ends


def run_sed_pipeline(
    test_paths: List[Path],
    primary_labels: List[str],
    n_classes: int,
    output_csv: str = "submission_sed.csv",
) -> Path:
    sed_dir = find_sed_dir()
    sed_fold_paths = sorted(
        sed_dir.glob("sed_fold*.onnx"),
        key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)),
    )
    sed_sessions = [make_sed_session(p) for p in sed_fold_paths]
    print(f"[sed] dir={sed_dir}  folds={[p.name for p in sed_fold_paths]}")

    sed_rows: List[str]      = []
    sed_preds: List[np.ndarray] = []

    for i, path in enumerate(test_paths, 1):
        chunks, ends = file_to_sed_chunks(path)
        mel          = audio_to_mel(chunks)
        p_sum        = np.zeros((len(chunks), n_classes), dtype=np.float32)

        for sess in sed_sessions:
            outs        = sess.run(None, {sess.get_inputs()[0].name: mel})
            clip_logits = outs[0]
            frame_max   = outs[1].max(axis=1)
            p_sum += 0.5 * _sigmoid(clip_logits) + 0.5 * _sigmoid(frame_max)

        p_mean = p_sum / len(sed_sessions)
        if len(p_mean) > 1:
            p_mean = gaussian_filter1d(
                p_mean, sigma=SED_CFG["smooth_sigma"], axis=0, mode="nearest",
            ).astype(np.float32)

        stem = path.stem
        sed_rows.extend([f"{stem}_{int(t)}" for t in ends])
        sed_preds.append(p_mean)

        if i == 1 or i % 50 == 0 or i == len(test_paths):
            print(f"[sed] {i}/{len(test_paths)}")

    sed_preds_arr = np.concatenate(sed_preds, axis=0)
    sub = pd.DataFrame(np.clip(sed_preds_arr, 0.0, 1.0), columns=primary_labels)
    sub.insert(0, "row_id", sed_rows)
    sub.to_csv(output_csv, index=False)
    print(f"[sed] saved {output_csv}  shape={sub.shape}")
    return Path(output_csv)
