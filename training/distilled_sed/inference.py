# -*- coding: utf-8 -*-
"""
inference.py — 推理流水线 + 写 submission.csv
==============================================

对应 notebook 末尾 4 个 inference cells:
  - mel 用 librosa (推理时不依赖 torchaudio / torch)
  - 加载 5 个 fold 的 ONNX session
  - 对每个 60s test 文件:
       1. 切 12 段 5s, 算 mel (B, 1, 128, 313)
       2. 每个 fold 跑 ONNX → clip_logits + framewise_logits
       3. fold 内: 0.5 * clip + 0.5 * frame_max (logit 空间)
       4. 5 fold 平均 (logit 空间)
       5. 沿 12 个 window 高斯平滑 (logit 空间)
       6. sigmoid 一次, 写 CSV

★ 跟其他推理脚本不同的 2 个细节:
  - 所有融合在 logit 空间做 (最后 sigmoid 一次), 不是每步 sigmoid
  - 高斯 kernel = [0.1, 0.2, 0.4, 0.2, 0.1] (5-tap, 比 sigma=0.65 略宽)
"""
import os
import re
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
from scipy.ndimage import convolve1d

from config import (TEST_DIR, COMP_DIR, OUT_DIR, NUM_CLASSES,
                    INF_SR, INF_N_MELS, INF_N_FFT, INF_HOP, INF_FMIN, INF_FMAX,
                    INF_TOP_DB, INF_CHUNK_S, INF_CHUNK_N, N_WINDOWS)


# =============================================================================
# 1. 推理用的 mel (librosa, 不依赖 torchaudio)
# =============================================================================
# Kaggle 离线推理时希望尽量少依赖, librosa 比 torchaudio 轻量.
# 输出 (B, 1, n_mels, T) 跟训练时一致.

def audio_to_mel(chunks):
    """
    Args:
      chunks: (N, INF_CHUNK_N) 原始音频段
    Returns:
      (N, 1, INF_N_MELS, T) float32 normalized mel dB
    """
    mels = []
    for i in range(chunks.shape[0]):
        S = librosa.feature.melspectrogram(
            y=chunks[i], sr=INF_SR, n_fft=INF_N_FFT, hop_length=INF_HOP,
            n_mels=INF_N_MELS, fmin=INF_FMIN, fmax=INF_FMAX, power=2.0,
        )
        S_dB = librosa.power_to_db(S, top_db=INF_TOP_DB)
        # per-clip 归一化 (跟训练时 _predict_from_waveforms 一致)
        S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)
        mels.append(S_dB)
    return np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)


# =============================================================================
# 2. ONNX session 管理
# =============================================================================

def discover_folds(sed_dir):
    """搜目录里所有 sed_fold{N}.onnx, 返回排好序的 N 列表."""
    pat = re.compile(r"sed_fold(\d+)\.onnx$")
    folds = []
    for fname in os.listdir(sed_dir):
        m = pat.match(fname)
        if m:
            folds.append(int(m.group(1)))
    return sorted(folds)


def make_session(onnx_path):
    """
    创建优化好的 ONNX session.
      - 4 线程 intra-op
      - 启用全部图优化
      - 优先 CUDA, fallback CPU
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def load_sed_sessions(sed_dir_candidates=None):
    """
    在多个候选路径里找 SED ONNX, 加载全部 fold session.

    Args:
      sed_dir_candidates: list of Path, 默认 [OUT_DIR, Kaggle 公开 dataset 路径]
    Returns:
      (fold_sessions, fold_ids, sed_dir)
    """
    if sed_dir_candidates is None:
        sed_dir_candidates = [
            str(OUT_DIR),
            "/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public",
        ]

    sed_dir = next(
        (p for p in sed_dir_candidates
         if os.path.isdir(p)
         and any(f.endswith(".onnx") and "sed" in f for f in os.listdir(p))),
        None,
    )
    assert sed_dir, f"No SED ONNX files found in {sed_dir_candidates}"

    fold_ids = discover_folds(sed_dir)
    assert fold_ids, f"No sed_fold*.onnx in {sed_dir}"
    print(f"Found {len(fold_ids)} fold(s) in {sed_dir}: {fold_ids}")

    sessions = []
    for fold in fold_ids:
        p = f"{sed_dir}/sed_fold{fold}.onnx"
        sess = make_session(p)
        sessions.append(sess)
        size_mb = os.path.getsize(p) / 1e6
        print(f"  fold {fold}: {size_mb:5.1f}MB  providers={sess.get_providers()}")
    return sessions, fold_ids, sed_dir


# =============================================================================
# 3. 音频加载 (60s 文件 → 12 个 5s chunk)
# =============================================================================
# 推理时音频用 soundfile (快), 缺了就 fallback librosa.

try:
    import soundfile as sf
    DECODER = "soundfile"
except ImportError:
    DECODER = "librosa"


def load_audio_32k_mono(path):
    """加载 ogg, 转 32 kHz 单声道 float32."""
    if DECODER == "soundfile":
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != INF_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=INF_SR)
        return wav.astype(np.float32)
    else:
        wav, _ = librosa.load(path, sr=INF_SR, mono=True)
        return wav.astype(np.float32)


def file_to_chunks(path):
    """
    60s 文件 → 12 个 5s chunks (numpy) + 12 个 end_time.
    短于 60s 补零, 长于 60s 截断.
    """
    wav = load_audio_32k_mono(path)
    target_len = 60 * INF_SR
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    elif len(wav) > target_len:
        wav = wav[:target_len]
    n_chunks = target_len // INF_CHUNK_N
    chunks = wav[:n_chunks * INF_CHUNK_N].reshape(n_chunks, INF_CHUNK_N)
    end_times = np.arange(1, n_chunks + 1) * INF_CHUNK_S            # [5, 10, ..., 60]
    return chunks.astype(np.float32), end_times


# =============================================================================
# 4. 推理工具: 数值稳定 sigmoid + 高斯平滑
# =============================================================================

def sigmoid_inf(x):
    """
    数值稳定的 sigmoid:
      x ≥ 0: 标准 1 / (1 + exp(-x))
      x < 0: exp(x) / (1 + exp(x))   (避免 exp(-x) 爆炸)
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
        np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))),
    ).astype(np.float32)


# 5-tap 高斯 kernel (比 sigma=0.65 稍宽, 适合 12 个 window)
GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])


def gauss_smooth_final(scores, weights=GAUSSIAN_KERNEL):
    """
    沿每个文件的 12 个 window 高斯平滑.
    输入: (n_files * 12, n_classes)
    输出: (n_files * 12, n_classes)
    """
    smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
    for i in range(smoothed.shape[0]):
        smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode="nearest")
    return smoothed.reshape(-1, scores.shape[1])


# =============================================================================
# 5. 主推理 loop
# =============================================================================

def run_inference(test_files, fold_sessions, fold_ids):
    """
    跑 test 文件 → 返回 row_ids list + probs array.

    流程:
      对每个 60s 文件:
        1. 切 12 段 5s, 算 mel (12, 1, 128, 313)
        2. 每个 fold 跑 ONNX → (clip_logits, framewise_logits)
        3. 在 logit 空间累积: 0.5 * clip + 0.5 * frame.max(axis=time)
        4. 5 fold 平均 (logit 空间)
        5. 沿 12 window 高斯平滑 (logit 空间)
        6. sigmoid 一次得 prob

    Args:
      test_files:    list of test ogg paths
      fold_sessions: list of ONNX sessions
      fold_ids:      list of fold int (跟 sessions 顺序一致)
    Returns:
      all_rows:      list of row_id 字符串 (n_files*12,)
      all_preds_arr: (n_files*12, n_classes) float32 概率
    """
    t0 = time.time()
    all_rows, all_preds = [], []

    for file_idx, file_path in enumerate(test_files):
        basename = os.path.basename(file_path).replace(".ogg", "")
        chunks, end_times = file_to_chunks(file_path)
        mel = audio_to_mel(chunks)                          # (12, 1, 128, 313)

        # 5 fold 在 logit 空间累积
        logits_sum = np.zeros((chunks.shape[0], NUM_CLASSES), dtype=np.float32)
        for sess in fold_sessions:
            outs        = sess.run(None, {"mel": mel})
            clip_logits = outs[0]                           # (12, 234)
            frame_max   = outs[1].max(axis=1)               # (12, 234) frame 维 max
            logits_sum += 0.5 * clip_logits + 0.5 * frame_max
        logits_mean = logits_sum / len(fold_ids)

        # 沿 12 window 高斯平滑 (logit 空间), 再 sigmoid 一次
        logits_smoothed = gauss_smooth_final(logits_mean)
        probs           = sigmoid_inf(logits_smoothed)

        all_rows.extend([f"{basename}_{int(t)}" for t in end_times])
        all_preds.append(probs)

        # 进度打印
        if (file_idx + 1) % 50 == 0 or file_idx == 0 or file_idx == len(test_files) - 1:
            elapsed = time.time() - t0
            rate    = (file_idx + 1) / elapsed
            print(f"  [{file_idx+1:4d}/{len(test_files)}] "
                  f"{elapsed:.1f}s  {rate:.2f} files/s")

    all_preds_arr = (np.concatenate(all_preds) if all_preds
                      else np.zeros((0, NUM_CLASSES), np.float32))
    print(f"\nInference: {len(all_rows)} rows, {time.time()-t0:.1f}s total")
    return all_rows, all_preds_arr


def find_test_files(debug_n=5):
    """
    搜 test 文件. 没找到就用 train_soundscapes 前 N 个 fallback (本地调试用).
    """
    test_files = sorted(glob.glob(f"{TEST_DIR}/*.ogg")) if TEST_DIR.is_dir() else []
    if len(test_files) == 0:
        fallback = COMP_DIR / "train_soundscapes"
        if fallback.is_dir():
            test_files = sorted(glob.glob(f"{fallback}/*.ogg"))[:debug_n]
            print(f"No test_soundscapes -- using {len(test_files)} train files for debug")
    print(f"Test files: {len(test_files)}")
    return test_files


def write_submission(all_rows, all_preds_arr, PRIMARY_LABELS,
                     out_csv="submission.csv"):
    """写最终 submission.csv (row_id + 234 物种列)."""
    submission = pd.DataFrame(all_preds_arr, columns=PRIMARY_LABELS)
    submission.insert(0, "row_id", all_rows)

    # 基础校验
    assert submission.shape[1] == NUM_CLASSES + 1
    assert submission["row_id"].is_unique
    assert not submission.iloc[:, 1:].isna().any().any()
    submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(0.0, 1.0)

    submission.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}: {len(submission)} rows x {submission.shape[1]} cols")
    print(submission.head(3).iloc[:, :6])
    return submission
