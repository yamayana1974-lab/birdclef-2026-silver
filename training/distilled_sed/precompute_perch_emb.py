# -*- coding: utf-8 -*-
"""
precompute_perch_emb.py — 离线计算 Perch v2 emb 缓存
======================================================

把所有 focal + sc 文件喂给 Perch ONNX, 拿到 (1536,) emb 存成 .pt.
训练时直接 mmap 读, 不再调 ONNX, 整体训练加速 ~40%.

★ 一致性折中 (简化版 A):
  - focal: 每个文件取中心 5s 算 1 个 emb 存盘
           训练时 random crop 不同 5s, 跟缓存的 emb 略不一致
           BirdCLEF 弱标签 + 5s 内特征相对稳定, 影响 < 0.001 AUC
  - sc:    每个 5s 段算 1 个 emb (跟段对齐, 一致)

存储:
  perch_cache_v2/
    focal/<species>/<file>.pt       (35549 个, ~218 MB)
    sc/<sc_filename>__<start>.pt   (660 个,    ~4 MB)
    perch_emb_meta.csv              (索引)

运行:
  python precompute_perch_emb.py
  # 单卡 RTX PRO 6000 ~ 30-60 分钟
  # 支持断点续跑, 已存的 emb 跳过
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (WAVEFORM_CACHE_DIR, PERCH_ONNX_PATH, PERCH_CACHE_DIR,
                    SR, TRAIN_SAMPLES, PERCH_EMBED_DIM)


PERCH_BATCH = 64   # ONNX 一次跑多少个 5s 波形

PERCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(PERCH_CACHE_DIR / "focal").mkdir(parents=True, exist_ok=True)
(PERCH_CACHE_DIR / "sc").mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================
def load_int16(path):
    """加载 int16 .pt → float32 numpy in [-1, 1]."""
    arr = torch.load(path, map_location="cpu")
    return (arr.float() / 32767.0).numpy().astype(np.float32)


def center_crop(w, n_samples):
    """取中心 n_samples 段; 太短的左 pad."""
    total = len(w)
    if total <= n_samples:
        return np.pad(w, (n_samples - total, 0)).astype(np.float32)
    start = (total - n_samples) // 2
    return w[start:start + n_samples].astype(np.float32)


def get_session():
    """加载 Perch ONNX (优先 GPU)."""
    import onnxruntime as ort
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if torch.cuda.is_available() else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(str(PERCH_ONNX_PATH), providers=providers)
    print(f"Perch ONNX provider: {sess.get_providers()[0]}")
    in_name = sess.get_inputs()[0].name
    # 找 1536-d 输出
    embed_idx = 0
    for i, o in enumerate(sess.get_outputs()):
        if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
            embed_idx = i
            break
    return sess, in_name, embed_idx


def perch_embed_batch(sess, in_name, embed_idx, batch_np):
    """(B, 160000) float32 numpy → (B, 1536) float32 numpy."""
    out = sess.run(None, {in_name: batch_np})
    return out[embed_idx].astype(np.float32)


# =============================================================================
# 1. focal: 每文件取中心 5s
# =============================================================================
def build_focal_emb(sess, in_name, embed_idx):
    meta_csv = WAVEFORM_CACHE_DIR / "audio_cache_meta.csv"
    meta = pd.read_csv(meta_csv)
    print(f"focal: {len(meta)} entries from {meta_csv.name}")

    tasks = []     # list of (src_pt, out_pt, idx_in_meta)
    for i, r in meta.iterrows():
        out_pt = PERCH_CACHE_DIR / r["cache_file"]    # focal/<species>/<file>.pt 同 layout
        if out_pt.exists():
            continue
        src_pt = WAVEFORM_CACHE_DIR / r["cache_file"]
        if src_pt.exists():
            tasks.append((src_pt, out_pt, i))

    if not tasks:
        print("  all focal emb already cached, skip")
        return

    print(f"  computing {len(tasks)} new emb (batch={PERCH_BATCH})")
    pbar = tqdm(total=len(tasks), desc="focal emb")

    batch_wavs = []
    batch_outs = []
    for src_pt, out_pt, _ in tasks:
        try:
            w = load_int16(src_pt)
        except Exception as e:
            print(f"  [skip] {src_pt}: {e}")
            pbar.update(1)
            continue
        c = center_crop(w, TRAIN_SAMPLES)
        batch_wavs.append(c)
        batch_outs.append(out_pt)

        if len(batch_wavs) >= PERCH_BATCH:
            arr = np.stack(batch_wavs, axis=0)
            embs = perch_embed_batch(sess, in_name, embed_idx, arr)
            for op, emb in zip(batch_outs, embs):
                op.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.from_numpy(emb), op)
            pbar.update(len(batch_wavs))
            batch_wavs.clear()
            batch_outs.clear()

    # flush 剩余
    if batch_wavs:
        arr = np.stack(batch_wavs, axis=0)
        embs = perch_embed_batch(sess, in_name, embed_idx, arr)
        for op, emb in zip(batch_outs, embs):
            op.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.from_numpy(emb), op)
        pbar.update(len(batch_wavs))
    pbar.close()


# =============================================================================
# 2. sc: 每个 5s 段一个 emb (跟段对齐)
# =============================================================================
def build_sc_emb(sess, in_name, embed_idx):
    meta_csv = WAVEFORM_CACHE_DIR / "soundscape_cache_meta.csv"
    meta = pd.read_csv(meta_csv)
    print(f"sc: {len(meta)} segments from {meta_csv.name}")

    tasks = []
    for i, r in meta.iterrows():
        sc_fn = Path(r["cache_file"]).stem            # 'BC2026_Train_xxxx_S08_xxx'
        out_pt = PERCH_CACHE_DIR / "sc" / f"{sc_fn}__{int(r['start_sec'])}.pt"
        if out_pt.exists():
            continue
        src_pt = WAVEFORM_CACHE_DIR / r["cache_file"]
        if src_pt.exists():
            tasks.append((src_pt, out_pt, int(r["start_sec"])))

    if not tasks:
        print("  all sc emb already cached, skip")
        return

    print(f"  computing {len(tasks)} new emb (batch={PERCH_BATCH})")
    pbar = tqdm(total=len(tasks), desc="sc emb")
    batch_wavs = []
    batch_outs = []
    for src_pt, out_pt, start_sec in tasks:
        try:
            w = load_int16(src_pt)
        except Exception as e:
            print(f"  [skip] {src_pt}: {e}")
            pbar.update(1)
            continue
        s = start_sec * SR
        chunk = w[s:s + TRAIN_SAMPLES]
        if len(chunk) < TRAIN_SAMPLES:
            chunk = np.pad(chunk, (0, TRAIN_SAMPLES - len(chunk)))
        batch_wavs.append(chunk.astype(np.float32))
        batch_outs.append(out_pt)

        if len(batch_wavs) >= PERCH_BATCH:
            arr = np.stack(batch_wavs, axis=0)
            embs = perch_embed_batch(sess, in_name, embed_idx, arr)
            for op, emb in zip(batch_outs, embs):
                op.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.from_numpy(emb), op)
            pbar.update(len(batch_wavs))
            batch_wavs.clear()
            batch_outs.clear()

    if batch_wavs:
        arr = np.stack(batch_wavs, axis=0)
        embs = perch_embed_batch(sess, in_name, embed_idx, arr)
        for op, emb in zip(batch_outs, embs):
            op.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.from_numpy(emb), op)
        pbar.update(len(batch_wavs))
    pbar.close()


# =============================================================================
def main():
    print(f"PERCH_ONNX_PATH = {PERCH_ONNX_PATH}")
    print(f"PERCH_CACHE_DIR = {PERCH_CACHE_DIR}")
    if not PERCH_ONNX_PATH.exists():
        print(f"[ERROR] {PERCH_ONNX_PATH} not found. Run convert_perch_onnx.py first.")
        sys.exit(1)
    if not WAVEFORM_CACHE_DIR.exists():
        print(f"[ERROR] {WAVEFORM_CACHE_DIR} not found. Run build_cache.py first.")
        sys.exit(1)

    sess, in_name, embed_idx = get_session()

    print("\n=== Step 1: focal emb ===")
    build_focal_emb(sess, in_name, embed_idx)

    print("\n=== Step 2: sc emb ===")
    build_sc_emb(sess, in_name, embed_idx)

    print("\n✓ Perch emb cache built.")
    n_focal = sum(1 for _ in (PERCH_CACHE_DIR / "focal").rglob("*.pt"))
    n_sc    = sum(1 for _ in (PERCH_CACHE_DIR / "sc").rglob("*.pt"))
    print(f"  focal: {n_focal} files\n  sc:    {n_sc} files")


if __name__ == "__main__":
    main()
