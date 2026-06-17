# -*- coding: utf-8 -*-
"""
build_cache.py — 生成 waveform cache(本地训练前必须跑一次)
============================================================

把 train_audio/*.ogg + train_soundscapes/*.ogg 全部 decode → 32 kHz 单声道 →
量化成 int16 → 存为 .pt 文件;同时生成 3 个元数据 CSV.

产出:
    waveform_cache/
        focal/<species>/<file>.pt        (35549 个,~50–80 GB)
        sc/<file>.pt                      (66 个,~250 MB)
        audio_cache_meta.csv              (focal 元数据)
        soundscape_cache_meta.csv         (sc 5s 段元数据)
        soundscape_file_meta.csv          (sc 文件级映射)

预计耗时: 16 核 CPU 约 30–45 分钟; 4 核 1.5–2 小时.

用法:
    python build_cache.py
    # 增量:已存在的 .pt 会跳过,可断点重跑
"""
import os
import re
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from tqdm import tqdm

# =============================================================================
# 路径(★ 按需修改)
# =============================================================================
# 数据根目录: 默认 ./data, 用环境变量 BIRDCLEF_SED_ROOT 覆盖.
_ROOT     = Path(os.environ.get("BIRDCLEF_SED_ROOT", "./data"))
COMP_DIR  = _ROOT / "birdclef-2026"
CACHE_DIR = _ROOT / "waveform_cache"

# ★ 用比赛全量 train_soundscapes_labels.csv (66 个文件 = 59 full + 7 partial),
#   跟 notebook / tuckerarrants 公开 ONNX 完全一致.
#   CV 由 5-fold GroupKFold 划分, 不再需要额外 holdout.
TRAIN_SC_LABELS_CSV = COMP_DIR / "train_soundscapes_labels.csv"

SR = 32_000
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


# =============================================================================
# 单文件解码 + 量化
# =============================================================================
def load_32k_mono(path):
    """读 ogg → 32k 单声道 float32."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        # 极少数 ogg 用 librosa fallback
        wav, sr = librosa.load(str(path), sr=None, mono=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1) if wav.shape[1] < wav.shape[0] else wav.mean(axis=0)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def save_int16_pt(wav, out_path: Path):
    """量化到 int16,torch.save 存 .pt."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(wav, -1.0, 1.0)
    arr = (arr * 32767.0).astype(np.int16)
    torch.save(torch.from_numpy(arr), out_path)


def _decode_one(args):
    """子进程入口: (src_path, out_path) → bool 成功与否."""
    src, out = args
    try:
        if Path(out).exists():
            return True
        wav = load_32k_mono(src)
        save_int16_pt(wav, Path(out))
        return True
    except Exception as e:
        print(f"[FAIL] {src}: {e}", file=sys.stderr)
        return False


# =============================================================================
# Step A: focal cache
# =============================================================================
def build_focal_cache():
    df = pd.read_csv(COMP_DIR / "train.csv")
    print(f"focal: {len(df)} rows in train.csv")

    tasks = []
    rows  = []
    for idx, row in df.iterrows():
        rel = str(row["filename"])                     # "1161364/iNat...ogg"
        src = COMP_DIR / "train_audio" / rel
        if not src.exists():
            continue
        cache_rel = "focal/" + rel.replace(".ogg", ".pt")
        out       = CACHE_DIR / cache_rel
        tasks.append((str(src), str(out)))
        rows.append({
            "filename":      rel,
            "cache_file":    cache_rel,
            "primary_label": str(row["primary_label"]),
            "original_idx":  int(idx),
            "start_sec":     0,                         # focal 训练时随机 crop
        })

    print(f"focal: decoding {len(tasks)} files with {N_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        list(tqdm(ex.map(_decode_one, tasks, chunksize=8),
                  total=len(tasks), desc="focal decode"))

    # 只保留实际 cache 成功的
    rows = [r for r in rows if (CACHE_DIR / r["cache_file"]).exists()]
    pd.DataFrame(rows).to_csv(CACHE_DIR / "audio_cache_meta.csv", index=False)
    print(f"  → audio_cache_meta.csv ({len(rows)} rows)")


# =============================================================================
# Step B: soundscape cache (★ 处理 train_soundscapes_labels.csv 里的全 66 个文件)
# =============================================================================
def build_sc_cache():
    labs = pd.read_csv(TRAIN_SC_LABELS_CSV).drop_duplicates()
    labs["start_sec"] = (
        pd.to_timedelta(labs["start"]).dt.total_seconds().astype(int)
    )
    files = sorted(labs["filename"].unique())
    print(f"soundscape (train only): {len(files)} files, {len(labs)} label rows  "
          f"[source: {TRAIN_SC_LABELS_CSV.name}]")

    # B1. decode 整个 60s 文件 → sc/<filename>.pt
    tasks      = []
    file_rows  = []
    for fn in files:
        src = COMP_DIR / "train_soundscapes" / fn
        if not src.exists():
            print(f"  [WARN] missing {fn}")
            continue
        cache_rel = "sc/" + fn.replace(".ogg", ".pt")
        out       = CACHE_DIR / cache_rel
        tasks.append((str(src), str(out)))
        file_rows.append({"filename": fn, "cache_file": cache_rel})

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        list(tqdm(ex.map(_decode_one, tasks, chunksize=2),
                  total=len(tasks), desc="sc decode"))

    file_rows = [r for r in file_rows if (CACHE_DIR / r["cache_file"]).exists()]
    pd.DataFrame(file_rows).to_csv(
        CACHE_DIR / "soundscape_file_meta.csv", index=False)
    print(f"  → soundscape_file_meta.csv ({len(file_rows)} rows)")

    # B2. 5s 段元数据(每文件 12 段, start_sec ∈ [0, 5, 10, ..., 55])
    file_dict = {r["filename"]: r["cache_file"] for r in file_rows}
    seg_rows  = []
    for fn in files:
        if fn not in file_dict:
            continue
        m    = re.search(r"_S(\d+)_", fn)
        site = ("S" + m.group(1)) if m else "S?"
        for start_sec in range(0, 60, 5):
            sub = labs[(labs["filename"] == fn) & (labs["start_sec"] == start_sec)]
            label_set = set()
            for v in sub["primary_label"].astype(str):
                for s in v.split(";"):
                    s = s.strip()
                    if s:
                        label_set.add(s)
            seg_rows.append({
                "filename":   fn,
                "cache_file": file_dict[fn],
                "start_sec":  start_sec,
                "site":       site,
                "label_list": ";".join(sorted(label_set)),
            })
    pd.DataFrame(seg_rows).to_csv(
        CACHE_DIR / "soundscape_cache_meta.csv", index=False)
    print(f"  → soundscape_cache_meta.csv ({len(seg_rows)} rows)")


# =============================================================================
def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"COMP_DIR  = {COMP_DIR}")
    print(f"CACHE_DIR = {CACHE_DIR}")
    build_focal_cache()
    build_sc_cache()
    print("\n✓ Done. You can now run training.")


if __name__ == "__main__":
    main()
