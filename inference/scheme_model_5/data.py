# -*- coding: utf-8 -*-
"""data.py — taxonomy / sample_submission / soundscape_labels → ``sc`` / ``Y_FULL``.

对应 cell_07 第 143-268 行: ``find_competition_dir`` 的运行期搜索 + 简化版 ``parse_fname``
+ 12-window 全标注文件筛选.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import BASE, N_WINDOWS, WINDOW_SEC


FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_fname(name: str) -> Dict:
    m = FNAME_RE.match(name)
    if not m:
        return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


def union_labels(series) -> List[str]:
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)


def find_competition_dir() -> Path:
    """cell_07 第 143-157 行: 优先用 ``BIRDCLEF_BASE`` 指定的, 否则在 /kaggle/input 搜索."""
    if BASE.exists() and (BASE / "sample_submission.csv").exists():
        print(f"[data] using competition data: {BASE}")
        return BASE
    candidates = [
        Path("/kaggle/input/competitions/birdclef-2026"),
        Path("/kaggle/input/birdclef-2026"),
    ]
    for p in candidates:
        if (p / "sample_submission.csv").exists() and (p / "taxonomy.csv").exists():
            print(f"[data] using competition data: {p}")
            return p
    root = Path("/kaggle/input")
    if root.exists():
        for p in root.rglob("sample_submission.csv"):
            parent = p.parent
            if (parent / "taxonomy.csv").exists() and (parent / "train_soundscapes_labels.csv").exists():
                print(f"[data] using competition data: {parent}")
                return parent
    raise FileNotFoundError("BirdCLEF competition data directory not found.")


def load_data() -> Dict:
    base              = find_competition_dir()
    taxonomy          = pd.read_csv(base / "taxonomy.csv")
    sample_sub        = pd.read_csv(base / "sample_submission.csv")
    soundscape_labels = pd.read_csv(base / "train_soundscapes_labels.csv")

    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    N_CLASSES      = len(PRIMARY_LABELS)
    label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

    sc = (
        soundscape_labels
        .groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"]  = (
        sc["filename"].str.replace(".ogg", "", regex=False)
        + "_" + sc["end_sec"].astype(str)
    )
    _meta = sc["filename"].apply(parse_fname).apply(pd.Series)
    sc    = pd.concat([sc, _meta], axis=1)

    Y_SC = np.zeros((len(sc), N_CLASSES), dtype=np.uint8)
    for i, lbls in enumerate(sc["label_list"]):
        for lbl in lbls:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1

    windows_per_file = sc.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    sc["fully_labeled"] = sc["filename"].isin(full_files)

    full_rows = (
        sc[sc["fully_labeled"]]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )
    Y_FULL = Y_SC[full_rows["index"].to_numpy()]

    print(f"[data] classes={N_CLASSES}  fully_labeled_files={len(full_files)}  "
          f"full_windows={len(full_rows)}  active={int((Y_FULL.sum(0)>0).sum())}")

    return {
        "BASE":              base,
        "taxonomy":          taxonomy,
        "sample_sub":        sample_sub,
        "soundscape_labels": soundscape_labels,
        "sc":                sc,
        "Y_SC":              Y_SC,
        "full_rows":         full_rows,
        "Y_FULL":            Y_FULL,
        "PRIMARY_LABELS":    PRIMARY_LABELS,
        "N_CLASSES":         N_CLASSES,
        "label_to_idx":      label_to_idx,
        "full_files":        full_files,
    }
