# -*- coding: utf-8 -*-
"""mapping.py — Perch labels mapping + genus proxy + per-taxon temperatures.

对应 cell_07 第 271-374 行 (Perch labels + mapping + proxy_map + temperatures).
"""
from __future__ import annotations

import re as _re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import INPUT_ROOT, MODEL_DIR


TEXTURE_TAXA: set = {"Amphibia", "Insecta"}
PROXY_TAXA:   set = {"Amphibia", "Insecta", "Aves"}


def find_perch_labels_path() -> Path:
    preferred = MODEL_DIR / "assets" / "labels.csv"
    if preferred.exists():
        return preferred
    if INPUT_ROOT.exists():
        for p in sorted(INPUT_ROOT.rglob("labels.csv")):
            try:
                cols = set(pd.read_csv(p, nrows=0).columns)
            except Exception:
                continue
            if {"inat2024_fsd50k", "scientific_name"} & cols:
                print(f"[mapping] using perch labels: {p}")
                return p
    raise FileNotFoundError(
        "Perch labels.csv not found. Attach Perch ONNX labels or "
        "google/bird-vocalization-classifier."
    )


def load_perch_labels(path: Path) -> pd.DataFrame:
    df = (pd.read_csv(path)
          .reset_index()
          .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}))
    if "scientific_name" not in df.columns:
        for c in ("label", "labels", "name"):
            if c in df.columns:
                df = df.rename(columns={c: "scientific_name"})
                break
    assert "scientific_name" in df.columns, f"No scientific_name column in {path}"
    return df


def build_mapping(taxonomy: pd.DataFrame, label_to_idx: Dict[str, int],
                  PRIMARY_LABELS: List[str], n_classes: int) -> Dict:
    bc_labels = load_perch_labels(find_perch_labels_path())
    NO_LABEL  = len(bc_labels)

    mapping_df = (
        taxonomy
        .merge(bc_labels.rename(columns={"scientific_name": "scientific_name"}),
               on="scientific_name", how="left")
    )
    mapping_df["bc_index"] = mapping_df["bc_index"].fillna(NO_LABEL).astype(int)
    lbl2bc = mapping_df.set_index("primary_label")["bc_index"]

    BC_INDICES    = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
    MAPPED_MASK   = BC_INDICES != NO_LABEL
    MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
    UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)

    CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()

    # genus proxy
    proxy_map: Dict[int, List[int]] = {}
    unmapped_df = (
        taxonomy[taxonomy["primary_label"]
        .isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])].copy()
    )
    for _, row in unmapped_df.iterrows():
        target = row["primary_label"]
        sci    = str(row["scientific_name"])
        genus  = sci.split()[0]
        hits = bc_labels[
            bc_labels["scientific_name"]
            .astype(str)
            .str.match(rf"^{_re.escape(genus)}\s", na=False)
        ]
        if len(hits) > 0:
            proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()
    proxy_map = {
        idx: bc_idxs
        for idx, bc_idxs in proxy_map.items()
        if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in PROXY_TAXA
    }
    print(f"[mapping] mapped={MAPPED_MASK.sum()}/{n_classes}  "
          f"unmapped={len(UNMAPPED_POS)}  proxy={len(proxy_map)}  "
          f"no_signal={len(UNMAPPED_POS) - len(proxy_map)}")

    # per-taxon temperatures (cell_07 第 370-374 行)
    temperatures = np.ones(n_classes, dtype=np.float32)
    for ci, label in enumerate(PRIMARY_LABELS):
        cls = CLASS_NAME_MAP.get(label, "Aves")
        temperatures[ci] = 0.95 if cls in TEXTURE_TAXA else 1.10

    return {
        "bc_labels":      bc_labels,
        "NO_LABEL":       NO_LABEL,
        "BC_INDICES":     BC_INDICES,
        "MAPPED_MASK":    MAPPED_MASK,
        "MAPPED_POS":     MAPPED_POS,
        "MAPPED_BC_IDX":  MAPPED_BC_IDX,
        "UNMAPPED_POS":   UNMAPPED_POS,
        "CLASS_NAME_MAP": CLASS_NAME_MAP,
        "TEXTURE_TAXA":   TEXTURE_TAXA,
        "PROXY_TAXA":     PROXY_TAXA,
        "proxy_map":      proxy_map,
        "temperatures":   temperatures,
    }
