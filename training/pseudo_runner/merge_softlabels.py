# -*- coding: utf-8 -*-
"""merge_softlabels.py - 把方案的 protossm/sed 软概率融合 + 过滤 + 5 折分组.

输入:
    output_dir/
      m5/{submission_protossm.csv, submission_sed.csv}
      (历史: 可传多个 scheme 子目录, 会等权平均; 现默认单套 m5)

融合公式 (复刻 SOLUT.xSED = [0.6, 0.4]):
    pseudo = 0.6 * proto + 0.4 * sed

多套时等权平均 (单套则 = 自身):
    pseudo_ensemble = mean(pseudo_<scheme> ...)

输出:
    output_dir/
      pseudo_<scheme>.csv    # 每套合并后的软概率
      pseudo_ensemble.csv    # 多套等权平均 (单套 = pseudo_m5)
      pseudo_filtered_grouped.csv   # primary_label_prob>0.5, <0.1 trim, 5折GroupKFold
                                      ← ★ 这个直接喂给 SED/HGNet 训练分支

格式同 2nd place:
    row_id, <234 类>, primary_label, primary_label_prob, fold_id
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROTO_W = 0.60
SED_W = 0.40


def _load_csv_aligned(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_id" not in df.columns:
        raise ValueError(f"{path} 缺 row_id 列")
    return df.sort_values("row_id").reset_index(drop=True)


def merge_one_scheme(scheme_dir: Path) -> pd.DataFrame:
    """读 protossm + sed,按 0.6/0.4 融合."""
    proto_csv = scheme_dir / "submission_protossm.csv"
    sed_csv = scheme_dir / "submission_sed.csv"
    if not proto_csv.exists():
        raise FileNotFoundError(proto_csv)
    if not sed_csv.exists():
        raise FileNotFoundError(sed_csv)

    proto = _load_csv_aligned(proto_csv)
    sed = _load_csv_aligned(sed_csv)

    if not (proto["row_id"].values == sed["row_id"].values).all():
        # 取交集再 align
        common = sorted(set(proto["row_id"]) & set(sed["row_id"]))
        proto = proto.set_index("row_id").loc[common].reset_index()
        sed = sed.set_index("row_id").loc[common].reset_index()

    cols = [c for c in proto.columns if c != "row_id"]
    assert cols == [c for c in sed.columns if c != "row_id"], (
        f"列不匹配: {scheme_dir}"
    )

    merged = proto.copy()
    merged[cols] = (
        PROTO_W * proto[cols].values + SED_W * sed[cols].values
    ).astype(np.float32)
    return merged


def merge_schemes(scheme_dirs: List[Path]) -> pd.DataFrame:
    """读多套方案 → 各自融合 → 等权平均."""
    all_dfs = [merge_one_scheme(d) for d in scheme_dirs]

    # 取所有方案 row_id 交集
    common = sorted(set(all_dfs[0]["row_id"]))
    for df in all_dfs[1:]:
        common = sorted(set(common) & set(df["row_id"]))
    print(f"[merge] {len(scheme_dirs)} schemes, common row_ids = {len(common)}")

    aligned = []
    for df in all_dfs:
        d = df.set_index("row_id").loc[common].reset_index()
        aligned.append(d)

    cols = [c for c in aligned[0].columns if c != "row_id"]
    stacked = np.stack([d[cols].values for d in aligned], axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)

    out = aligned[0].copy()
    out[cols] = mean
    return out


def add_primary_label(df: pd.DataFrame) -> pd.DataFrame:
    """加 primary_label / primary_label_prob 两列 (argmax + max)."""
    cls_cols = [c for c in df.columns if c != "row_id"]
    probs = df[cls_cols].values.astype(np.float32)
    primary_idx = np.argmax(probs, axis=1)
    out = df.copy()
    out["primary_label"] = [cls_cols[i] for i in primary_idx]
    out["primary_label_prob"] = probs[np.arange(len(probs)), primary_idx].astype(np.float32)
    return out


def filter_and_split(
    df: pd.DataFrame,
    primary_min_prob: float = 0.5,
    trim_min_prob: float = 0.1,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """复刻 2nd place 的 create_pseudo.ipynb:
        1. primary_label_prob > 0.5 过滤
        2. < 0.1 类别概率清零 (trim, 防多标签噪声)
        3. StratifiedGroupKFold by sample_id (filename), 给每行打 fold_id
    """
    sel = df[df["primary_label_prob"] > primary_min_prob].reset_index(drop=True)
    print(f"[filter] before {len(df)} -> after {len(sel)} "
          f"({len(sel) / max(1, len(df)) * 100:.1f}%)")

    cls_cols = [c for c in sel.columns
                if c not in ("row_id", "primary_label", "primary_label_prob")]
    probs = sel[cls_cols].values.astype(np.float32)
    probs[probs < trim_min_prob] = 0.0
    sel[cls_cols] = probs

    # sample_id = filename stem (去掉最后 _<sec>)
    sel["sample_id"] = sel["row_id"].apply(lambda x: "_".join(x.split("_")[:-1]))

    # 5 folds, group=sample_id, stratify=primary_label
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    sel["fold_id"] = -1
    for fid, (_, val_idx) in enumerate(sgkf.split(
            sel, sel["primary_label"].values, sel["sample_id"].values)):
        sel.iloc[val_idx, sel.columns.get_loc("fold_id")] = fid
    assert (sel["fold_id"] >= 0).all()

    print(f"[fold] distribution:\n{sel['fold_id'].value_counts().sort_index().to_dict()}")
    return sel.drop(columns=["sample_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="包含方案子目录 (默认 m5/) 的输出根")
    parser.add_argument("--schemes", nargs="+", default=["m5"],
                        help="要融合的方案子目录名 (空格分隔)")
    parser.add_argument("--primary-min-prob", type=float, default=0.5)
    parser.add_argument("--trim-min-prob", type=float, default=0.1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scheme_dirs = [args.output_dir / s for s in args.schemes]
    for d in scheme_dirs:
        if not d.exists():
            raise FileNotFoundError(d)

    # 1) 单套合并 (proto + sed)
    for s, d in zip(args.schemes, scheme_dirs):
        merged = merge_one_scheme(d)
        merged_with_pl = add_primary_label(merged)
        out_csv = args.output_dir / f"pseudo_{s}.csv"
        merged_with_pl.to_csv(out_csv, index=False)
        print(f"[save] {out_csv}  shape={merged_with_pl.shape}")

    # 2) 多套等权平均
    ensemble = merge_schemes(scheme_dirs)
    ensemble_with_pl = add_primary_label(ensemble)
    ensemble_csv = args.output_dir / "pseudo_ensemble.csv"
    ensemble_with_pl.to_csv(ensemble_csv, index=False)
    print(f"[save] {ensemble_csv}  shape={ensemble_with_pl.shape}")

    # 3) 过滤 + 分组 (训练直接用)
    filtered = filter_and_split(
        ensemble_with_pl,
        primary_min_prob=args.primary_min_prob,
        trim_min_prob=args.trim_min_prob,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    filtered_csv = args.output_dir / "pseudo_filtered_grouped.csv"
    filtered.to_csv(filtered_csv, index=False)
    print(f"[save] {filtered_csv}  shape={filtered.shape}")

    # 4) 简要统计
    cls_cols = [c for c in filtered.columns
                if c not in ("row_id", "primary_label", "primary_label_prob", "fold_id")]
    nz_per_row = (filtered[cls_cols].values > 0).sum(axis=1)
    print(f"[stat] primary_label_prob: min={filtered['primary_label_prob'].min():.3f} "
          f"median={filtered['primary_label_prob'].median():.3f} "
          f"max={filtered['primary_label_prob'].max():.3f}")
    print(f"[stat] non-zero classes per row: "
          f"min={nz_per_row.min()} median={int(np.median(nz_per_row))} max={nz_per_row.max()}")
    print(f"[stat] unique primary_labels: {filtered['primary_label'].nunique()} / 234")


if __name__ == "__main__":
    main()
