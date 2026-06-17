# -*- coding: utf-8 -*-
"""merge_batches.py - 把所有 batch 的 csv 跨批合并 + 方案融合 + 过滤 + 5 折.

输入结构 (run_batches.py 的输出):
    output_dir/
        batch_00/
            m5/{submission_protossm.csv, submission_sed.csv}
        batch_01/...
        ...

输出:
    output_dir/
        m5/{submission_protossm.csv, submission_sed.csv}    ← 跨 batch concat
        pseudo_m5.csv                                        ← 单套 proto*0.6 + sed*0.4
        pseudo_ensemble.csv                                  ← 多套等权平均 (单套=pseudo_m5)
        pseudo_filtered_grouped.csv                          ← ★★ 训练直接用

第三步 (filter + group) 完全复用 merge_softlabels.py 的 add_primary_label /
filter_and_split, 保证跟单批跑出的结果格式一致.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .merge_softlabels import (
    PROTO_W, SED_W,
    add_primary_label, filter_and_split, merge_one_scheme, merge_schemes,
)


def concat_batches_for_scheme(batch_dirs: List[Path], scheme: str,
                                output_root: Path) -> None:
    """把所有 batch 下某套方案的 protossm/sed csv concat 起来,
    写到 ``output_root/{scheme}/submission_protossm.csv`` 和 _sed.csv."""
    proto_dfs, sed_dfs = [], []
    for bd in batch_dirs:
        p_csv = bd / scheme / "submission_protossm.csv"
        s_csv = bd / scheme / "submission_sed.csv"
        if not p_csv.exists() or not s_csv.exists():
            print(f"[merge_batches] WARN missing: {p_csv} or {s_csv}, skip {bd.name}")
            continue
        proto_dfs.append(pd.read_csv(p_csv))
        sed_dfs.append(pd.read_csv(s_csv))
        print(f"[merge_batches] {bd.name}/{scheme}: "
              f"proto={proto_dfs[-1].shape}  sed={sed_dfs[-1].shape}")

    if not proto_dfs:
        raise RuntimeError(f"No batch has {scheme} csvs")

    proto_all = pd.concat(proto_dfs, ignore_index=True)
    sed_all = pd.concat(sed_dfs, ignore_index=True)

    # 去重 (相同 row_id 只保留第一份, 防止 batch 边界重叠 — 理论上不会有)
    proto_all = proto_all.drop_duplicates(subset="row_id").reset_index(drop=True)
    sed_all = sed_all.drop_duplicates(subset="row_id").reset_index(drop=True)

    out_dir = output_root / scheme
    out_dir.mkdir(parents=True, exist_ok=True)
    proto_all.to_csv(out_dir / "submission_protossm.csv", index=False)
    sed_all.to_csv(out_dir / "submission_sed.csv", index=False)
    print(f"[merge_batches] {scheme}: concatenated proto={proto_all.shape}  "
          f"sed={sed_all.shape} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="run_batches 的输出根, 含 batch_00..batch_NN/ 子目录")
    parser.add_argument("--schemes", nargs="+", default=["m5"])
    parser.add_argument("--primary-min-prob", type=float, default=0.5)
    parser.add_argument("--trim-min-prob", type=float, default=0.1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1) 找所有 batch_NN/ 子目录
    batch_dirs = sorted(p for p in args.output_dir.iterdir()
                         if p.is_dir() and p.name.startswith("batch_"))
    if not batch_dirs:
        raise RuntimeError(f"No batch_* subdirs under {args.output_dir}")
    print(f"[merge_batches] found {len(batch_dirs)} batches: "
          f"{[p.name for p in batch_dirs]}")

    # 2) 每套方案: concat 所有 batch 的 proto/sed csv -> output_dir/{scheme}/
    for s in args.schemes:
        concat_batches_for_scheme(batch_dirs, s, args.output_dir)

    # 3) 单套合并 (proto*0.6 + sed*0.4)
    scheme_dirs = [args.output_dir / s for s in args.schemes]
    for s, d in zip(args.schemes, scheme_dirs):
        merged = merge_one_scheme(d)
        merged_pl = add_primary_label(merged)
        out_csv = args.output_dir / f"pseudo_{s}.csv"
        merged_pl.to_csv(out_csv, index=False)
        print(f"[save] {out_csv}  shape={merged_pl.shape}")

    # 4) 多套等权平均
    ensemble = merge_schemes(scheme_dirs)
    ensemble_pl = add_primary_label(ensemble)
    ens_csv = args.output_dir / "pseudo_ensemble.csv"
    ensemble_pl.to_csv(ens_csv, index=False)
    print(f"[save] {ens_csv}  shape={ensemble_pl.shape}")

    # 5) 过滤 + 5 折分组
    filtered = filter_and_split(
        ensemble_pl,
        primary_min_prob=args.primary_min_prob,
        trim_min_prob=args.trim_min_prob,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    out_csv = args.output_dir / "pseudo_filtered_grouped.csv"
    filtered.to_csv(out_csv, index=False)
    print(f"[save] {out_csv}  shape={filtered.shape}")

    # 6) 简要统计
    cls_cols = [c for c in filtered.columns
                if c not in ("row_id", "primary_label", "primary_label_prob", "fold_id")]
    import numpy as np
    nz_per_row = (filtered[cls_cols].values > 0).sum(axis=1)
    print(f"[stat] primary_label_prob: min={filtered['primary_label_prob'].min():.3f} "
          f"median={filtered['primary_label_prob'].median():.3f} "
          f"max={filtered['primary_label_prob'].max():.3f}")
    print(f"[stat] non-zero classes per row: "
          f"min={nz_per_row.min()} median={int(np.median(nz_per_row))} max={nz_per_row.max()}")
    print(f"[stat] unique primary_labels: {filtered['primary_label'].nunique()} / {len(cls_cols)}")
    print(f"\n{'='*70}\nDONE: {out_csv}\n{'='*70}")


if __name__ == "__main__":
    main()
