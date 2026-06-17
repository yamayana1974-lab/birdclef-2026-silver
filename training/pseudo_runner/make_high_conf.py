# -*- coding: utf-8 -*-
"""make_high_conf.py - 从已有的 pseudo_ensemble.csv 重新过滤出"高置信度"伪标签.

不重跑任何推理. 直接读 merge 阶段产出的 pseudo_ensemble.csv (未阈值化的原始软概率),
用更高的 primary_label_prob 门槛重新过滤 + trim + 分折.

★ 路 A + 稀有类保护:
    - trim 维持 0.1 (不砍次要物种弱信号, 保长尾)
    - 常见类 (train 样本多) primary 门槛高 (默认 0.7), 保证主标签质量
    - 稀有类 (train 样本 <= rare_train_max) primary 门槛低 (默认 0.4),
      让稀有物种即使置信度没到 0.7 也能作为主标签保留下来.

稀有度判定: 用 birdclef-2026/train.csv 里每个 primary_label 的样本数衡量,
           <= --rare-train-max (默认 20) 视为稀有.

用法:
    cd /path/to/repo

    # 默认: 常见 0.7 / 稀有 0.4, trim 0.1, 4 折
    python -m pseudo_runner.make_high_conf

    # 自定义
    python -m pseudo_runner.make_high_conf \
        --primary-min-prob 0.7 --rare-min-prob 0.35 --rare-train-max 20 \
        --out-csv pseudo_output/pseudo_high_conf_a.csv

注意:
    - fold_id 用 StratifiedGroupKFold 重新打 (折数 --n-folds).
      hgnet_train_test_weak_labels 的 build_pseudo_dataframe 会忽略它、
      自己按 audio_id 重划, 所以 --n-folds 设几不影响训练.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .merge_softlabels import add_primary_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ensemble-csv", type=str,
        default="pseudo_output/pseudo_ensemble.csv",
        help="未阈值化的融合软概率 csv (merge 阶段产物)",
    )
    parser.add_argument(
        "--train-csv", type=str, default="birdclef-2026/train.csv",
        help="官方 train.csv, 用于统计每个物种样本数 (判定稀有度)",
    )
    parser.add_argument(
        "--out-csv", type=str, default="",
        help="输出 csv; 留空自动命名 pseudo_high_conf_<prob>_rare<rare>.csv",
    )
    parser.add_argument("--primary-min-prob", type=float, default=0.7,
                        help="常见类的 primary 门槛")
    parser.add_argument("--rare-min-prob", type=float, default=0.4,
                        help="稀有类的 primary 门槛 (更低, 护长尾)")
    parser.add_argument("--rare-train-max", type=int, default=20,
                        help="train 样本数 <= 此值视为稀有类")
    parser.add_argument("--trim-min-prob", type=float, default=0.1,
                        help="段内小于此值的类别概率清零")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensemble_csv = Path(args.ensemble_csv)
    if not ensemble_csv.exists():
        raise FileNotFoundError(f"ensemble csv not found: {ensemble_csv}")
    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    out_csv = (
        Path(args.out_csv) if args.out_csv
        else ensemble_csv.parent
        / f"pseudo_high_conf_{args.primary_min_prob:.2f}_rare{args.rare_min_prob:.2f}.csv"
    )

    # ---- 1) 稀有度: train.csv 每类样本数 ----
    train = pd.read_csv(train_csv)
    sp_count = train["primary_label"].value_counts()
    rare_species = set(sp_count[sp_count <= args.rare_train_max].index)
    print(f"[rare] train 物种={train['primary_label'].nunique()}  "
          f"稀有类(<= {args.rare_train_max})={len(rare_species)}")

    # ---- 2) 读 ensemble 软概率 ----
    print(f"[load] {ensemble_csv}")
    df = pd.read_csv(ensemble_csv, low_memory=False)
    print(f"[load] rows={len(df)}  cols={len(df.columns)}")
    if "primary_label" not in df.columns or "primary_label_prob" not in df.columns:
        print("[prep] add_primary_label (argmax + max)")
        df = add_primary_label(df)

    # ---- 3) 自适应 primary 门槛过滤 ----
    is_rare = df["primary_label"].isin(rare_species).values
    row_thr = np.where(is_rare, args.rare_min_prob, args.primary_min_prob)
    keep = df["primary_label_prob"].values > row_thr
    sel = df[keep].reset_index(drop=True)
    n_rare_kept = int(is_rare[keep].sum())
    print(f"[filter] before={len(df)} -> after={len(sel)} "
          f"({len(sel)/max(1,len(df))*100:.1f}%)  "
          f"其中稀有类行={n_rare_kept}  常见类行={len(sel)-n_rare_kept}")

    # ---- 4) trim: 小于 trim_min_prob 的类别概率清零 ----
    cls_cols = [c for c in sel.columns
                if c not in ("row_id", "primary_label", "primary_label_prob")]
    probs = sel[cls_cols].values.astype(np.float32)
    probs[probs < args.trim_min_prob] = 0.0
    sel[cls_cols] = probs

    # ---- 5) StratifiedGroupKFold by sample_id, stratify=primary_label ----
    sel["sample_id"] = sel["row_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    sel["fold_id"] = -1
    for fid, (_, val_idx) in enumerate(sgkf.split(
            sel, sel["primary_label"].values, sel["sample_id"].values)):
        sel.iloc[val_idx, sel.columns.get_loc("fold_id")] = fid
    assert (sel["fold_id"] >= 0).all()
    sel = sel.drop(columns=["sample_id"])
    print(f"[fold] dist = {sel['fold_id'].value_counts().sort_index().to_dict()}")

    # ---- 6) 落盘 ----
    sel.to_csv(out_csv, index=False)
    print(f"[save] {out_csv}  shape={sel.shape}")

    # ---- 7) 统计 (重点看稀有类覆盖) ----
    nz_per_row = (sel[cls_cols].values > 0).sum(axis=1)
    n_audio = sel["row_id"].apply(lambda x: "_".join(x.split("_")[:-1])).nunique()
    prim_species = set(sel["primary_label"].unique())
    rare_as_primary = rare_species & prim_species
    print(
        f"[stat] kept rows={len(sel)}  unique_audio={n_audio}  "
        f"作为primary的物种={len(prim_species)}\n"
        f"[stat] primary_label_prob: min={sel['primary_label_prob'].min():.3f} "
        f"median={sel['primary_label_prob'].median():.3f} "
        f"max={sel['primary_label_prob'].max():.3f}\n"
        f"[stat] non-zero classes/row: min={nz_per_row.min()} "
        f"median={int(pd.Series(nz_per_row).median())} max={nz_per_row.max()}\n"
        f"[stat] ★ 稀有类作为primary保留={len(rare_as_primary)}/{len(rare_species)}"
    )


if __name__ == "__main__":
    main()
