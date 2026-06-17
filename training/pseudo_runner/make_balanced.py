# -*- coding: utf-8 -*-
"""make_balanced.py - 从已有的 pseudo_ensemble.csv 重新产出一份"分布健康"的伪标签.

★ 动机 (针对全局阈值方案的塌缩问题)
    直接对 pseudo_ensemble.csv 用全局阈值 (make_high_conf) 会让 primary_label 极度
    集中在少数高频物种:
        top5 物种 ~68% (全量) / ~86% (阈值0.7后), 234 类里 71 类从不出现.
    拿这种分布做自训练会进一步放大偏置, 长尾物种被饿死.

    本脚本用三招把分布拉平, 全程只读 pseudo_ensemble.csv, 不重跑推理:

    1) per-class 阈值 (--per-class-quantile)
        每个类只保留"该类自己 primary 概率"的高分位段 (默认每类取 top 50%),
        并夹在 [--floor-prob, --ceil-prob] 之间. 让每个类按自己的尺度贡献,
        而不是用一刀切的全局门槛把低分类全砍光.

    2) per-class capping (--max-per-class)
        每个类作为 primary 的段数封顶 (默认 3000). 超出的按概率降序保留 top-N,
        把训练预算从 compot1 (19798 段) 这种霸榜类腾给中低频类.

    3) 稀有类保护 (--rare-train-max / --rare-min-prob / --rare-cap-mult)
        train 样本 <= rare-train-max 的类用更低的 primary 门槛, 且 cap 放大,
        尽量把长尾物种喂进来.

    其余沿用 2nd place 习惯:
        - trim: 段内 < --trim-min-prob 的类别概率清零 (默认 0.1, 保软标签宽度)
        - StratifiedGroupKFold by audio_id, stratify=primary_label

用法:
    cd /path/to/repo

    # 默认: 每类 top50% 分位, 夹在 [0.45,0.80], 每类封顶 3000, 稀有类(<=20)门槛0.40且cap x2
    python -m pseudo_runner.make_balanced

    # 自定义
    python -m pseudo_runner.make_balanced \
        --per-class-quantile 0.6 --floor-prob 0.5 --ceil-prob 0.85 \
        --max-per-class 2500 --rare-train-max 20 --rare-min-prob 0.4 \
        --out-csv pseudo_output/pseudo_balanced.csv

注意:
    - fold_id 用 StratifiedGroupKFold(by audio_id) 重新打. 下游
      build_pseudo_dataframe 若自己按 audio_id 重划, 折数不影响训练.
    - 只影响"哪些段被选为训练样本"+"primary 列", 软概率本身不改 (只 trim).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .merge_softlabels import add_primary_label


def _audio_id(row_id_series: pd.Series) -> pd.Series:
    """row_id = {soundscape_stem}_{end_sec} -> soundscape_stem."""
    return row_id_series.apply(lambda x: "_".join(x.split("_")[:-1]))


def select_per_class(
    df: pd.DataFrame,
    rare_species: set,
    per_class_quantile: float,
    floor_prob: float,
    ceil_prob: float,
    rare_min_prob: float,
    max_per_class: int,
    rare_cap_mult: float,
) -> pd.DataFrame:
    """对每个 primary_label 分组, 应用 per-class 阈值 + capping.

    阈值 = clip( 该类 primary_label_prob 的 per_class_quantile 分位, floor, ceil )
    稀有类 floor 用 rare_min_prob; cap = max_per_class * rare_cap_mult.
    """
    keep_parts = []
    stats = []
    for sp, g in df.groupby("primary_label", sort=False):
        probs = g["primary_label_prob"].values
        is_rare = sp in rare_species

        # --- per-class 阈值 ---
        q = float(np.quantile(probs, per_class_quantile)) if len(probs) else floor_prob
        lo = rare_min_prob if is_rare else floor_prob
        thr = min(max(q, lo), ceil_prob)
        sel = g[g["primary_label_prob"] >= thr]

        # --- capping (按概率降序留 top-N) ---
        # 稀有类放大 cap 多收样本, 但不超过 2*max_per_class 绝对上限,
        # 防止"段数本就很多的稀有类"自己变成新的霸榜.
        if is_rare:
            cap = min(int(max_per_class * rare_cap_mult), 2 * max_per_class)
        else:
            cap = max_per_class
        if len(sel) > cap:
            sel = sel.nlargest(cap, "primary_label_prob")

        if len(sel):
            keep_parts.append(sel)
        stats.append((sp, len(g), thr, len(sel), is_rare))

    out = pd.concat(keep_parts, axis=0).reset_index(drop=True) if keep_parts else df.iloc[0:0]

    # 打印最受影响的类 (霸榜被削 + 稀有被救)
    st = pd.DataFrame(stats, columns=["sp", "n_in", "thr", "n_kept", "rare"])
    st["dropped"] = st["n_in"] - st["n_kept"]
    print("[per-class] 削减最多的 top8 (霸榜类被 capping/阈值压下):")
    for _, r in st.nlargest(8, "dropped").iterrows():
        print(f"    {r.sp:10s} {int(r.n_in):6d} -> {int(r.n_kept):5d} "
              f"(thr={r.thr:.3f}{' RARE' if r.rare else ''})")
    rare_kept = st[(st.rare) & (st.n_kept > 0)]
    print(f"[per-class] 稀有类被保留为 primary: {len(rare_kept)} 类, "
          f"贡献 {int(rare_kept.n_kept.sum())} 段")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("--ensemble-csv", type=str,
                        default="pseudo_output/pseudo_ensemble.csv",
                        help="未阈值化的融合软概率 csv (merge 阶段产物)")
    parser.add_argument("--train-csv", type=str, default="birdclef-2026/train.csv",
                        help="官方 train.csv, 用于统计每类样本数 (判定稀有度)")
    parser.add_argument("--out-csv", type=str, default="",
                        help="输出 csv; 留空自动命名 pseudo_balanced_q<..>_cap<..>.csv")
    # per-class 阈值
    parser.add_argument("--per-class-quantile", type=float, default=0.5,
                        help="每类只保留 primary 概率高于该分位的段 (0=全留,0.5=top50%%)")
    parser.add_argument("--floor-prob", type=float, default=0.45,
                        help="常见类 per-class 阈值下限")
    parser.add_argument("--ceil-prob", type=float, default=0.80,
                        help="per-class 阈值上限 (防高频类把门槛抬太高反而漏样本)")
    # capping
    parser.add_argument("--max-per-class", type=int, default=3000,
                        help="每个 primary 类的段数封顶 (按概率降序留 top-N)")
    # 稀有类保护
    parser.add_argument("--rare-train-max", type=int, default=20,
                        help="train 样本数 <= 此值视为稀有类")
    parser.add_argument("--rare-min-prob", type=float, default=0.40,
                        help="稀有类阈值下限 (比 floor 低, 护长尾)")
    parser.add_argument("--rare-cap-mult", type=float, default=1.0,
                        help="稀有类 cap 放大倍数. 默认 1.0: 'train稀有'≠'soundscape稀有', "
                             "放大会让在 soundscape 里常见的伪稀有类霸榜. 真稀有类段数本就不足 cap.")
    # 通用
    parser.add_argument("--trim-min-prob", type=float, default=0.1,
                        help="段内小于此值的类别概率清零")
    parser.add_argument("--n-folds", type=int, default=5)
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
        / f"pseudo_balanced_q{args.per_class_quantile:.2f}_cap{args.max_per_class}.csv"
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

    vc0 = df["primary_label"].value_counts()
    print(f"[before] rows={len(df)}  unique_primary={df['primary_label'].nunique()}/234  "
          f"top5占比={vc0.head(5).sum()/len(df)*100:.1f}%")

    # ---- 3) per-class 阈值 + capping ----
    sel = select_per_class(
        df,
        rare_species=rare_species,
        per_class_quantile=args.per_class_quantile,
        floor_prob=args.floor_prob,
        ceil_prob=args.ceil_prob,
        rare_min_prob=args.rare_min_prob,
        max_per_class=args.max_per_class,
        rare_cap_mult=args.rare_cap_mult,
    ).reset_index(drop=True)

    vc1 = sel["primary_label"].value_counts()
    print(f"[after ] rows={len(sel)}  unique_primary={sel['primary_label'].nunique()}/234  "
          f"top5占比={vc1.head(5).sum()/max(1,len(sel))*100:.1f}%  "
          f"top10={vc1.head(10).sum()/max(1,len(sel))*100:.1f}%")

    # ---- 4) trim: 小于 trim_min_prob 的类别概率清零 ----
    sel = sel.copy()  # 去碎片化 (groupby+concat 后列很碎)
    cls_cols = [c for c in sel.columns
                if c not in ("row_id", "primary_label", "primary_label_prob")]
    probs = sel[cls_cols].values.astype(np.float32)
    probs[probs < args.trim_min_prob] = 0.0
    sel[cls_cols] = probs

    # ---- 5) StratifiedGroupKFold by audio_id, stratify=primary_label ----
    sample_id = _audio_id(sel["row_id"]).values
    n_groups = len(np.unique(sample_id))
    n_folds = min(args.n_folds, n_groups)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    fold_id = np.full(len(sel), -1, dtype=int)
    for fid, (_, val_idx) in enumerate(sgkf.split(
            sel, sel["primary_label"].values, sample_id)):
        fold_id[val_idx] = fid
    assert (fold_id >= 0).all()
    sel["fold_id"] = fold_id
    print(f"[fold] dist = {pd.Series(fold_id).value_counts().sort_index().to_dict()}")

    # ---- 6) 落盘 ----
    sel.to_csv(out_csv, index=False)
    print(f"[save] {out_csv}  shape={sel.shape}")

    # ---- 7) 统计 ----
    nz_per_row = (sel[cls_cols].values > 0).sum(axis=1)
    n_audio = _audio_id(sel["row_id"]).nunique()
    prim_species = set(sel["primary_label"].unique())
    rare_as_primary = rare_species & prim_species
    print(
        f"[stat] kept rows={len(sel)}  unique_audio={n_audio}  "
        f"作为primary的物种={len(prim_species)}/234\n"
        f"[stat] primary_label_prob: min={sel['primary_label_prob'].min():.3f} "
        f"median={sel['primary_label_prob'].median():.3f} "
        f"max={sel['primary_label_prob'].max():.3f}\n"
        f"[stat] non-zero classes/row: min={nz_per_row.min()} "
        f"median={int(np.median(nz_per_row))} max={nz_per_row.max()}\n"
        f"[stat] ★ 稀有类作为primary保留={len(rare_as_primary)}/{len(rare_species)}"
    )


if __name__ == "__main__":
    main()
