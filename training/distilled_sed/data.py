# -*- coding: utf-8 -*-
"""
data.py — 数据加载 + fold 分配 + 标签矩阵 + 稀有类上采样
==========================================================

对应 notebook 的 S2 cell.

主要产出 (load_data 返回 dict):
  - PRIMARY_LABELS:        234 个物种名 (sample_submission 列顺序)
  - LABEL2IDX:             {物种名 → 0~233}
  - TAXON_MASKS:           {纲名 → 该纲物种列索引} 用于评估细分
  - audio_cache_meta:      focal 录音元数据 (35549 条, 含 fold/secondary_labels)
  - sc_cache_meta:         soundscape 5s 段元数据 (739 条, 含 fold)
  - Y_SC:                  (n_sc, 234) labeled soundscape 多标签矩阵
  - non_s22_mask_sc:       (n_sc,) bool, True = 不是 S22 站点 (主指标用)
  - focal_secondary_labels: {original_idx → [secondary label list]}
  - sc_mixup_sources:      给 Focal-Soundscape MixUp 用的池子 (S4 里用)

数据来源依赖:
  - Kaggle dataset `bc2026-waveform-cache`: 预切的 int16 .pt waveform + metadata CSV
  - 没有这个 cache 训练就跑不起来 (除非你自己 decode 35549 个 ogg)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

from config import (COMP_DIR, WAVEFORM_CACHE_DIR, LABELS_PATH,
                    TRAIN_SC_LABELS,
                    TAXONOMY_PATH, SAMPLE_SUB_PATH, NUM_CLASSES, N_FOLDS,
                    SEED, MIN_SAMPLE, DEBUG, USE_FOCAL_SC_MIXUP)


def load_data():
    """
    加载所有训练数据 + 标签 + fold 分配.

    返回一个 dict, 含 12 个 key 给后续模块用 (见模块开头注释).
    """
    # ── 1. 标签 → idx 映射 (用 sample_submission 的列顺序) ──────────
    sample_sub     = pd.read_csv(SAMPLE_SUB_PATH)
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    LABEL2IDX      = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}

    # ── 2. taxonomy → 每个纲的列 mask (评估用) ─────────────────────
    taxonomy       = pd.read_csv(TAXONOMY_PATH)
    label_to_taxon = dict(zip(taxonomy["primary_label"].astype(str),
                              taxonomy["class_name"].astype(str)))
    TAXON_MASKS = {
        t: np.array([i for i, l in enumerate(PRIMARY_LABELS)
                     if label_to_taxon.get(l, "") == t])
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
    }

    # ── 3. focal 录音元数据 ────────────────────────────────────────
    # 从 waveform cache 读 (跟 train.csv merge 拿 secondary_labels)
    audio_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "audio_cache_meta.csv")
    train_df         = pd.read_csv(COMP_DIR / "train.csv")
    audio_cache_meta = audio_cache_meta.merge(
        train_df[["filename", "secondary_labels"]], on="filename", how="left",
    )
    # 只留 primary_label 在 LABEL2IDX 里的 (跟比赛 234 类对齐)
    audio_cache_meta = audio_cache_meta[
        audio_cache_meta["primary_label"].isin(LABEL2IDX)
    ].reset_index(drop=True)
    print(f"Focal audio cache: {len(audio_cache_meta)} entries")

    # ── 4. soundscape 5s 段元数据 ──────────────────────────────────
    sc_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_cache_meta.csv")
    sc_cache_meta["label_list"] = sc_cache_meta["label_list"].apply(
        lambda x: x.split(";") if isinstance(x, str) else []
    )
    print(f"Soundscape cache: {len(sc_cache_meta)} windows")

    # ── 5. 构造 labeled soundscape 标签矩阵 Y_SC ───────────────────
    # ★ 训练用比赛全量 train_soundscapes_labels.csv (66 个文件 = 59 full + 7 partial),
    #   跟 notebook / tuckerarrants 公开 ONNX 完全一致. CV 由 5-fold GroupKFold 提供.
    sc_labels_raw = pd.read_csv(TRAIN_SC_LABELS).drop_duplicates()
    sc_labels_raw["start_sec"] = (
        pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    )
    Y_SC = np.zeros((len(sc_cache_meta), NUM_CLASSES), dtype=np.float32)
    for i, row in sc_cache_meta.iterrows():
        matches = sc_labels_raw[
            (sc_labels_raw["filename"]  == row["filename"])
            & (sc_labels_raw["start_sec"] == row["start_sec"])
        ]
        for _, m in matches.iterrows():
            for lbl in str(m["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in LABEL2IDX:
                    Y_SC[i, LABEL2IDX[lbl]] = 1.0

    labeled_mask = Y_SC.sum(axis=1) > 0
    print(f"Soundscape labels: {labeled_mask.sum()}/{len(Y_SC)} windows labeled, "
          f"{int(Y_SC.sum())} positives, "
          f"{int((Y_SC.sum(axis=0) > 0).sum())} species  "
          f"[source: {TRAIN_SC_LABELS.name}]")

    # ── 6. Fold 分配 ───────────────────────────────────────────────
    # focal: StratifiedKFold by species (保证每折每类样本数均匀)
    # soundscape: GroupKFold by filename (66 个文件做 5-fold, 每折 ~13 个文件做 OOF 验证)
    audio_for_split = (audio_cache_meta.drop_duplicates("original_idx")
                                        .reset_index(drop=True))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    audio_for_split["fold"] = -1
    for fold, (_, val_idx) in enumerate(
            skf.split(audio_for_split, audio_for_split["primary_label"])):
        audio_for_split.loc[val_idx, "fold"] = fold
    audio_cache_meta = audio_cache_meta.merge(
        audio_for_split[["original_idx", "fold"]],
        on="original_idx", how="left",
    )
    print(f"\nFocal fold distribution:\n"
          f"{audio_cache_meta['fold'].value_counts().sort_index()}")

    # soundscape: 66 个文件 → 5-fold (GroupKFold by filename, 防同文件跨 fold, 每折 ~13 个文件)
    sc_files = (sc_cache_meta[["filename", "site"]]
                .drop_duplicates().reset_index(drop=True))
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_cache_meta["fold"] = (sc_cache_meta["filename"].map(file_to_fold)
                              .fillna(-1).astype(int))
    print(f"\nSoundscape fold distribution ({sc_files['filename'].nunique()} files):\n"
          f"{sc_cache_meta['fold'].value_counts().sort_index()}")

    # ── 7. 稀有类上采样 (focal 端) ─────────────────────────────────
    # 样本数 < MIN_SAMPLE 的物种, 复制到至少有 MIN_SAMPLE 个
    counts       = audio_cache_meta["primary_label"].value_counts()
    rare_species = counts[counts < MIN_SAMPLE].index
    extra_rows   = []
    for sp in rare_species:
        sp_rows = audio_cache_meta[audio_cache_meta["primary_label"] == sp]
        n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
        for _ in range(n_copies):
            extra_rows.append(sp_rows)
    n_before = len(audio_cache_meta)
    if extra_rows:
        audio_cache_meta = pd.concat([audio_cache_meta] + extra_rows,
                                      ignore_index=True)
    print(f"\nUpsampled {len(rare_species)} rare species (min={MIN_SAMPLE}): "
          f"{n_before} -> {len(audio_cache_meta)} samples")

    # ── 8. S22 mask (评估时排除噪声站点 S22, 主指标用 non_s22_macro) ─
    sc_sites        = sc_cache_meta["site"].values
    non_s22_mask_sc = sc_sites != "S22"
    print(f"S22: {(~non_s22_mask_sc).sum()}, non-S22: {non_s22_mask_sc.sum()}")

    # ── 9. focal secondary labels lookup ─────────────────────────
    # train.csv 每行有 secondary_labels='[...]' 字符串, 解析成 list
    focal_secondary_labels = {}
    for idx, row in train_df.iterrows():
        sec = row.get("secondary_labels", "")
        if pd.isna(sec) or sec in ("", "[]"):
            continue
        try:
            sec_list = eval(sec) if isinstance(sec, str) else []
        except Exception:
            continue
        valid = [s for s in sec_list if s in LABEL2IDX]
        if valid:
            focal_secondary_labels[idx] = valid
    print(f"Focal secondary labels: {len(focal_secondary_labels)} files")

    # ── 10. SC MixUp 池 (Focal-Soundscape MixUp 用) ──────────────
    # 给 dataset.FocalDS 用: 每个 labeled soundscape 段进入混合池
    sc_mixup_sources = []
    if USE_FOCAL_SC_MIXUP:
        _sc_file_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_file_meta.csv")
        _sc_file_dict = dict(zip(_sc_file_meta["filename"], _sc_file_meta["cache_file"]))
        _labeled_rows = []
        for i in range(len(sc_cache_meta)):
            row = sc_cache_meta.iloc[i]
            if Y_SC[i].sum() > 0:                       # 有标签的段才进池
                cf = _sc_file_dict.get(row["filename"])
                if cf is not None:
                    _labeled_rows.append({
                        "filename":   row["filename"],
                        "start_sec":  int(row["start_sec"]),
                        "cache_file": cf,
                        "label_idx":  i,
                        "fold":       int(row.get("fold", -1)),
                    })
        if _labeled_rows:
            _labeled_meta = pd.DataFrame(_labeled_rows)
            sc_mixup_sources.append((WAVEFORM_CACHE_DIR, _labeled_meta, Y_SC))
            print(f"SC MixUp pool: {len(_labeled_meta)} labeled windows")

    # ── 11. DEBUG 子采样 (Kaggle 上调试用) ──────────────────────
    if DEBUG:
        # 每类只留 3 个 focal + 前 50 个 sc, 跑通流程用
        audio_cache_meta = (audio_cache_meta.groupby("primary_label")
                            .head(3).reset_index(drop=True))
        sc_cache_meta    = sc_cache_meta.head(50)
        Y_SC             = Y_SC[:50]
        non_s22_mask_sc  = non_s22_mask_sc[:50]
        print(f"DEBUG MODE: {len(audio_cache_meta)} focal, "
              f"{len(sc_cache_meta)} sc")

    return {
        "PRIMARY_LABELS":         PRIMARY_LABELS,
        "LABEL2IDX":              LABEL2IDX,
        "TAXON_MASKS":            TAXON_MASKS,
        "taxonomy":               taxonomy,
        "train_df":               train_df,
        "audio_cache_meta":       audio_cache_meta,
        "sc_cache_meta":          sc_cache_meta,
        "Y_SC":                   Y_SC,
        "non_s22_mask_sc":        non_s22_mask_sc,
        "focal_secondary_labels": focal_secondary_labels,
        "sc_mixup_sources":       sc_mixup_sources,
    }
