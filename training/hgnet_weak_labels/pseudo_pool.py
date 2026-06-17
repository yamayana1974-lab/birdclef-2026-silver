# -*- coding: utf-8 -*-
"""pseudo_pool.py - 加载伪标签 csv, 按 primary_label 建索引,
供 BirdTrainDataset 在 __getitem__ 里做 "40% 同类替换" 用.

伪标签 csv 字段 (merge_batches.py 产物):
    row_id, <234 类概率>, primary_label, primary_label_prob, fold_id

替换规则:
    1. 加载 csv 后按 primary_label 建索引: {label_name -> [row_idx, ...]}
    2. 训练 dataset 接到一条 XC 样本时, 看其 primary_label 是否在 pseudo_birds 内
    3. 是, 则以 prob=replace_prob 触发替换:
       a) 在 label2idx[label_name] 索引列表里随机挑一条
       b) 用 row_id 还原 (sc_filename.ogg, start_sec)
       c) 从 sc_dir 读取那段 5s wave
       d) target = 该行 234 维软概率向量 (< 0.1 已 trim, 已经在 csv 里)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class PseudoPool:
    """同类替换的伪标签池.

    Attributes:
        df: 过滤后的伪标签 DataFrame, 含 row_id / primary_label / 234 列概率.
        classes: 234 类名 (跟 train_df 的 multi-hot 列顺序保持一致).
        label2pseudo_idx: {primary_label_name -> np.ndarray[row_idx_in_df]}
        pseudo_birds: set, 含至少 1 个伪标签样本的类名 (= label2pseudo_idx.keys())
        soft_targets: np.ndarray (n_pseudo, n_classes), 跟 df 同行序的软概率矩阵 (float32)
        sample_ids: np.ndarray (n_pseudo,), 每行的 sc 文件 stem (= filename 去掉 .ogg)
        end_secs: np.ndarray (n_pseudo,), 每行的 5s 段结束秒
        sc_dir: soundscape ogg 目录, 替换时用于定位音频文件
    """

    def __init__(self, csv_path: Path, classes: List[str], sc_dir: Path):
        csv_path = Path(csv_path)
        sc_dir = Path(sc_dir)
        if not csv_path.exists():
            raise FileNotFoundError(f"pseudo csv not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # 必要列校验
        required = {"row_id", "primary_label", "primary_label_prob"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"pseudo csv missing columns: {missing}")

        # 234 类列顺序必须跟 train_df 的 classes 一致
        for cls in classes:
            if cls not in df.columns:
                raise ValueError(
                    f"pseudo csv missing class column: {cls}. "
                    f"Make sure pseudo csv has all 234 class columns matching taxonomy."
                )

        df = df.reset_index(drop=True)

        # 软标签矩阵 (n_pseudo, 234), 跟 classes 顺序对齐
        soft_targets = df[classes].values.astype(np.float32)

        # row_id = "{sc_stem}_{end_sec}" -> 还原
        sample_ids = np.empty(len(df), dtype=object)
        end_secs = np.zeros(len(df), dtype=np.int32)
        for i, rid in enumerate(df["row_id"].values):
            parts = rid.rsplit("_", 1)
            sample_ids[i] = parts[0]
            end_secs[i] = int(parts[1])

        # 按 primary_label 建索引 (np.ndarray, np.random.choice 用)
        label2pseudo_idx: Dict[str, np.ndarray] = {}
        for lbl, grp in df.groupby("primary_label"):
            label2pseudo_idx[lbl] = grp.index.to_numpy(dtype=np.int64)
        pseudo_birds: Set[str] = set(label2pseudo_idx.keys())

        self.df = df
        self.classes = classes
        self.label2pseudo_idx = label2pseudo_idx
        self.pseudo_birds = pseudo_birds
        self.soft_targets = soft_targets
        self.sample_ids = sample_ids
        self.end_secs = end_secs
        self.sc_dir = sc_dir

        n_total = len(df)
        n_classes_covered = len(pseudo_birds)
        n_classes_total = len(classes)
        print(
            f"[pseudo_pool] loaded {csv_path}\n"
            f"              rows={n_total}  unique primary_labels={n_classes_covered}/{n_classes_total}\n"
            f"              prob: min={df['primary_label_prob'].min():.3f} "
            f"median={df['primary_label_prob'].median():.3f} "
            f"max={df['primary_label_prob'].max():.3f}\n"
            f"              sc_dir={sc_dir}"
        )

    def has_label(self, primary_label: str) -> bool:
        """该类是否在伪标签池中 (= 至少有 1 个样本可替换)."""
        return primary_label in self.pseudo_birds

    def sample_for_label(
        self,
        primary_label: str,
        rng: Optional[np.random.Generator] = None,
        exclude_fold_id: Optional[int] = None,
    ) -> Optional[Tuple[Path, int, np.ndarray]]:
        """从指定 primary_label 的伪标签池中随机抽一条.

        Args:
            primary_label: XC 样本的 primary_label 名
            rng: 可选随机数生成器
            exclude_fold_id: 排除该 fold (防 OOF 泄漏). None = 不排除.

        Returns:
            (sc_ogg_path, start_sec, soft_target_234d) 或 None (类不存在或无候选).
        """
        if primary_label not in self.label2pseudo_idx:
            return None

        candidates = self.label2pseudo_idx[primary_label]

        if exclude_fold_id is not None:
            fold_ids = self.df.iloc[candidates]["fold_id"].astype(int).values
            keep_mask = fold_ids != exclude_fold_id
            if keep_mask.any():
                candidates = candidates[keep_mask]
            # 全被排除的极端情况: 退化成不排除 (保持训练能跑)

        if rng is None:
            j = int(np.random.choice(candidates))
        else:
            j = int(rng.choice(candidates))

        sc_ogg = self.sc_dir / f"{self.sample_ids[j]}.ogg"
        start_sec = int(self.end_secs[j]) - 5
        soft_target = self.soft_targets[j]
        return sc_ogg, start_sec, soft_target


def load_pseudo_pool(
    csv_path: Path,
    classes: List[str],
    sc_dir: Path,
) -> PseudoPool:
    """便捷函数 (跟 train.py 配合)."""
    return PseudoPool(csv_path=csv_path, classes=classes, sc_dir=sc_dir)
