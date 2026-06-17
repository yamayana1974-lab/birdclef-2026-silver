"""训练 / 验证用的 PyTorch Dataset 与 DataLoader 构造工具
================================================================

- ``BirdTrainDataset`` : 随机裁 5s (不足则 0-pad 随机起点) 的训练集.
                          可选 40% 同类替换: 若启用 pseudo_pool, 对 XC 行
                          按 primary_label 同类替换为伪标签 5s 段, label 改成软概率.
- ``BirdValidDataset`` : 取前 5s (head-crop) 的验证集, 行为确定.
- ``get_data_loader``  : 根据 ``fold_id`` 把 ``train_df`` 拆成 trn / val 两个 DataLoader.

只读 wav 中必要的 5 秒 (``soundfile.seek + read``), 不全文件加载, 速度差距很大.
"""

from __future__ import annotations

import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
from torch.utils.data import DataLoader, Dataset

from .config import CFG, SAMPLING_RATE, SEGMENT_SEC
from .pseudo_pool import PseudoPool


class BirdTrainDataset(Dataset):
    """训练集: 每条样本随机裁剪 ``segment_sec`` 秒, 不足则 0-padding 随机起点.

    若传入 ``pseudo_pool``, 对 ``primary_labels[idx]`` 在 pseudo_birds 内的样本,
    以 ``replace_prob`` 概率走"同类替换"路径:
        wave  = 该类某条伪标签 5s 段
        label = 234 维软概率向量 (覆盖原 multi-hot, < 0.1 已 trim)
    其它情况 (XC 类不在 pseudo_birds, 或概率没触发, 或 soundscape 5s 行)
    走原来的随机裁剪路径.
    """

    def __init__(
        self,
        paths,
        labels,
        sampling_rate: int = SAMPLING_RATE,
        segment_sec: int = SEGMENT_SEC,
        primary_labels: tp.Optional[tp.Sequence[str]] = None,
        is_focal_xc: tp.Optional[tp.Sequence[bool]] = None,
        pseudo_pool: tp.Optional[PseudoPool] = None,
        replace_prob: float = 0.4,
        fold_id_for_exclusion: tp.Optional[int] = None,
    ):
        self.paths = paths
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.segment_sec = segment_sec
        # 同类替换需要的元数据 (跟 paths/labels 等长, 一一对应)
        self.primary_labels = primary_labels
        self.is_focal_xc = is_focal_xc
        self.pseudo_pool = pseudo_pool
        self.replace_prob = replace_prob
        self.fold_id_for_exclusion = fold_id_for_exclusion

        # 校验启用替换时元数据齐全
        if self.pseudo_pool is not None:
            assert self.primary_labels is not None, (
                "primary_labels required when pseudo_pool is set"
            )
            assert self.is_focal_xc is not None, (
                "is_focal_xc required when pseudo_pool is set"
            )
            assert len(self.primary_labels) == len(self.paths)
            assert len(self.is_focal_xc) == len(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        # ─── 同类替换分支 (仅 XC focal + 该类在 pseudo_birds 内 + 命中概率) ───
        if (
            self.pseudo_pool is not None
            and bool(self.is_focal_xc[idx])
            and np.random.random() < self.replace_prob
        ):
            cls = str(self.primary_labels[idx])
            if self.pseudo_pool.has_label(cls):
                picked = self.pseudo_pool.sample_for_label(
                    cls,
                    exclude_fold_id=self.fold_id_for_exclusion,
                )
                if picked is not None:
                    sc_ogg, start_sec, soft_target = picked
                    wave = self._load_audio_segment(sc_ogg, start_sec)
                    if wave is not None:
                        return {
                            "wave": wave,
                            "label": soft_target.astype(np.float32),
                        }
                    # 文件不存在则 fallthrough 走原 XC 路径

        # ─── 原路径: 随机 5s crop XC 文件 (或 soundscape 5s 行) ───
        path = self.paths[idx]
        wave = self._load_audio_file(path)
        label = self.labels[idx].astype(np.float32)
        return {"wave": wave, "label": label}

    def _load_audio_file(self, path: str) -> np.ndarray:
        duration = self.sampling_rate * self.segment_sec
        with soundfile.SoundFile(path) as f:
            n_frames = f.frames
            if n_frames < duration:
                # 短于目标长度: 0-pad, 起点随机.
                # 注意 ogg metadata 里 frames 可能与实际解码不一致, 这里不假设
                # f.read() 一定返回 n_frames 样本, 用真实长度填.
                data = f.read(dtype="float32")
                n_actual = len(data)
                wave = np.zeros(duration, dtype="float32")
                if n_actual >= duration:
                    wave[:] = data[:duration]
                else:
                    start = np.random.randint(duration - n_actual + 1)
                    wave[start : start + n_actual] = data
            else:
                # 长够: 在文件内随机裁剪.
                start = np.random.randint(n_frames - duration + 1)
                f.seek(start)
                wave = f.read(frames=duration, dtype="float32")
                # 极端情况下指定 frames 也可能少读, 兜底 pad 一下.
                if len(wave) < duration:
                    pad = np.zeros(duration - len(wave), dtype="float32")
                    wave = np.concatenate([wave, pad], axis=0)
        return wave

    def _load_audio_segment(self, path, start_sec: int) -> tp.Optional[np.ndarray]:
        """从 path 切固定起点的 5s 段 (用于伪标签替换). 不存在则 None."""
        duration = self.sampling_rate * self.segment_sec
        path = Path(path)
        if not path.exists():
            return None
        try:
            with soundfile.SoundFile(str(path)) as f:
                n_frames = f.frames
                start = int(start_sec) * self.sampling_rate
                if start < 0:
                    start = 0
                if start >= n_frames:
                    return None
                f.seek(start)
                wave = f.read(frames=duration, dtype="float32")
                if wave.ndim > 1:
                    wave = wave.mean(axis=1)
                if len(wave) < duration:
                    pad = np.zeros(duration - len(wave), dtype="float32")
                    wave = np.concatenate([wave, pad], axis=0)
                return wave.astype("float32")
        except Exception as e:
            print(f"[dataset] WARN failed to load pseudo seg {path} @ {start_sec}s: {e}")
            return None


class BirdValidDataset(Dataset):
    """验证集: 取每条 wav 的前 ``segment_sec`` 秒, 行为确定, 利于 OOF 评估."""

    def __init__(self, paths, labels, sampling_rate: int = SAMPLING_RATE, segment_sec: int = SEGMENT_SEC):
        self.paths = paths
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.segment_sec = segment_sec

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        wave = self._load_audio_file(path)
        label = self.labels[idx]
        return {"wave": wave, "label": label.astype(np.float32)}

    def _load_audio_file(self, path: str) -> np.ndarray:
        duration = self.sampling_rate * self.segment_sec
        with soundfile.SoundFile(path) as f:
            n_frames = f.frames
            if n_frames < duration:
                # ogg metadata 里 frames 可能与实际解码不一致, 用真实长度填.
                data = f.read(dtype="float32")
                n_actual = len(data)
                wave = np.zeros(duration, dtype="float32")
                wave[: min(n_actual, duration)] = data[: min(n_actual, duration)]
            else:
                wave = f.read(frames=duration, dtype="float32")
                if len(wave) < duration:
                    pad = np.zeros(duration - len(wave), dtype="float32")
                    wave = np.concatenate([wave, pad], axis=0)
        return wave


# ============================================================
#                  DataLoader 构造
# ============================================================
def get_data_loader(
    train_df: pd.DataFrame,
    fold_id: int,
    classes: tp.List[str],
    cfg=CFG,
    pseudo_pool: tp.Optional[PseudoPool] = None,
) -> tp.Tuple[DataLoader, DataLoader]:
    """按 fold_id 拆出 train / val 子集并返回 DataLoader.

    Args:
        train_df : 包含 ``file_path`` / ``fold`` / 234 类 one-hot 列的 DataFrame.
                   若启用伪标签替换, 还需要 ``primary_label`` 列以及一个能
                   判断 "是否 XC focal" 的列 (这里通过 filename 后缀检测).
        fold_id  : 0..N_FOLDS-1, 表示当前折索引 (该折作验证).
        classes  : 长度 234 的类名列表, 决定 multi-hot 标签的列顺序.
        cfg      : 训练超参 (主要用 batch_size / num_workers / use_pseudo_replace
                   / pseudo_replace_prob).
        pseudo_pool : 可选, 启用同类替换时传入. 默认 None = 关掉.
    """
    trn_idxs = train_df.query("fold != @fold_id").index.values
    val_idxs = train_df.query("fold == @fold_id").index.values

    file_paths = train_df["file_path"].values.tolist()
    labels_arr = train_df[classes].values

    trn_paths = [file_paths[idx] for idx in trn_idxs]
    trn_labels = [labels_arr[idx] for idx in trn_idxs]
    val_paths = [file_paths[idx] for idx in val_idxs]
    val_labels = [labels_arr[idx] for idx in val_idxs]

    # 同类替换需要的元数据 (跟 trn_paths 等长)
    trn_primary_labels = None
    trn_is_focal_xc = None
    if pseudo_pool is not None:
        trn_primary_labels = train_df.iloc[trn_idxs]["primary_label"].astype(str).tolist()
        # XC focal 行通过 filename 后缀 .ogg 判定 (soundscape 行是 .wav)
        # USE_OGG_DIRECT=True 时, train_audio 是 .ogg; soundscape 切片仍是 .wav.
        # 兼容两种情况: 优先看 file_path 后缀.
        trn_is_focal_xc = [
            not str(p).lower().endswith(".wav")
            for p in trn_paths
        ]

    trn_dataset = BirdTrainDataset(
        trn_paths, trn_labels,
        primary_labels=trn_primary_labels,
        is_focal_xc=trn_is_focal_xc,
        pseudo_pool=pseudo_pool,
        replace_prob=getattr(cfg, "pseudo_replace_prob", 0.4),
        fold_id_for_exclusion=fold_id,
    )
    val_dataset = BirdValidDataset(val_paths, val_labels)

    trn_loader = DataLoader(
        trn_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    return trn_loader, val_loader
