"""训练 / 验证用的 PyTorch Dataset 与 DataLoader 构造工具
================================================================

- ``BirdTrainDataset`` : 随机裁 5s (不足则 0-pad 随机起点) 的训练集.
- ``BirdValidDataset`` : 取前 5s (head-crop) 的验证集, 行为确定.
- ``get_data_loader``  : 根据 ``fold_id`` 把 ``train_df`` 拆成 trn / val 两个 DataLoader.

只读 wav 中必要的 5 秒 (``soundfile.seek + read``), 不全文件加载, 速度差距很大.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import pandas as pd
import soundfile
from torch.utils.data import DataLoader, Dataset

from .config import CFG, SAMPLING_RATE, SEGMENT_SEC


class BirdTrainDataset(Dataset):
    """训练集: 每条样本随机裁剪 ``segment_sec`` 秒, 不足则 0-padding 随机起点."""

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
) -> tp.Tuple[DataLoader, DataLoader]:
    """按 fold_id 拆出 train / val 子集并返回 DataLoader.

    Args:
        train_df : 包含 ``file_path`` / ``fold`` / 234 类 one-hot 列的 DataFrame.
        fold_id  : 0..N_FOLDS-1, 表示当前折索引 (该折作验证).
        classes  : 长度 234 的类名列表, 决定 multi-hot 标签的列顺序.
        cfg      : 训练超参 (主要用 batch_size / num_workers).
    """
    trn_idxs = train_df.query("fold != @fold_id").index.values
    val_idxs = train_df.query("fold == @fold_id").index.values

    file_paths = train_df["file_path"].values.tolist()
    labels_arr = train_df[classes].values

    trn_paths = [file_paths[idx] for idx in trn_idxs]
    trn_labels = [labels_arr[idx] for idx in trn_idxs]
    val_paths = [file_paths[idx] for idx in val_idxs]
    val_labels = [labels_arr[idx] for idx in val_idxs]

    trn_dataset = BirdTrainDataset(trn_paths, trn_labels)
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
