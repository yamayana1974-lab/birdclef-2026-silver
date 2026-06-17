"""通用小工具
====================

- ``set_random_seed``  : 统一固定 python/numpy/torch/cudnn 的随机种子.
- ``to_device``        : 递归把 tuple/dict/tensor 搬到指定 device.
- ``rank_normalize``   : 按列把矩阵做 rank-to-[0,1] 归一化, 用于融合 / OOF.
- ``sigmoid``          : 纯 numpy 版 sigmoid (推理阶段没必要带 torch).
- ``device``           : 全局自动选择的 torch.device.
"""

from __future__ import annotations

import os
import random
import typing as tp

import numpy as np
import pandas as pd
import torch


# ============================================================
#                     0. device
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================
#                     1. 随机种子
# ============================================================
def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """统一固定 python / numpy / torch / cudnn 的随机种子."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic


# ============================================================
#                     2. to_device
# ============================================================
def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor, ...], tp.Dict[str, torch.Tensor], torch.Tensor],
    device: torch.device,
    *args,
    **kwargs,
):
    """递归把 tuple / dict / tensor 搬到指定 device."""
    if isinstance(tensors, tuple):
        return tuple(t.to(device, *args, **kwargs) for t in tensors)
    if isinstance(tensors, dict):
        return {k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    return tensors.to(device, *args, **kwargs)


# ============================================================
#                     3. 数值工具
# ============================================================
def rank_normalize(x: np.ndarray) -> np.ndarray:
    """按列做秩归一化到 [0, 1], 并列取 max 秩.

    多折 / 多模型融合时常用: 把不同分类器的输出分布拉成均匀分布, 对 AUC 等基于排序
    的指标无害, 但能让分数更可比.
    """
    r_x = np.zeros_like(x)
    for i in range(x.shape[1]):
        r_x_i = pd.Series(x[:, i]).rank(method="max")
        r_x[:, i] = r_x_i / r_x_i.shape[0]
    return r_x


def sigmoid(x: np.ndarray) -> np.ndarray:
    """numpy 版 sigmoid (推理阶段把 OpenVINO 输出的 logits 转概率)."""
    return 1.0 / (1.0 + np.exp(-x))
