"""波形 -> 图像 的预处理 + MixUp
=================================

- ``LogMelSpectrogramTransform`` : 波形 batch -> (B, 1, 256, 256) 的 log-mel "图像",
  含 AmplitudeToDB / Resize / per-sample min-max 归一化.
- ``MixUp``                      : 频谱 + 多标签的 MixUp, ``lambda`` 取 Beta(alpha, alpha)
  并截到 [0.5, 1] 区间.
- ``dummy_mixup``                : warmup 阶段占位用的恒等 "MixUp".
"""

from __future__ import annotations

import typing as tp

import torch
import torchaudio
from torch import nn
from torchvision.transforms import v2 as tvt_v2


class LogMelSpectrogramTransform(nn.Module):
    """波形 batch -> log-mel 图像 batch, 形状 (B, 1, lms_shape[0], lms_shape[1]).

    流程:
        wave (B, T)
          -> MelSpectrogram (B, n_mels, time)
          -> AmplitudeToDB
          -> Resize -> (B, H, W)
          -> per-sample min-max 归一化到 [0, 1]
          -> 增加通道维 -> (B, 1, H, W)
    """

    def __init__(
        self,
        mel_spectrogram_params: tp.Dict,
        top_db: float,
        lms_shape: tp.Tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spectrogram_params)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
        self.resize = tvt_v2.Resize(size=lms_shape)

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_transform(wave)
        lms = self.db(mel_spec)        # (B, n_mels, time)
        lms = self.resize(lms)         # (B, H, W)

        # per-sample min-max 归一化.
        batch_size = lms.shape[0]
        lms_flat = lms.reshape(batch_size, -1)
        lms_min = lms_flat.min(dim=1)[0][:, None, None]
        lms_max = lms_flat.max(dim=1)[0][:, None, None]
        lms = (lms - lms_min) / (lms_max - lms_min + 1e-7)

        return lms[:, None, :, :]  # (B, 1, H, W)


class MixUp(nn.Module):
    """对 log-mel 图谱与多标签做 MixUp.

    - lambda ~ Beta(alpha, alpha), 再取 max(lambda, 1-lambda) 让主体更明显.
    - label 中 >= theta 的位置截断为 1, 保留多标签的二值语义.
    """

    def __init__(self, alpha: float = 0.5, theta: float = 1.0):
        super().__init__()
        self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)
        self.theta = theta

    def forward(self, lms: torch.Tensor, label: torch.Tensor):
        batch_size = lms.shape[0]
        device = lms.device

        lambda_tensor = self.beta_dist.sample(sample_shape=(batch_size,)).to(device)
        lambda_tensor = torch.maximum(lambda_tensor, 1 - lambda_tensor).float()

        shuffle_idxs = torch.randperm(batch_size).to(device)

        # 图谱 (B, C, F, T) 的混合.
        lms_lambda = lambda_tensor[:, None, None, None]
        lms = lms_lambda * lms + (1 - lms_lambda) * lms[shuffle_idxs]

        # 标签 (B, N_CLS) 的混合.
        label_lambda = lambda_tensor[..., None]
        label = label_lambda * label + (1 - label_lambda) * label[shuffle_idxs]
        label[label >= self.theta] = 1

        return lms, label


def dummy_mixup(lms: torch.Tensor, label: torch.Tensor):
    """占位用恒等 mixup, 用于 warmup 阶段不做混合."""
    return lms, label
