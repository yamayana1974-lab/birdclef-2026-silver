"""模型定义
================

- ``GeMPooling``   : Generalized Mean Pooling.
- ``AttnSEDHead`` + ``AttnSEDModel`` : SED 风格的 attention head + timm backbone.
- ``LSEPooling`` + ``LSEHead`` + ``LSEModel`` : 时间维 LSE pooling 的 head + timm backbone.
- ``CustomBCEWithLogitsLoss``        : 带时间轴 LSE 辅助监督的 BCE Loss (供 AttnSEDModel 用).

Baseline 训练用的是 ``LSEModel`` + ``nn.BCEWithLogitsLoss``;
``AttnSED*`` 与 ``CustomBCEWithLogitsLoss`` 保留是为了方便对比实验.
"""

from __future__ import annotations

import math
import typing as tp

import timm
import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
#                     1. GeM Pooling
# ============================================================
class GeMPooling(nn.Module):
    """Generalized Mean Pooling, p=1 -> mean, p->inf -> max.

    输入: (B, C, H, W); 输出: (B, C). 默认沿 (H, W) pool, 也可以指定 mean_axis.
    """

    def __init__(self, init_p: float = 3.0, mean_axis: tp.Tuple = (1, 2), eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(init_p))
        self.mean_axis = mean_axis
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # clamp p >= 1 防止幂运算崩溃.
        _ = self.p.clip(min=1.0)
        h = h.clip(min=self.eps).pow(self.p)
        h = h.mean(dim=self.mean_axis)
        h = h.pow(1.0 / self.p)
        return h


# ============================================================
#                  2. AttnSED Head / Model
# ============================================================
class AttnSEDHead(nn.Module):
    """Attention-style SED head: 在时间轴上学到 attention 权重再加权求和.

    Input : (B, C, Time)
    Output: logits (B, N_CLASSES), timewise_logits (B, Time, N_CLASSES)
    """

    def __init__(self, num_features: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.pre_fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_fc = nn.Linear(num_features, num_classes)
        self.cls_fc = nn.Linear(num_features, num_classes)

    def forward(self, h: torch.Tensor):
        h = h.permute(0, 2, 1)              # (B, Time, C)
        h = self.pre_fc(h)
        att_w = torch.tanh(self.att_fc(h))  # (B, Time, N_CLASSES)
        att_w = F.softmax(att_w, dim=1)     # 时间维 softmax.
        timewise_logits = self.cls_fc(h)
        logits = (att_w * timewise_logits).sum(dim=1)
        return logits, timewise_logits


class AttnSEDModel(nn.Module):
    """timm backbone + GeMPooling(频率轴) + AttnSEDHead."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_path_rate: float = 0.0,
        head_dropout: float = 0.1,
        num_classes: int = 234,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=1,
            global_pool="", num_classes=0, drop_path_rate=drop_path_rate,
        )
        # 部分 backbone 的 num_features 与实际输出不一致, dummy 跑一遍取真实通道.
        dummy_input = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            dummy_output = self.backbone(dummy_input)
        num_features = dummy_output.shape[1]
        print(f"num_features: {self.backbone.num_features}, dummy_output's dim: {num_features}")

        self.gem_pool = GeMPooling(mean_axis=2)
        self.head = AttnSEDHead(num_features, num_classes, head_dropout)

    def forward_for_training(self, x: torch.Tensor):
        h = self.backbone(x)        # (B, C, F', T')
        h = self.gem_pool(h)        # (B, C, T')
        logits, twl = self.head(h)
        return logits, twl

    def forward(self, x: torch.Tensor):
        logits, _ = self.forward_for_training(x)
        return logits


# ============================================================
#                  3. LSE Head / Model
# ============================================================
class LSEPooling(nn.Module):
    """Log-Sum-Exp pooling, 温度可控的 soft-max.

        y = T * (logsumexp(x / T, dim) - log(N))

    T -> 0   时趋近于 max;
    T -> inf 时趋近于 mean.
    """

    def __init__(self, pool_axis: int = -1, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.pool_axis = pool_axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.temperature * (
            torch.logsumexp(x / self.temperature, axis=self.pool_axis)
            - math.log(x.shape[self.pool_axis])
        )
        return y


class LSEHead(nn.Module):
    """先把频率轴 mean 掉, 再在时间轴上做 LSE pooling, 得到 (B, N_CLS)."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dropout: float = 0.2,
        lse_temperature: float = 1.0,
    ):
        super().__init__()
        self.lse_pool = LSEPooling(pool_axis=1, temperature=lse_temperature)
        self.cls_fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, C, Freq, Time)
        h = torch.mean(h, axis=2)                # 频率轴 mean -> (B, C, Time)
        h = h.transpose(1, 2)                    # -> (B, Time, C)
        timewise_logits = self.cls_fc(h)         # (B, Time, N_CLS)
        logits = self.lse_pool(timewise_logits)  # 时间维 LSE -> (B, N_CLS)
        return logits


class LSEModel(nn.Module):
    """Baseline 实际使用的模型: timm backbone + LSEHead."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_path_rate: float = 0.0,
        num_classes: int = 234,
        head_dropout: float = 0.2,
        lse_temperature: float = 1.0,
    ):
        super().__init__()
        # 部分 timm backbone (如 RepVit) 不接受 drop_path_rate kwarg.
        # 先带着 drop_path_rate 试一次, 抛 TypeError 就回退到不带.
        common_kwargs = dict(
            in_chans=1, global_pool="", num_classes=0,
        )

        # 检查是否需要从本地目录加载 (HF Hub 下载不稳时的备选).
        # 设 HGNET_LOCAL_PRETRAINED_DIR=<dir>, dir 下应有按 model_name 命名的子目录,
        # 子目录里有 model.safetensors 或 pytorch_model.bin.
        # ModelScope 下载的目录会把 "." 替换成 "___", 这里也兼容.
        import os
        local_dir_root = os.environ.get("HGNET_LOCAL_PRETRAINED_DIR", "")
        local_state_dict = None
        if pretrained and local_dir_root:
            from pathlib import Path as _Path
            candidates = [
                _Path(local_dir_root) / model_name,
                _Path(local_dir_root) / model_name.replace(".", "___"),
            ]
            for cand in candidates:
                if cand.exists():
                    sft = cand / "model.safetensors"
                    binp = cand / "pytorch_model.bin"
                    if sft.exists():
                        from safetensors.torch import load_file as _safe_load
                        local_state_dict = _safe_load(str(sft))
                        print(f"[LSEModel] loading local weights from {sft}")
                        break
                    if binp.exists():
                        local_state_dict = torch.load(str(binp), map_location="cpu")
                        print(f"[LSEModel] loading local weights from {binp}")
                        break
            if local_state_dict is None:
                print(f"[LSEModel] HGNET_LOCAL_PRETRAINED_DIR set but no weights found "
                      f"for {model_name}, fallback to HF Hub.")

        # 用本地权重时, 让 timm 不去网络拉取.
        effective_pretrained = pretrained if local_state_dict is None else False

        try:
            self.backbone = timm.create_model(
                model_name, pretrained=effective_pretrained,
                drop_path_rate=drop_path_rate, **common_kwargs,
            )
        except TypeError as e:
            if "drop_path_rate" not in str(e):
                raise
            if drop_path_rate > 0:
                print(
                    f"[LSEModel] backbone '{model_name}' does not support "
                    f"drop_path_rate, the value {drop_path_rate} is ignored."
                )
            self.backbone = timm.create_model(
                model_name, pretrained=effective_pretrained, **common_kwargs,
            )

        # 用本地 state_dict 给 backbone 加载权重 (跳过 head 相关 key).
        if local_state_dict is not None:
            # in_chans=1, 但预训练是 3 通道 stem. 复刻 timm 内置的 ``adapt_input_conv``
            # 逻辑: 沿通道 SUM (不是 mean), 这样 stem 在 1ch mel 输入下激活
            # magnitude 与 RGB 时近似一致, 前期梯度不会被弱化.
            adapted = dict(local_state_dict)
            for k, v in list(adapted.items()):
                if v.ndim == 4 and v.shape[1] == 3:
                    # 第一个出现 in_chans=3 的卷积视为 stem, 改完就跳出.
                    target_in = self.backbone.state_dict().get(k, None)
                    if target_in is not None and target_in.shape[1] == 1:
                        adapted[k] = v.sum(dim=1, keepdim=True)
                        print(f"[LSEModel] adapt stem '{k}' from 3ch to 1ch by SUM "
                              f"(matches timm.adapt_input_conv).")
                        break
            missing, unexpected = self.backbone.load_state_dict(adapted, strict=False)
            if unexpected:
                print(f"[LSEModel] {len(unexpected)} unexpected keys ignored "
                      f"(usually classifier head): {unexpected[:3]}...")

        dummy_input = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            dummy_output = self.backbone(dummy_input)
        num_features = dummy_output.shape[1]

        self.head = LSEHead(num_features, num_classes, head_dropout, lse_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)   # (B, C, Freq, Time)
        return self.head(h)


# ============================================================
#               4. 带时间轴辅助监督的 BCE Loss
# ============================================================
class CustomBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """带时间轴 LSE 辅助监督的 BCE Loss, 供 ``AttnSEDModel`` 使用.

        loss = (1 - w) * BCE(logits, label) + w * BCE(LSE(timewise_logits), label)
    """

    def __init__(self, timewise_weight: float = 0.5, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.timewise_weight = timewise_weight
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, timewise_logits: torch.Tensor, label: torch.Tensor):
        loss = super().forward(logits, label)
        logits_timeaxis_lse = self.temperature * (
            torch.logsumexp(timewise_logits / self.temperature, axis=1)
            - math.log(timewise_logits.shape[1])
        )
        loss_timeaxis_lse = super().forward(logits_timeaxis_lse, label)
        return (1 - self.timewise_weight) * loss + self.timewise_weight * loss_timeaxis_lse
