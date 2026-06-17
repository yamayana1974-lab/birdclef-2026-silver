# -*- coding: utf-8 -*-
"""models.py — SelectiveSSM / LightProtoSSM / ResidualSSM (Model_5 紧凑版).

对应 cell_07 第 957-1043 行. 跟 Model_2 的 ProtoSSMv2 区别:
* ``LightProtoSSM`` 固定 2 层 BiSSM + 2 层 cross-attn (notebook 写在了一行内)
* ``ResidualSSM`` 跟 Model_2 完全一致, 只是这里没有 ``count_parameters`` 方法
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SelectiveSSM
# =============================================================================
class SelectiveSSM(nn.Module):

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d  = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(d_model))
        self.B_proj   = nn.Linear(d_model, d_state, bias=False)
        self.C_proj   = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2))
        dt = F.softplus(self.dt_proj(x_conv))
        A  = -torch.exp(self.A_log)
        B  = self.B_proj(x_conv)
        C  = self.C_proj(x_conv)
        h  = torch.zeros(B_sz, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h  = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        return torch.stack(ys, dim=1) + x * self.D[None, None, :]


# =============================================================================
# LightProtoSSM (cell_07 第 978-1020 行, 2 层 SSM + cross-attn)
# =============================================================================
class LightProtoSSM(nn.Module):

    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 128,
        d_state: int = 16,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.15,
        n_sites: int = 20,
        meta_dim: int = 16,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 2,
    ):
        super().__init__()
        self.n_classes      = n_classes
        self.n_windows      = n_windows
        self.use_cross_attn = use_cross_attn

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc   = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb  = nn.Embedding(n_sites, meta_dim)
        self.hour_emb  = nn.Embedding(24,      meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_bwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(2)])
        self.ssm_norm  = nn.ModuleList([nn.LayerNorm(d_model)            for _ in range(2)])
        self.drop      = nn.Dropout(dropout)

        if use_cross_attn:
            self.cross_attn = nn.ModuleList([
                nn.MultiheadAttention(d_model, cross_attn_heads,
                                      dropout=dropout, batch_first=True)
                for _ in range(2)
            ])
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

        self.prototypes   = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp   = nn.Parameter(torch.tensor(5.0))
        self.class_bias   = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, emb_tensor, labels_tensor):
        with torch.no_grad():
            h = self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask = labels_tensor[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat(
                [self.site_emb(site_ids), self.hour_emb(hours)], dim=-1,
            ))
            h = h + meta[:, None, :]

        for i, (fwd, bwd, merge, norm) in enumerate(zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm,
        )):
            res = h
            hf  = fwd(h)
            hb  = bwd(h.flip(1)).flip(1)
            h   = self.drop(merge(torch.cat([hf, hb], dim=-1)))
            h   = norm(h + res)
            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)

        h_n  = F.normalize(h, dim=-1)
        p_n  = F.normalize(self.prototypes, dim=-1)
        sim  = torch.matmul(h_n, p_n.T) * F.softplus(self.proto_temp) + self.class_bias[None, None, :]

        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            out   = alpha * sim + (1 - alpha) * perch_logits
        else:
            out = sim
        return out


# =============================================================================
# ResidualSSM (cell_07 第 1022-1043 行)
# =============================================================================
class ResidualSSM(nn.Module):

    def __init__(
        self,
        d_input: int = 1536,
        d_scores: int = 234,
        d_model: int = 64,
        d_state: int = 8,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.1,
        n_sites: int = 20,
        meta_dim: int = 8,
    ):
        super().__init__()
        self.n_classes  = n_classes
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.site_emb  = nn.Embedding(n_sites, meta_dim)
        self.hour_emb  = nn.Embedding(24,      meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc   = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_fwd   = SelectiveSSM(d_model, d_state)
        self.ssm_bwd   = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm  = nn.LayerNorm(d_model)
        self.ssm_drop  = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat([
                self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1)),
                self.hour_emb(hours.clamp(0, 23)),
            ], dim=-1))
            h = h + meta.unsqueeze(1)
        res = h
        hf  = self.ssm_fwd(h)
        hb  = self.ssm_bwd(h.flip(1)).flip(1)
        h   = self.ssm_drop(self.ssm_merge(torch.cat([hf, hb], dim=-1)))
        h   = self.ssm_norm(h + res)
        return self.output_head(h)
