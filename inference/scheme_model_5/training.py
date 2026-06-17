# -*- coding: utf-8 -*-
"""training.py — train_light_proto_ssm + run_tta_proto + train_residual_ssm.

对应 cell_07 第 1045-1143 行 (LightProtoSSM 训练 + 5-shift TTA + temporal flip,
ResidualSSM 训练 + correction grid search 在 ``protossm_pipeline.py`` 里).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import N_WINDOWS
from .models import LightProtoSSM, ResidualSSM


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# =============================================================================
# train_light_proto_ssm (cell_07 第 1045-1079 行)
# =============================================================================
def train_light_proto_ssm(
    emb_full: np.ndarray,    # emb_full <-> emb_tr
    scores_full: np.ndarray, # scores_full <-> sc_tr
    Y_full: np.ndarray,
    meta_full: pd.DataFrame,
    n_classes: int,
    n_epochs: int = 40,
    patience: int = 8,
    lr: float = 1e-3,
    n_sites: int = 20,
    verbose: bool = False,
) -> Tuple[LightProtoSSM, Dict[str, int]]:
    n_files  = len(emb_full) // N_WINDOWS
    emb_f    = emb_full.reshape   (n_files, N_WINDOWS, -1)
    log_f    = scores_full.reshape(n_files, N_WINDOWS, -1)
    lab_f    = Y_full.reshape     (n_files, N_WINDOWS, -1).astype(np.float32)

    fnames    = meta_full["filename"].unique()
    sites_u   = sorted(meta_full["site"].unique())
    site2i    = {s: i + 1 for i, s in enumerate(sites_u)}
    site_ids  = np.array([
        min(site2i.get(meta_full.loc[meta_full["filename"] == fn, "site"].iloc[0], 0),
            n_sites - 1)
        for fn in fnames
    ], dtype=np.int64)
    hour_ids  = np.array([
        int(meta_full.loc[meta_full["filename"] == fn, "hour_utc"].iloc[0]) % 24
        for fn in fnames
    ], dtype=np.int64)

    model = LightProtoSSM(
        n_classes=n_classes, n_sites=n_sites,
        use_cross_attn=True, cross_attn_heads=2,
    )
    model.init_prototypes(
        torch.tensor(emb_full, dtype=torch.float32),
        torch.tensor(Y_full,   dtype=torch.float32),
    )

    emb_t  = torch.tensor(emb_f,    dtype=torch.float32)
    log_t  = torch.tensor(log_f,    dtype=torch.float32)
    lab_t  = torch.tensor(lab_f,    dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    pos_cnt   = lab_t.sum(dim=(0, 1))
    total     = lab_t.shape[0] * lab_t.shape[1]
    pos_weight = ((total - pos_cnt) / (pos_cnt + 1)).clamp(max=25.0)

    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched  = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos",
    )
    best_loss, best_state, wait = float("inf"), None, 0

    swa_model  = torch.optim.swa_utils.AveragedModel(model)
    swa_start  = int(n_epochs * 0.65)
    swa_sched  = torch.optim.swa_utils.SWALR(opt, swa_lr=4e-4)

    for ep in range(n_epochs):
        model.train()
        out  = model(emb_t, log_t, site_ids=site_t, hours=hour_t)
        loss = F.binary_cross_entropy_with_logits(
            out, lab_t, pos_weight=pos_weight[None, None, :],
        ) + 0.15 * F.mse_loss(out, log_t)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if ep >= swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa_model)
        model = swa_model
    else:
        model.load_state_dict(best_state)
    model.eval()
    return model, site2i


# =============================================================================
# TTA — 5 shift + temporal flip (cell_07 第 1081-1110 行 含 Tweak F)
# =============================================================================
def run_tta_proto(
    proto_model,
    emb_files: np.ndarray,
    sc_files: np.ndarray,
    site_t: torch.Tensor,
    hour_t: torch.Tensor,
    shifts: List[int] = (0, 1, -1, 2, -2),
) -> np.ndarray:
    proto_model.eval()
    all_preds = []

    emb_t = torch.tensor(emb_files, dtype=torch.float32)
    sc_t  = torch.tensor(sc_files,  dtype=torch.float32)

    for shift in shifts:
        e = torch.roll(emb_t, shift, dims=1) if shift else emb_t
        s = torch.roll(sc_t,  shift, dims=1) if shift else sc_t
        with torch.no_grad():
            out = proto_model(e, s, site_ids=site_t, hours=hour_t).numpy()
        if shift:
            out = np.roll(out, -shift, axis=1)
        all_preds.append(out)

    # Tweak F: temporal flip
    with torch.no_grad():
        out_flip = proto_model(
            emb_t.flip(1), sc_t.flip(1), site_ids=site_t, hours=hour_t,
        ).numpy()
    all_preds.append(out_flip[:, ::-1, :].copy())

    return np.mean(all_preds, axis=0)


# =============================================================================
# train_residual_ssm (cell_07 第 1113-1141 行)
# =============================================================================
def train_residual_ssm(
    emb_full: np.ndarray,
    first_pass_flat: np.ndarray,
    Y_full: np.ndarray,
    site_ids: np.ndarray,
    hour_ids: np.ndarray,
    n_classes: int,
    n_epochs: int = 30,
    patience: int = 8,
    lr: float = 1e-3,
    correction_weight: float = 0.30,
    verbose: bool = False,
) -> Tuple[ResidualSSM, float]:
    n_files = len(emb_full) // N_WINDOWS
    emb_f   = emb_full.reshape       (n_files, N_WINDOWS, -1)
    fp_f    = first_pass_flat.reshape(n_files, N_WINDOWS, -1)
    lab_f   = Y_full.reshape         (n_files, N_WINDOWS, -1).astype(np.float32)
    fp_prob = 1.0 / (1.0 + np.exp(-np.clip(fp_f, -30, 30)))
    residuals = lab_f - fp_prob

    n_val = max(1, int(n_files * 0.15))
    rng   = torch.Generator(); rng.manual_seed(42)
    perm  = torch.randperm(n_files, generator=rng).numpy()
    val_i, train_i = perm[:n_val], perm[n_val:]

    emb_t  = torch.tensor(emb_f,     dtype=torch.float32)
    fp_t   = torch.tensor(fp_f,      dtype=torch.float32)
    res_t  = torch.tensor(residuals, dtype=torch.float32)
    site_t = torch.tensor(site_ids,  dtype=torch.long)
    hour_t = torch.tensor(hour_ids,  dtype=torch.long)

    model = ResidualSSM(n_classes=n_classes)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos",
    )
    best_loss, best_state, wait = float("inf"), None, 0

    for ep in range(n_epochs):
        model.train()
        corr = model(emb_t[train_i], fp_t[train_i],
                     site_ids=site_t[train_i], hours=hour_t[train_i])
        loss = F.mse_loss(corr, res_t[train_i])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_corr = model(emb_t[val_i], fp_t[val_i],
                             site_ids=site_t[val_i], hours=hour_t[val_i])
            val_loss = F.mse_loss(val_corr, res_t[val_i])

        if val_loss.item() < best_loss:
            best_loss  = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, correction_weight
