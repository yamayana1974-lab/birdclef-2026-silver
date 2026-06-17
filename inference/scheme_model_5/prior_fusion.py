# -*- coding: utf-8 -*-
"""prior_fusion.py — Joint site-hour prior tables + post-processing helpers.

对应 cell_07 第 571-710 行 (``build_prior_tables`` with circular Gaussian smoothing,
``apply_prior``, ``smooth_predictions``, ``file_confidence_scale``,
``rank_aware_scaling``, ``adaptive_delta_smooth``, ``macro_auc``).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

from .config import N_WINDOWS


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")


def smooth_predictions(probs: np.ndarray, n_windows: int = N_WINDOWS,
                       alpha: float = 0.3) -> np.ndarray:
    N, C = probs.shape
    assert N % n_windows == 0
    view   = probs.reshape(-1, n_windows, C).copy()
    prev_w = np.concatenate([view[:, :1, :],  view[:, :-1, :]], axis=1)
    next_w = np.concatenate([view[:, 1:,  :], view[:, -1:, :]], axis=1)
    return ((1 - alpha) * view + 0.5 * alpha * (prev_w + next_w)).reshape(N, C)


def build_prior_tables(sc_df: pd.DataFrame, Y_labels: np.ndarray) -> Dict:
    """Joint site-hour 表 + 24h 圆形高斯平滑 (notebook 第 586-662 行)."""
    sc_df    = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)

    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p    = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n    = np.zeros(len(site_keys), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)

    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p    = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n    = np.zeros(len(hour_keys), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)

    sh_keys = sorted({
        (str(s), int(h))
        for s, h in zip(sc_df["site"].dropna(), sc_df["hour_utc"].dropna())
        if not pd.isna(s) and not pd.isna(h)
    })
    sh_to_i = {k: i for i, k in enumerate(sh_keys)}
    sh_p    = np.zeros((len(sh_keys), Y_labels.shape[1]), dtype=np.float32)
    sh_n    = np.zeros(len(sh_keys), dtype=np.float32)
    for (s, h) in sh_keys:
        i = sh_to_i[(s, h)]
        mask = (
            (sc_df["site"].astype(str).values == s)
            & (sc_df["hour_utc"].astype(int).values == h)
        )
        sh_n[i] = mask.sum()
        sh_p[i] = Y_labels[mask].mean(axis=0)

    # Circular Gaussian smoothing on hour priors, sigma=1.5
    if len(hour_keys) >= 3:
        full_hour_p = np.zeros((24, hour_p.shape[1]), dtype=np.float32)
        for h, i in hour_to_i.items():
            full_hour_p[int(h)] = hour_p[i]
        tiled        = np.tile(full_hour_p, (3, 1))
        tiled_smooth = gaussian_filter1d(tiled, sigma=1.5, axis=0, mode="wrap")
        full_smooth  = tiled_smooth[24:48]
        for h, i in hour_to_i.items():
            hour_p[i] = full_smooth[int(h)]
        hour_p = np.clip(hour_p, 0.0, 1.0)

    return {
        "global_p":  global_p,
        "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
        "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
        "sh_to_i":   sh_to_i,   "sh_p":   sh_p,   "sh_n":   sh_n,
    }


def apply_prior(scores: np.ndarray, sites, hours, tables: Dict,
                lambda_prior: float = 0.4) -> np.ndarray:
    eps = 1e-4
    n   = len(scores)
    out = scores.copy()
    p   = np.tile(tables["global_p"], (n, 1))

    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j  = tables["hour_to_i"][h]
            nh = tables["hour_n"][j]
            w  = nh / (nh + 8.0)
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]
    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j  = tables["site_to_i"][s]
            ns = tables["site_n"][j]
            w  = ns / (ns + 8.0)
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]
    if "sh_to_i" in tables:
        for i, (s, h) in enumerate(zip(sites, hours)):
            key = (str(s), int(h))
            if key in tables["sh_to_i"]:
                j   = tables["sh_to_i"][key]
                nsh = tables["sh_n"][j]
                w   = nsh / (nsh + 4.0)
                p[i] = w * tables["sh_p"][j] + (1 - w) * p[i]

    p = np.clip(p, eps, 1 - eps)
    out += lambda_prior * (np.log(p) - np.log1p(-p))
    return out.astype(np.float32)


# =============================================================================
# Post-processing
# =============================================================================
def file_confidence_scale(probs: np.ndarray, n_windows: int = N_WINDOWS,
                          top_k: int = 2, power: float = 0.4) -> np.ndarray:
    N, C = probs.shape
    view       = probs.reshape(-1, n_windows, C)
    sorted_v   = np.sort(view, axis=1)
    top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)
    return (view * np.power(top_k_mean, power)).reshape(N, C)


def rank_aware_scaling(probs: np.ndarray, n_windows: int = N_WINDOWS,
                       power: float = 0.5) -> np.ndarray:
    N, C = probs.shape
    view     = probs.reshape(-1, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)
    return (view * np.power(file_max, power)).reshape(N, C)


def adaptive_delta_smooth(probs: np.ndarray, n_windows: int = N_WINDOWS,
                          base_alpha: float = 0.20) -> np.ndarray:
    N, C   = probs.shape
    result = probs.copy()
    view   = probs.reshape(-1, n_windows, C)
    out    = result.reshape(-1, n_windows, C)
    for t in range(n_windows):
        conf  = view[:, t, :].max(axis=-1, keepdims=True)
        alpha = base_alpha * (1.0 - conf)
        if t == 0:
            neighbor_avg = (view[:, t, :] + view[:, t + 1, :]) / 2.0
        elif t == n_windows - 1:
            neighbor_avg = (view[:, t - 1, :] + view[:, t, :]) / 2.0
        else:
            neighbor_avg = (view[:, t - 1, :] + view[:, t + 1, :]) / 2.0
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg
    return result
