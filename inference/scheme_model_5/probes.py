# -*- coding: utf-8 -*-
"""probes.py — MLP probes (Tweak E: 按 n_pos 切 (256,128) vs (128,64)).

对应 cell_07 第 713-953 行: ``train_mlp_probes`` (含 Tweak E 双架构),
``VectorizedMLPProbes`` (PyTorch 包装), ``apply_mlp_probes_vectorized``
(按 architecture 分组), ``calibrate_and_optimize_thresholds``,
``apply_per_class_thresholds``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .config import N_WINDOWS


# =============================================================================
# Helpers
# =============================================================================
def build_class_freq_weights(Y: np.ndarray, cap: float = 10.0) -> np.ndarray:
    pos_count = Y.sum(axis=0).astype(np.float32) + 1.0
    freq      = pos_count / Y.shape[0]
    weights   = np.clip(1.0 / (freq ** 0.5), 1.0, cap)
    return (weights / weights.mean()).astype(np.float32)


def build_sequential_features(scores_col: np.ndarray, n_windows: int = N_WINDOWS):
    x     = scores_col.reshape(-1, n_windows)
    prev  = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    next_ = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    mean  = np.repeat(x.mean(axis=1), n_windows)
    max_  = np.repeat(x.max(axis=1),  n_windows)
    std   = np.repeat(x.std(axis=1),  n_windows)
    return prev.reshape(-1), next_.reshape(-1), mean, max_, std


# =============================================================================
# Train MLP probes (Tweak E)
# =============================================================================
def train_mlp_probes(
    emb: np.ndarray,
    scores_raw: np.ndarray,
    Y: np.ndarray,
    min_pos: int = 5,
    pca_dim: int = 64,
    alpha_blend: float = 0.4,
):
    """notebook 第 736-806 行. 返回 ``(probe_models, scaler, pca, alpha_blend)``."""
    scaler = StandardScaler()
    emb_s  = scaler.fit_transform(emb)

    pca = PCA(n_components=min(pca_dim, emb_s.shape[1] - 1))
    Z   = pca.fit_transform(emb_s).astype(np.float32)
    print(f"[probes] embedding {emb.shape} → PCA {Z.shape}  "
          f"(variance retained: {pca.explained_variance_ratio_.sum():.2%})")

    class_weights = build_class_freq_weights(Y, cap=10.0)
    probe_models  = {}
    active        = np.where(Y.sum(axis=0) >= min_pos)[0]
    MAX_ROWS      = 3000

    for ci in tqdm(active, desc="MLP probes"):
        y = Y[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        X = np.hstack([
            Z,
            scores_raw[:, ci:ci + 1],
            prev[:,  None],
            next_[:, None],
            mean[:,  None],
            max_[:,  None],
            std[:,   None],
        ])

        n_pos   = int(y.sum())
        n_neg   = len(y) - n_pos
        pos_idx = np.where(y == 1)[0]
        w       = float(class_weights[ci])
        repeat  = max(1, min(int(round(w * n_neg / max(n_pos, 1))), 8))
        if n_pos * repeat + len(y) > MAX_ROWS:
            repeat = max(1, (MAX_ROWS - len(y)) // max(n_pos, 1))

        X_bal = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(n_pos * repeat, dtype=y.dtype)])

        # Tweak E: wider net for frequent classes
        hidden = (256, 128) if n_pos >= 50 else (128, 64)
        clf = MLPClassifier(
            hidden_layer_sizes  = hidden,
            activation          = "relu",
            max_iter            = 300,
            early_stopping      = True,
            validation_fraction = 0.15,
            n_iter_no_change    = 15,
            random_state        = 42,
            learning_rate_init  = 5e-4,
            alpha               = 0.005,
        )
        clf.fit(X_bal, y_bal)
        probe_models[int(ci)] = clf

    print(f"[probes] trained {len(probe_models)} MLP probes (Tweak E dual-arch)")
    return probe_models, scaler, pca, alpha_blend


# =============================================================================
# VectorizedMLPProbes (单一 arch 组), 按 arch 分组的 wrapper
# =============================================================================
class VectorizedMLPProbes(nn.Module):
    """同一 hidden-layer arch 的一组 probes 堆成 PyTorch ParameterList."""

    def __init__(self, probe_models: Dict):
        super().__init__()
        self.valid_classes = sorted(probe_models.keys())
        V = len(self.valid_classes)
        if V == 0:
            self.weights = nn.ParameterList()
            self.biases  = nn.ParameterList()
            self.n_layers = 0
            return

        sample        = probe_models[self.valid_classes[0]]
        self.n_layers = len(sample.coefs_)
        self.weights  = nn.ParameterList()
        self.biases   = nn.ParameterList()
        for li in range(self.n_layers):
            W = np.stack([probe_models[c].coefs_[li]      for c in self.valid_classes], axis=0)
            b = np.stack([probe_models[c].intercepts_[li] for c in self.valid_classes], axis=0)
            self.weights.append(nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False))
            self.biases. append(nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False))

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)


def _run_probe_group(group_models: Dict, valid_classes_group: List[int],
                     scores_test: np.ndarray, Z_test: np.ndarray, N: int,
                     n_windows: int = N_WINDOWS) -> np.ndarray:
    """notebook 第 856-880 行 — 输出 ``(Vg, N)`` preds."""
    Vg = len(valid_classes_group)
    raw_g = scores_test[:, valid_classes_group].T
    n_files = N // n_windows
    raw_view_g = raw_g.reshape(Vg, n_files, n_windows)

    prev_g = np.concatenate([raw_view_g[:, :, :1], raw_view_g[:, :, :-1]], axis=2).reshape(Vg, N)
    nxt_g  = np.concatenate([raw_view_g[:, :, 1:], raw_view_g[:, :, -1:]], axis=2).reshape(Vg, N)
    mean_g = np.repeat(raw_view_g.mean(axis=2), n_windows, axis=1)
    mx_g   = np.repeat(raw_view_g.max(axis=2),  n_windows, axis=1)
    std_g  = np.repeat(raw_view_g.std(axis=2),  n_windows, axis=1)

    scalar_g  = np.stack([raw_g, prev_g, nxt_g, mean_g, mx_g, std_g], axis=-1).astype(np.float32)
    Z_exp_g   = np.broadcast_to(Z_test, (Vg, N, Z_test.shape[1]))
    X_g       = np.concatenate([Z_exp_g.astype(np.float32), scalar_g], axis=-1)

    vec_probe = VectorizedMLPProbes(group_models).eval()
    with torch.no_grad():
        preds_g = vec_probe(torch.tensor(X_g)).numpy()
    return preds_g


def apply_mlp_probes_vectorized(
    emb_test: np.ndarray,
    scores_test: np.ndarray,
    probe_models: Dict,
    scaler: StandardScaler,
    pca: PCA,
    alpha_blend: float = 0.4,
) -> np.ndarray:
    """按 arch 分组分别向量化 (Tweak E fix), 然后 alpha-blend."""
    if len(probe_models) == 0:
        return scores_test.copy()

    Z_test = pca.transform(scaler.transform(emb_test)).astype(np.float32)
    N      = len(scores_test)
    result = scores_test.copy()

    def _arch_key(clf):
        return tuple(w.shape[1] for w in clf.coefs_)

    groups: Dict = defaultdict(dict)
    for ci, clf in probe_models.items():
        groups[_arch_key(clf)][ci] = clf

    for arch, group_models in groups.items():
        valid_classes_group = sorted(group_models.keys())
        preds_g = _run_probe_group(group_models, valid_classes_group, scores_test, Z_test, N)
        result[:, valid_classes_group] = (
            (1.0 - alpha_blend) * scores_test[:, valid_classes_group]
            + alpha_blend * preds_g.T
        )
    return result


# =============================================================================
# Per-class thresholds (cell_07 第 924-953 行)
# =============================================================================
def calibrate_and_optimize_thresholds(
    oof_probs: np.ndarray,
    Y_FULL: np.ndarray,
    threshold_grid=None,
    n_windows: int = N_WINDOWS,
) -> np.ndarray:
    if threshold_grid is None:
        threshold_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    n_samples, n_cls = oof_probs.shape
    thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    n_files    = n_samples // n_windows
    file_oof   = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y     = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)
    n_calibrated = 0
    for c in range(n_cls):
        y_true, y_prob = file_y[:, c], file_oof[:, c]
        if y_true.sum() < 3:
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_prob, y_true)
            y_cal = ir.transform(y_prob)
        except Exception:
            y_cal = y_prob
        best_f1, best_t = 0.0, 0.5
        for t in threshold_grid:
            pred = (y_cal >= t).astype(int)
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
        n_calibrated += 1
    print(f"[probes] calibrated {n_calibrated} classes  mean={thresholds.mean():.3f}  "
          f"range=[{thresholds.min():.2f}, {thresholds.max():.2f}]")
    return thresholds


def apply_per_class_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    C = scores.shape[1]
    scaled = np.copy(scores)
    for c in range(C):
        t     = thresholds[c]
        above = scores[:, c] > t
        scaled[above, c]  = 0.5 + 0.5 * (scores[above, c]  - t) / (1 - t + 1e-8)
        scaled[~above, c] = 0.5 * scores[~above, c] / (t + 1e-8)
    return np.clip(scaled, 0.0, 1.0)
