# -*- coding: utf-8 -*-
"""protossm_pipeline.py — ProtoSSM full pipeline → ``submission_protossm.csv``.

对应 cell_07 第 1146-1323 行: Perch test → ProtoSSM TTA → prior + MLP probes →
ensemble per-class → ResidualSSM correction (含 Tweak C grid search) → 后处理 →
``apply_per_class_thresholds``.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .config import CFG, N_WINDOWS
from .perch_inference import PerchBackend, run_perch
from .prior_fusion import (
    adaptive_delta_smooth, apply_prior, build_prior_tables,
    file_confidence_scale, macro_auc, rank_aware_scaling,
)
from .probes import (
    apply_mlp_probes_vectorized, apply_per_class_thresholds,
    calibrate_and_optimize_thresholds, train_mlp_probes,
)
from .training import (
    run_tta_proto, sigmoid, train_light_proto_ssm, train_residual_ssm,
)


def run_protossm_pipeline(
    backend: PerchBackend,
    mapping: Dict,
    primary_labels: List[str],
    n_classes: int,
    base_dir: Path,
    sc: pd.DataFrame,
    Y_SC: np.ndarray,
    meta_tr: pd.DataFrame,
    sc_tr: np.ndarray,
    emb_tr: np.ndarray,
    Y_FULL_aligned: np.ndarray,
    output_csv: str = "submission_protossm.csv",
) -> Path:

    # ── 1. ProtoSSM training ──
    t0 = time.time()
    proto_model, site2i_tr = train_light_proto_ssm(
        emb_tr, sc_tr, Y_FULL_aligned, meta_tr, n_classes=n_classes,
        n_epochs=40, patience=8, lr=1e-3, verbose=False,
    )
    print(f"[proto] training {time.time() - t0:.1f}s")

    # ── 2. test paths (dry-run fallback) ──
    test_paths = sorted((base_dir / "test_soundscapes").glob("*.ogg"))
    if len(test_paths) == 0:
        n = CFG["dryrun_n_files"] or 20
        print(f"[proto] no hidden test — dry-run on {n} train files")
        test_paths = sorted((base_dir / "train_soundscapes").glob("*.ogg"))[:n]
    else:
        print(f"[proto] hidden test files: {len(test_paths)}")

    meta_te, sc_te, emb_te = run_perch(
        test_paths, backend, mapping, n_classes,
        batch_files=CFG["batch_files"], verbose=CFG["verbose"],
    )
    print(f"[proto] test scores: {sc_te.shape}")

    # ── 3. ProtoSSM TTA on test ──
    n_test_files  = len(sc_te) // N_WINDOWS
    emb_te_f      = emb_te.reshape(n_test_files, N_WINDOWS, -1)
    sc_te_f       = sc_te.reshape (n_test_files, N_WINDOWS, -1)
    test_fnames   = meta_te.drop_duplicates("filename")["filename"].tolist()
    n_sites_cap   = 20

    test_site_ids = np.array([
        min(site2i_tr.get(meta_te.loc[meta_te["filename"] == fn, "site"].iloc[0], 0),
            n_sites_cap - 1)
        for fn in test_fnames
    ], dtype=np.int64)
    test_hour_ids = np.array([
        int(meta_te.loc[meta_te["filename"] == fn, "hour_utc"].iloc[0]) % 24
        for fn in test_fnames
    ], dtype=np.int64)

    proto_out = run_tta_proto(
        proto_model, emb_te_f, sc_te_f,
        site_t = torch.tensor(test_site_ids, dtype=torch.long),
        hour_t = torch.tensor(test_hour_ids, dtype=torch.long),
        shifts = CFG["tta_shifts"],
    )
    proto_scores_flat = proto_out.reshape(-1, n_classes).astype(np.float32)

    # ── 4. prior + MLP probes ──
    prior_tables = build_prior_tables(sc, Y_SC)
    sc_te_adj = apply_prior(
        sc_te,
        sites = meta_te["site"].to_numpy(),
        hours = meta_te["hour_utc"].to_numpy(),
        tables= prior_tables,
        lambda_prior = CFG["lambda_prior"],
    )

    probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
        emb=emb_tr, scores_raw=sc_tr, Y=Y_FULL_aligned,
        min_pos=5, pca_dim=64, alpha_blend=0.4,
    )
    sc_te_adj = apply_mlp_probes_vectorized(
        emb_te, sc_te_adj, probe_models, emb_scaler, emb_pca, alpha_blend,
    )

    # ── 5. per-class first-pass weight (mapped vs unmapped) ──
    MAPPED_MASK = mapping["MAPPED_MASK"]
    w_map = CFG["ensemble_w_per_class_mapped"]
    w_unm = CFG["ensemble_w_per_class_unmapped"]
    ENSEMBLE_W = np.where(MAPPED_MASK, w_map, w_unm).astype(np.float32)
    first_pass_flat = (
        ENSEMBLE_W[None, :]        * proto_scores_flat
        + (1.0 - ENSEMBLE_W)[None, :] * sc_te_adj
    )
    print(f"[proto] per-class first-pass weights:  "
          f"mapped={ENSEMBLE_W[MAPPED_MASK].mean():.2f}  "
          f"unmapped={ENSEMBLE_W[~MAPPED_MASK].mean():.2f}")

    # ── 6. training-side first-pass for threshold + ResidualSSM ──
    n_tr_files  = len(sc_tr) // N_WINDOWS
    emb_tr_f    = emb_tr.reshape(n_tr_files, N_WINDOWS, -1)
    sc_tr_f     = sc_tr.reshape (n_tr_files, N_WINDOWS, -1)
    tr_fnames   = meta_tr.drop_duplicates("filename")["filename"].tolist()
    tr_site_ids = np.array([
        min(site2i_tr.get(meta_tr.loc[meta_tr["filename"] == fn, "site"].iloc[0], 0),
            n_sites_cap - 1)
        for fn in tr_fnames
    ], dtype=np.int64)
    tr_hour_ids = np.array([
        int(meta_tr.loc[meta_tr["filename"] == fn, "hour_utc"].iloc[0]) % 24
        for fn in tr_fnames
    ], dtype=np.int64)

    proto_tr_out = run_tta_proto(
        proto_model, emb_tr_f, sc_tr_f,
        site_t=torch.tensor(tr_site_ids, dtype=torch.long),
        hour_t=torch.tensor(tr_hour_ids, dtype=torch.long),
        shifts=CFG["tta_shifts"],
    )
    proto_tr_flat = proto_tr_out.reshape(-1, n_classes).astype(np.float32)

    sc_tr_prior = apply_prior(
        sc_tr,
        sites=meta_tr["site"].to_numpy(),
        hours=meta_tr["hour_utc"].to_numpy(),
        tables=prior_tables,
        lambda_prior=CFG["lambda_prior"],
    )
    sc_tr_mlp = apply_mlp_probes_vectorized(
        emb_tr, sc_tr_prior, probe_models, emb_scaler, emb_pca, alpha_blend,
    )
    first_pass_tr = (
        ENSEMBLE_W[None, :]        * proto_tr_flat
        + (1.0 - ENSEMBLE_W)[None, :] * sc_tr_mlp
    )

    # PER_CLASS_THRESHOLDS — Tweak 3: 更细的阈值 grid
    train_probs_for_calib = sigmoid(first_pass_tr)
    threshold_grid = (
        [round(t, 3) for t in np.arange(0.20, 0.45, 0.025)]
        + [round(t, 3) for t in np.arange(0.45, 0.75, 0.05)]
    )
    PER_CLASS_THRESHOLDS = calibrate_and_optimize_thresholds(
        oof_probs=train_probs_for_calib,
        Y_FULL=Y_FULL_aligned,
        threshold_grid=threshold_grid,
        n_windows=N_WINDOWS,
    )

    # ── 7. ResidualSSM training ──
    t0 = time.time()
    res_model, correction_weight = train_residual_ssm(
        emb_full=emb_tr, first_pass_flat=first_pass_tr, Y_full=Y_FULL_aligned,
        site_ids=tr_site_ids, hour_ids=tr_hour_ids, n_classes=n_classes,
        n_epochs=30, patience=8, lr=1e-3, correction_weight=0.30,
    )
    print(f"[proto] residual training {time.time() - t0:.1f}s")

    # ── 8. Tweak C: correction_weight grid search on training residuals ──
    res_model.eval()
    with torch.no_grad():
        tr_correction = res_model(
            torch.tensor(emb_tr_f,             dtype=torch.float32),
            torch.tensor(first_pass_tr.reshape(n_tr_files, N_WINDOWS, -1),
                         dtype=torch.float32),
            site_ids=torch.tensor(tr_site_ids, dtype=torch.long),
            hours   =torch.tensor(tr_hour_ids, dtype=torch.long),
        ).numpy().reshape(-1, n_classes).astype(np.float32)

    best_auc, best_w = -1.0, 0.30
    for w in CFG["correction_grid"]:
        trial_scores = first_pass_tr + w * tr_correction
        trial_probs  = sigmoid(trial_scores)
        auc = macro_auc(Y_FULL_aligned, trial_probs)
        print(f"  correction_weight={w:.2f}  OOF macro-AUC={auc:.5f}")
        if auc > best_auc:
            best_auc, best_w = auc, w
    correction_weight = best_w
    print(f"[proto] best correction_weight={correction_weight:.2f}  AUC={best_auc:.5f}")
    del tr_correction

    # ── 9. final test forward ──
    first_pass_te_f = first_pass_flat.reshape(n_test_files, N_WINDOWS, -1)
    res_model.eval()
    with torch.no_grad():
        test_correction = res_model(
            torch.tensor(emb_te_f,         dtype=torch.float32),
            torch.tensor(first_pass_te_f,  dtype=torch.float32),
            site_ids=torch.tensor(test_site_ids, dtype=torch.long),
            hours   =torch.tensor(test_hour_ids, dtype=torch.long),
        ).numpy()
    correction_flat = test_correction.reshape(-1, n_classes).astype(np.float32)

    final_scores = first_pass_flat + correction_weight * correction_flat
    final_scores = final_scores / mapping["temperatures"][None, :]
    probs = sigmoid(final_scores)
    probs = file_confidence_scale(probs, n_windows=N_WINDOWS,
                                   top_k=CFG["file_conf_top_k"],
                                   power=CFG["file_conf_power"])
    probs = rank_aware_scaling   (probs, n_windows=N_WINDOWS, power=CFG["rank_aware_power"])
    probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=CFG["delta_alpha"])
    probs = np.clip(probs, 0.0, 1.0)
    probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)

    # ── 10. write CSV ──
    out_path = Path(output_csv)
    sub = pd.DataFrame(probs.astype(np.float32), columns=primary_labels)
    sub.insert(0, "row_id", meta_te["row_id"].values)
    sub.to_csv(out_path, index=False)
    print(f"[proto] saved {out_path}  shape={sub.shape}")

    del emb_tr_f, sc_tr_f, proto_model, res_model
    gc.collect()
    return out_path
