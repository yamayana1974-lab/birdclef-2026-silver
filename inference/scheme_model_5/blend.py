# -*- coding: utf-8 -*-
"""blend.py — xSED rank blend + 5 道 gate + dry-run alignment → ``subm_5.csv``.

对应 cell_07 第 1419-1718 行:

* xSED rank blend (proto vs SED, 权重来自 ``solut.Models[0].xSED``)
* Gate 1 — fake-only noise suppression
* Gate 2 — temporal continuity (35s t-distribution kernel)
* Gate 3 — SED spike preservation
* Gate 4 — sonotype mirroring (visually identical species groups)
* Gate 5 — adaptive rare-class thresholding (Amphibia / Mammalia / Reptilia)
* dry-run sample-submission alignment
* ``write_final_submission`` — sanity check + 写最终 CSV
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import INTERMEDIATE_SUBM, OUTPUT_SUBM, SOLUT


EPS_BLEND = 1e-5


# 共 4 组镜像 (cell_07 第 1492-1497 行)
MIRROR_PAIRS: Tuple[Tuple[str, ...], ...] = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
)


def _xsed_weights() -> Tuple[float, float]:
    model = SOLUT["Models"][0]
    proto_w, sed_w = [float(v) for v in model.get("xSED", [0.60, 0.40])]
    s = proto_w + sed_w
    if s <= 0:
        raise ValueError(f"Invalid xSED weights for {model['Model']}: {model.get('xSED')}")
    return proto_w / s, sed_w / s


def blend_proto_sed(
    proto_csv: str,
    sed_csv: str,
    base_dir: Path,
    output_csv: str = INTERMEDIATE_SUBM,
) -> Path:
    df_proto = pd.read_csv(proto_csv)
    df_sed   = pd.read_csv(sed_csv)
    cols     = [c for c in df_proto.columns if c != "row_id"]
    df_sed   = df_sed.set_index("row_id").loc[df_proto["row_id"]].reset_index()

    p_proto = np.clip(df_proto[cols].to_numpy(np.float32), EPS_BLEND, 1.0 - EPS_BLEND)
    p_sed   = np.clip(df_sed  [cols].to_numpy(np.float32), EPS_BLEND, 1.0 - EPS_BLEND)
    rank_proto = pd.DataFrame(p_proto).rank(axis=0, pct=True).to_numpy(np.float32)
    rank_sed   = pd.DataFrame(p_sed  ).rank(axis=0, pct=True).to_numpy(np.float32)

    # xSED rank blend (Karnakbayev_PowerOptimization_LB0948)
    proto_w, sed_w = _xsed_weights()
    print(f"[blend] xSED rank blend  proto={proto_w:.4f}  sed={sed_w:.4f}")
    pred = rank_proto * proto_w + rank_sed * sed_w

    row_ids  = df_proto["row_id"].astype(str).to_numpy()
    file_ids = np.array(["_".join(r.split("_")[:-1]) for r in row_ids])

    # Gate 1: fake-only noise suppression
    fake_only = (p_proto > 0.50) & (p_sed < 0.05)
    pred = np.where(fake_only, (1.0 - 0.08) * pred + 0.08 * rank_proto, pred)

    # Gate 2: temporal continuity (35s context, t-distribution kernel)
    offs         = np.arange(-3, 4, dtype=np.float32)
    proto_kernel = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
    proto_kernel = (proto_kernel / proto_kernel.sum()).astype(np.float32)

    pa_ctx = p_proto.copy()
    for fid in pd.unique(file_ids):
        m = file_ids == fid
        x = p_proto[m]
        if len(x) > 1:
            xp = np.pad(x, ((3, 3), (0, 0)), mode="edge")
            pa_ctx[m] = sum(proto_kernel[i] * xp[i:i + len(x)] for i in range(7))

    xctx = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
    proto_cont = (xctx > 0.88) & (rank_proto > 0.75) & (p_sed < 0.12) & (~fake_only)
    pred = np.where(
        proto_cont,
        (1.0 - 0.15) * pred + 0.15 * np.maximum(rank_proto, xctx),
        pred,
    )

    # Gate 3: SED spike preservation
    sed_only = (rank_sed > 0.95) & (rank_proto < 0.80) & (~fake_only) & (~proto_cont)
    pred = np.where(sed_only, (1.0 - 0.12) * pred + 0.12 * rank_sed, pred)
    sub = df_proto.copy()
    sub[cols] = pred.astype(np.float32)

    # Gate 4: sonotype mirroring
    col_to_idx   = {l: i for i, l in enumerate(cols)}
    mirror_count = 0
    for group in MIRROR_PAIRS:
        valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
        if len(valid_idx) >= 2:
            group_max = sub[cols].iloc[:, valid_idx].max(axis=1).to_numpy(np.float32)
            for idx in valid_idx:
                sub.iloc[:, idx + 1] = group_max
            mirror_count += len(valid_idx)
    print(f"[blend] sonotype mirroring applied to {mirror_count} columns")

    # Gate 5: adaptive rare-class thresholding
    try:
        tax_df = pd.read_csv(base_dir / "taxonomy.csv").set_index("primary_label")
        rare_classes = {"Amphibia", "Mammalia", "Reptilia"}
        rare_count   = 0
        for ci, species in enumerate(cols):
            if species in tax_df.index and tax_df.loc[species, "class_name"] in rare_classes:
                col_idx = ci + 1
                vals    = sub.iloc[:, col_idx].to_numpy(np.float32)
                thr     = vals.mean() + 0.05
                sub.iloc[:, col_idx] = np.where(vals < thr, vals * 0.9, vals)
                rare_count += 1
        print(f"[blend] adaptive thresholding applied to {rare_count} rare species")
    except Exception as e:
        print(f"[blend] adaptive thresholding skipped: {e}")

    # Dry-run alignment (覆盖整张表为 sample_submission 的均值)
    test_paths = list((base_dir / "test_soundscapes").glob("*.ogg"))
    is_dry_run = len(test_paths) == 0
    if is_dry_run:
        print("[blend] dry-run detected — aligning rows with sample_submission.csv")
        sample_public = pd.read_csv(base_dir / "sample_submission.csv")
        template      = sub[cols].mean(axis=0).astype(np.float32)
        sub = sample_public.copy()
        for label in cols:
            sub[label] = template[label]

    out = Path(output_csv)
    sub.to_csv(out, index=False)
    print(f"[blend] saved {out}  shape={sub.shape}")
    return out


# =============================================================================
# 最终输出: subm_5.csv (cell_07 第 1593-1718 行)
# =============================================================================
def _read_submission_checked(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "row_id" in df.columns, f"row_id missing in {path}"
    assert not any(str(c).startswith("Unnamed") for c in df.columns), (
        f"unexpected unnamed column in {path}: {df.columns.tolist()[:5]}"
    )
    assert df["row_id"].is_unique, f"duplicate row_id in {path}"
    prob_cols = [c for c in df.columns if c != "row_id"]
    assert prob_cols, f"no probability columns in {path}"
    values = df[prob_cols].to_numpy(dtype=np.float32)
    assert np.isfinite(values).all(), f"NaN/inf in {path}"
    assert values.min() >= 0.0 and values.max() <= 1.0, (
        f"probabilities outside [0, 1] in {path}"
    )
    out = df.set_index("row_id")
    out.index = out.index.astype(str)
    out.index.name = "row_id"
    return out


def direct_add_safe(files: List[str], weights: List[float],
                    ensemble_models: List[str], lbs: List[str]) -> pd.DataFrame:
    print(f"[blend] direct_add_safe ensemble={ensemble_models} lb={lbs} weights={weights}")
    assert len(files) == len(weights)
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("ensemble weights must sum to positive")
    if not np.isclose(s, 1.0, atol=1e-6):
        print(f"[blend] normalizing weights from sum={s:.6f}")
    norm_w = [float(w) / s for w in weights]

    dfs = [_read_submission_checked(p) for p in files]
    base_idx, base_cols = dfs[0].index, dfs[0].columns
    for path, df in zip(files, dfs):
        assert df.columns.equals(base_cols), f"Column mismatch in {path}"
        assert len(base_idx.difference(df.index)) == 0, f"row_id mismatch in {path}"
        assert len(df.index.difference(base_idx)) == 0, f"row_id mismatch in {path}"

    out = sum(w * df.loc[base_idx, base_cols] for w, df in zip(norm_w, dfs))
    out.index.name = "row_id"
    values = out.to_numpy(dtype=np.float32)
    assert np.isfinite(values).all(), "NaN/inf in final blend"
    assert values.min() >= 0.0 and values.max() <= 1.0
    return out


def _find_sample_submission_path(base_dir: Path) -> Path:
    p = base_dir / "sample_submission.csv"
    if p.exists():
        return p
    for cand in (
        Path("/kaggle/input/competitions/birdclef-2026/sample_submission.csv"),
        Path("/kaggle/input/birdclef-2026/sample_submission.csv"),
    ):
        if cand.exists():
            return cand
    root = Path("/kaggle/input")
    if root.exists():
        for cand in sorted(root.rglob("sample_submission.csv")):
            if (cand.parent / "taxonomy.csv").exists():
                return cand
    return None


def _as_explicit_submission_table(pred) -> pd.DataFrame:
    if isinstance(pred, pd.DataFrame) and "row_id" in pred.columns:
        df = pred.copy()
    elif isinstance(pred, pd.DataFrame) and pred.index.name == "row_id":
        df = pred.reset_index()
    else:
        raise AssertionError("final prediction must be DF with row_id column or index")
    assert "row_id" in df.columns
    df["row_id"] = df["row_id"].astype(str)
    assert df["row_id"].is_unique
    return df


def _align_to_sample_submission_if_possible(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    sample_path = _find_sample_submission_path(base_dir)
    if sample_path is None:
        return df
    sample = pd.read_csv(sample_path)
    sample["row_id"] = sample["row_id"].astype(str)
    sample_cols = sample.columns.tolist()
    missing_cols = [c for c in sample_cols if c not in df.columns]
    assert not missing_cols, f"final submission missing sample columns: {missing_cols[:5]}"
    final_ids  = set(df["row_id"])
    sample_ids = set(sample["row_id"])
    if final_ids == sample_ids:
        aligned = (df.set_index("row_id")
                   .loc[sample["row_id"], sample_cols[1:]]
                   .reset_index())
        aligned.columns = sample_cols
        return aligned
    missing = sorted(sample_ids - final_ids)[:5]
    extra   = sorted(final_ids  - sample_ids)[:5]
    raise AssertionError(
        f"final row_id set differs from sample_submission: "
        f"missing={len(sample_ids - final_ids)} first={missing}, "
        f"extra={len(final_ids  - sample_ids)} first={extra}"
    )


def write_final_submission(pred, base_dir: Path, path: str = OUTPUT_SUBM) -> pd.DataFrame:
    df = _as_explicit_submission_table(pred)
    df = _align_to_sample_submission_if_possible(df, base_dir)
    prob_cols = [c for c in df.columns if c != "row_id"]
    values    = df[prob_cols].to_numpy(dtype=np.float32)
    assert np.isfinite(values).all() and values.min() >= 0.0 and values.max() <= 1.0
    df.to_csv(path, index=False)
    print(f"[blend] wrote {path}: rows={len(df)} cols={df.shape[1]} "
          f"min={values.min():.6f} max={values.max():.6f}")
    return df
