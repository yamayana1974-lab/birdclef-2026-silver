# Inference / Submission

> Language: **English** | [中文](README.zh.md)

Both silver submissions are built on the **Model_5 pipeline** (Perch v2 +
ProtoSSM/MLP + site×hour prior + distilled-SED) with an **HGNet branch**
rank-blended on top. v20 (0.946) and v22 (0.947) use **exactly the same code** —
they differ only in the HGNet branch configuration, switched entirely through
`BIRDCLEF_*` environment variables.

```
inference/
├── scheme_model_5/     # Model_5 pipeline (main) → submission csv
├── hgnet_branch.py     # HGNet branch: single- AND multi-backbone ONNX ensemble
├── SETTINGS.example.json   # template: fill paths to data + model weights
└── kaggle_notebooks/   # the actual Kaggle submission notebooks
```

> The earlier "Model_2" branch (a second Perch/ProtoSSM variant) was **not used**
> in either silver submission and has been removed. The HGNet branch hangs
> directly off Model_5.

## ⚠️ This does not run out-of-the-box

The pipeline loads several **external model weights / datasets** that are *not*
in this repo (they are git-ignored, see the root README). You must obtain them
and point the `BIRDCLEF_*` environment variables (or `SETTINGS.example.json`) at
them. Without these, nothing will run:

- Perch v2 (TF SavedModel) and/or Perch v2 ONNX
- Distilled-SED `sed_fold{0..4}.onnx` (trained via `training/distilled_sed/`, backbone hgnetv2_b2; or Tucker's public ONNX)
- HGNet branch weights: `best_model_fold{0..3}.pt` (single backbone, v20) or `best_model_fold{0..3}.onnx` per backbone dir (ensemble, v22), trained via `training/hgnet/` and `training/hgnet_weak_labels/`
- `birdclef-2026` competition data (audio + `sample_submission.csv` + `taxonomy.csv`)

See `docs/CREDITS.md` for sources.

## Settings

Copy `SETTINGS.example.json` to `SETTINGS.json`, fill in your local paths, then
export them as env vars (or just set the env vars directly). The pipeline reads
**environment variables**; the JSON is only a documented checklist of what to set.

| Env var | Meaning |
|---|---|
| `BIRDCLEF_BASE` | competition dir (`sample_submission.csv` / `taxonomy.csv` / `test_soundscapes/`) |
| `BIRDCLEF_INPUT_ROOT` | root used to rglob for ONNX / caches (Kaggle: `/kaggle/input`) |
| `BIRDCLEF_MODEL_DIR` | Perch v2 TF SavedModel dir (fallback if ONNX missing) |
| `BIRDCLEF_SED_DIR` | dir containing `sed_fold*.onnx` |
| `BIRDCLEF_MODE` | `submit` (default) or `train` |
| `BIRDCLEF_MODEL5_SUBM` | output csv path (default `subm_5.csv`) |
| `BIRDCLEF_M5_USE_HGNET` | `1` to enable the HGNet branch |
| `BIRDCLEF_HGNET_W` | HGNet branch total weight in the rank-blend |
| `BIRDCLEF_HGNET_BACKBONE` | (single backbone) timm name, e.g. `hgnetv2_b4.ssld_stage2_ft_in1k` |
| `BIRDCLEF_HGNET_DIR` | (single backbone) dir with `best_model_fold{0..3}.pt` |
| `BIRDCLEF_HGNET_ENSEMBLE_DIRS` | (ensemble) os.pathsep-separated backbone ONNX dirs |
| `BIRDCLEF_HGNET_ENSEMBLE_WEIGHTS` | (ensemble) internal weights, e.g. `1,1,3` |
| `BIRDCLEF_HGNET_BRANCH_PATH` | path to `hgnet_branch.py` (default: auto-resolve) |

> Paths default to my environment (Kaggle `/kaggle/input/...`). Override with
> these env vars rather than editing `config.py`.

## Reproducing each submission

### Model_5 alone (public LB 0.949)
```bash
cd inference
python -m scheme_model_5.main          # → subm_5.csv
```

### v20 — single-backbone HGNet branch (private LB 0.946)
Leave `BIRDCLEF_HGNET_ENSEMBLE_DIRS` **unset** → falls back to single backbone.
```bash
export BIRDCLEF_M5_USE_HGNET=1
export BIRDCLEF_HGNET_BACKBONE=hgnetv2_b4.ssld_stage2_ft_in1k
export BIRDCLEF_HGNET_DIR=/path/to/hgnetv2_b4_fold_pt   # best_model_fold{0..3}.pt
export BIRDCLEF_HGNET_W=0.15                            # branch weight (main 0.85)
python -m scheme_model_5.main          # → subm_5.csv (HGNet rank-blended in)
```

### v22 — multi-backbone HGNet ONNX ensemble (private LB 0.947)
Set `BIRDCLEF_HGNET_ENSEMBLE_DIRS` (os.pathsep-separated) → multi-backbone path.
```bash
export BIRDCLEF_M5_USE_HGNET=1
export BIRDCLEF_HGNET_ENSEMBLE_DIRS="/path/hgnetv2_b0_onnx:/path/hgnetv2_b1_onnx:/path/hgnetv2_b4_onnx_weak_labels"
export BIRDCLEF_HGNET_ENSEMBLE_WEIGHTS="1,1,3"          # b0:b1:b4 internal weights
export BIRDCLEF_HGNET_W=0.20                            # total (b0 4% + b1 4% + b4 12%)
python -m scheme_model_5.main          # → subm_5.csv (HGNet ensemble rank-blended in)
```
(On Windows use `;` as the separator and `set VAR=...`.)

## How the HGNet branch switch works

`scheme_model_5/hgnet_addon.py` decides the branch type from config:

- `HGNET_ENSEMBLE_DIRS` non-empty → **multi-backbone ONNX ensemble** (v22). Mel
  is computed once per file and fed to every backbone×fold ONNX session.
- `HGNET_ENSEMBLE_DIRS` empty → **single backbone** (v20), using
  `HGNET_CKPT_DIR` + `HGNET_BACKBONE`.

Both then `blend_with_hgnet(..., mode="rank")` into the Model_5 submission.

## Kaggle notebooks

`kaggle_notebooks/` contains only the two **silver-medal submission entrypoints**:

| Notebook | Submission | HGNet branch | `HGNET_W` |
|---|---|---|---|
| `notebook_v20_hgnet_b4.ipynb` | **0.946** | single backbone `hgnetv2_b4` (4-fold ONNX) | 0.15 |
| `notebook_v22_hgnet3_ensemble.ipynb` | **0.947** | ensemble `hgnetv2_b0 + b1 + b4(weak labels)`, internal `[1,1,3]` | 0.20 |

Each sets the env vars in its first cells, then imports `scheme_model_5.main`.

## Model configurations behind each submission

| Branch | v20 (0.946) | v22 (0.947) |
|---|---|---|
| Perch v2 + SED main (Model_5) | weight 0.85 | weight 0.80 |
| SED backbone | hgnetv2_b2 (Perch-distilled) | hgnetv2_b2 (Perch-distilled) |
| HGNet branch | single hgnetv2_b4, weight 0.15 | b0 + b1 + b4(weak), total 0.20 (b0 4% / b1 4% / b4 12%) |

The distilled-SED branch backbone is **hgnetv2_b2** in both submissions (I also
tried b0 / b4 / b2-with-weak-labels — see `docs/EXPERIMENTS.md`).
