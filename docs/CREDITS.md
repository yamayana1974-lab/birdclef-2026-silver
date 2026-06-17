# Credits & Attribution

> Language: **English** | [中文](CREDITS.zh.md)

This solution is an **integration and extension of publicly shared BirdCLEF 2026
community code**. It does not claim novel algorithmic contributions. The value of
this repository is in the *engineering integration*, the *weak-label / pseudo-label
extension*, the *modular local-reproducible refactor*, and the *ensemble tuning* that
together reached a silver-medal private LB score.

Every module below lists its upstream source and exactly what was changed.

---

## Module attribution

| Module (this repo) | Upstream source | What I changed / added |
|---|---|---|
| `training/distilled_sed/` | Tucker Arrants' public notebook **`bc2026-distilled-sed`** (Perch-distilled SED) | Swapped backbone to **hgnetv2_b2** (via `BIRDCLEF_BACKBONE` env var; the silver submissions use b2), tuned training hyper-params, split the single notebook into a modular local-reproducible package, added 5-fold ONNX export + consistency check. Also tried b0 / b4. |
| `training/distilled_sed_weak_labels/` | Same Tucker distilled-SED base | **My extension**: distilled-SED (hgnetv2_b2) trained on weak / pseudo labels over all data. An experiment line; not the final submission SED. |
| `training/hgnet/` | **ttahara (Tawara)**'s public notebook **"BirdCLEF+2026: HGNetV2-B0 Baseline [Training]"** (community baseline, Notebooks Grandmaster; uses LSEModel + LSEHead + BCEWithLogitsLoss) | Modularized into a Python package, fixed several latent bugs from the original notebook (see module README), supported swapping backbone to **hgnetv2_b4** and other timm backbones, added OpenVINO export + CPU OOF validation. |
| `training/hgnet_weak_labels/` | Same ttahara HGNetV2-B0 baseline as above | **My extension**: weak-label / pseudo-label training flow on top of the baseline. |
| `training/pseudo_runner/` | — | **My code**: pseudo-label generation pipeline (build base, high-confidence filtering, class balancing, soft-label merging) feeding `hgnet_weak_labels`. |
| `inference/scheme_model_5/` | Kaggle notebook **"birdclef 2026 exp019 eos4 rank power 06"** by **Derek** (public LB 0.949; the "Model_5" entry of the community ensemble leaderboard). Its **Perch v2 + ProtoSSM + ResidualSSM** core originates from **yukiZ (hideyukizushi)**'s notebook **"Bird26.REPRODUCE.Perch+ProtoSSM+ResSSM.INF/TRAIN"**. Derek's notebook adds the **Karnakbayev** "Power Optimization" rank/post-processing chain (LB 0948) + the EOS-4 / Model_6 correction grid, and blends in Tucker Arrants' distilled-SED ONNX branch. | Split the single `cell_07` notebook into a modular, locally-reproducible Python package (config / data / mapping / perch_inference / probes / prior_fusion / protossm_pipeline / sed_pipeline / blend), made all hardcoded paths overridable via `BIRDCLEF_*` env vars, and added the HGNet-branch rank-blend (v20 / v22). Core algorithm and score path left unchanged. |
| `inference/hgnet_branch.py` | Built on the HGNet baseline above | My code: 4-fold / multi-backbone ONNX ensemble inference + rank-blend helper to fuse the HGNet branch into Model_5's submission. |

> The `scheme_model_5` Perch + ProtoSSM/MLP + site×hour prior + distilled-SED
> pipeline derives from Derek's public notebook *"birdclef 2026 exp019 eos4 rank
> power 06"* (the community "Model_5" line, LB 0.949). Its Perch + ProtoSSM +
> ResidualSSM core originates from **yukiZ (hideyukizushi)**'s
> *"Bird26.REPRODUCE.Perch+ProtoSSM+ResSSM.INF/TRAIN"*; Derek's notebook layers
> the Karnakbayev power-optimization post-processing on top. The code had been
> forked widely; this is the closest confirmable upstream. If you are the
> original author of any part and want different/added
> attribution, please open an issue.
>
> *(A second "Model_2" Perch/ProtoSSM variant (yukiZ reproduce) existed in the
> original notebook but was not used in either silver submission and has been
> removed from this repo.)*

---

## Pretrained models / external assets (not redistributed here)

These are required at train/inference time but are **not** included in this repo.
Download them from their original sources:

| Asset | Source | Used for |
|---|---|---|
| Perch v2 (bird vocalization classifier) | Google / Kaggle `google/bird-vocalization-classifier` | SED distillation teacher + Model_5 inference |
| Perch v2 ONNX (`perch_v2_no_dft.onnx`) | Kaggle community dataset | CPU inference of Perch |
| Distilled-SED public ONNX | `tuckerarrants/bc2026-distilled-sed-public` | Optional drop-in for the SED branch |
| ttahara train-audio WAV datasets (00–03) | Kaggle `ttahara/birdclef2026-train-audio-wav-0{0..3}` | Pre-converted training audio for HGNet |

---

## How to cite the original competition

BirdCLEF++ 2026, Kaggle / LifeCLEF. Please credit the competition organizers and
the original notebook authors listed above when building on this work.
