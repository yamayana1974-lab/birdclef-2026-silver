# Third-Party Notices

> Language: **English** | [中文](THIRD_PARTY_NOTICES.zh.md)

**Read this before assuming the MIT `LICENSE` covers everything in this repo.**

The MIT `LICENSE` in this repository applies **only to the original integration
work and modifications authored by the repository owner** — i.e. the modular
refactor, the `BIRDCLEF_*` env-var plumbing, the weak-label / pseudo-label
extension, the ensemble/rank-blend wiring, and the documentation.

This project **integrates and adapts publicly shared BirdCLEF++ 2026 community
code** (Kaggle notebooks and community baselines). Those upstream components
remain the intellectual property of their respective authors and are **governed
by their own licenses and by Kaggle's rules**, not by this repository's MIT
license. Where an upstream source declares a license, that license controls the
corresponding code.

This repository does **not** redistribute competition data or pretrained model
weights. Obtain those from their original sources (see below and `docs/CREDITS.md`).

## Upstream components

| Component in this repo | Upstream author / source | Notes |
|---|---|---|
| `inference/scheme_model_5/` (Perch v2 + ProtoSSM + ResidualSSM core) | **yukiZ (hideyukizushi)** — *"Bird26.REPRODUCE.Perch+ProtoSSM+ResSSM.INF/TRAIN"* (Kaggle) | Core algorithm. Refactored into a package here. |
| `inference/scheme_model_5/` (rank power-optimization post-processing + SED blend) | **Derek** — *"birdclef 2026 exp019 eos4 rank power 06"* (Kaggle); references the **Karnakbayev** power-optimization chain | The "Model_5" community line (LB 0.949). |
| `training/distilled_sed/`, `training/distilled_sed_weak_labels/` | **Tucker Arrants** — *`bc2026-distilled-sed`* (Kaggle, Perch-distilled SED) | Backbone swapped to hgnetv2_b2; modularized. |
| `training/hgnet/`, `training/hgnet_weak_labels/`, `inference/hgnet_branch.py` | **ttahara (Tawara)** — *"BirdCLEF+2026: HGNetV2-B0 Baseline [Training]"* (Kaggle) | Modularized; multi-backbone ONNX ensemble added. |
| `training/pseudo_runner/` | Repository owner (original) | Pseudo-label generation around the above. |

## External assets (not redistributed here)

| Asset | Source |
|---|---|
| Perch v2 bird vocalization classifier (+ ONNX) | Google / Kaggle `google/bird-vocalization-classifier` |
| Distilled-SED public ONNX | `tuckerarrants/bc2026-distilled-sed-public` |
| Pre-converted train-audio WAV datasets | Kaggle `ttahara/birdclef2026-train-audio-wav-0{0..3}` |
| BirdCLEF++ 2026 competition data | Kaggle competition `birdclef-2026` (subject to competition rules) |

## If you are an upstream author

This code had been forked widely and exact provenance was reconstructed from
code comments and the public ensemble leaderboard. If you authored any part and
want different, added, or removed attribution, please open an issue.

See [`docs/CREDITS.md`](docs/CREDITS.md) for the detailed per-module breakdown of
what was changed.
