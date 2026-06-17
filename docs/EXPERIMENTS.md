# Experiment Notes (honest record)

> Language: **English** | [中文](EXPERIMENTS.zh.md)

What I actually tried, what helped, and what didn't. The point of an open
competition write-up is reproducibility and honesty, not claiming invention.

## Headline numbers

| Config | Public LB | Private LB | Note |
|---|---|---|---|
| Model_5 alone | 0.949 | — | Perch v2 + ProtoSSM/MLP + site×hour prior + distilled-SED |
| v20 = M5 + single HGNet b4 | — | **0.946** | silver; HGNet b4 branch at weight 0.15 |
| v22 = M5 + HGNet 3-backbone ensemble | — | **0.947** | silver; b0+b1+b4(weak) at total weight 0.20 |

## Branch weights (final submissions)

| Branch | v20 (0.946) | v22 (0.947) |
|---|---|---|
| Perch v2 + distilled-SED main (Model_5) | 0.85 | 0.80 |
| HGNet branch | single `hgnetv2_b4`, 0.15 | `b0 + b1 + b4(weak)` `[1,1,3]`, 0.20 total (b0 4% / b1 4% / b4 12%) |

## SED branch backbone

The distilled-SED branch in both submissions uses **hgnetv2_b2** (Perch v2
distilled). During the competition I tried several SED backbones:

- `hgnetv2_b0`
- `hgnetv2_b2`  ← used in the silver submissions
- `hgnetv2_b4`
- `hgnetv2_b2` with **weak / pseudo labels** (training code:
  `training/distilled_sed_weak_labels/`)

The SED backbone is swapped via the `BIRDCLEF_BACKBONE` env var in
`training/distilled_sed/config.py`.

## HGNet branch models

- v20: a single `hgnetv2_b4` (4-fold) — trained via `training/hgnet/`.
- v22: ensemble of `hgnetv2_b4` (**weak/pseudo labels**, `training/hgnet_weak_labels/`)
  + `hgnetv2_b0` (no weak labels) + `hgnetv2_b1` (no weak labels), all 4-fold ONNX,
  internal weights `[1,1,3]` (b4 dominant).

## Key observations

- **The HGNet branch lowers public LB but was kept for diversity.** The pure
  Model_5 (0.949 public) is stronger on the public split than v20/v22. The HGNet
  branch (a mel-spectrogram image classifier, very different from the Perch
  embedding line) was rank-blended in at a small weight to add ensemble
  diversity and reduce private-LB shake-up risk. On private LB the two blended
  submissions landed at 0.946 / 0.947 — solid silver.

- **Multi-backbone HGNet > single backbone.** v22 (b0 + b1 + b4-weak, 3-backbone
  ONNX ensemble with mel computed once and fed to all sessions) beat v20
  (single b4) by ~0.001 private. The internal weighting `[1,1,3]` leaned heavily
  on the weak-label b4 model. (A 4-backbone variant adding repvit_m1_1 was also
  tried but the 3-backbone version was the final 0.947 submission.)

- **SED backbone swap.** The distilled-SED was retrained with **hgnetv2_b2**
  instead of the upstream EfficientNet-B0. Training keeps Perch v2 as the
  distillation teacher (MSE on 1536-d embeddings) with stop-gradient between the
  distillation head and the SED classification head.

- **Weak / pseudo labels.** A pseudo-label pipeline (`training/pseudo_runner/`)
  generated high-confidence soft labels to train an additional HGNet (b4) model
  used inside the v22 ensemble.

## Inference engineering (matters for a code competition)

- Everything runs on **CPU via ONNX / OpenVINO** to fit the notebook runtime
  limit.
- HGNet branch: 12×5s windows, 4-fold (or multi-backbone) ONNX, sigmoid-average
  in prob space, Gaussian σ=0.65 time smoothing.
- Final blend uses **rank-percentile** averaging (scale-robust) rather than raw
  probability averaging.

## Upstream attribution

Full per-module attribution is in [`docs/CREDITS.md`](CREDITS.md). In short:

- `scheme_model_5` (Perch v2 + ProtoSSM/MLP + prior + distilled-SED) derives from
  **Derek**'s *"birdclef 2026 exp019 eos4 rank power 06"* notebook, whose
  Perch + ProtoSSM + ResidualSSM core originates from **yukiZ (hideyukizushi)**.
- distilled-SED line: **Tucker Arrants**' `bc2026-distilled-sed`.
- HGNet branch: **ttahara (Tawara)**'s HGNetV2-B0 baseline.
