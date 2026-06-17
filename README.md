# BirdCLEF++ 2026 — Silver Medal Solution (75/4091, top 1.83%, private LB 0.946 / 0.947)

> Language: **English** | [中文](README.zh.md)

Reproducible code for my two silver-medal submissions to the
[BirdCLEF++ 2026](https://www.kaggle.com/competitions/birdclef-2026) research
competition (multi-label species recognition from soundscape recordings,
macro-averaged ROC-AUC).

**Final standing: 75 / 4091 teams (top 1.83%, 🥈 silver medal).**

> **Honest scope.** This is **not** a novel-algorithm solution. It is an
> *integration + extension* of publicly shared community code (Tucker Arrants'
> distilled-SED notebook, ttahara's HGNetV2 baseline, the Perch line, etc.).
> The contribution here is the engineering: a modular, locally-reproducible
> pipeline, a weak-label / pseudo-label training extension, and the ensemble
> tuning that held up on the private leaderboard. Full attribution is in
> [`docs/CREDITS.md`](docs/CREDITS.md).

## Results

| Submission | Entrypoint notebook | Private LB | What's different |
|---|---|---|---|
| v20 | `kaggle_notebooks/notebook_v20_hgnet_b4.ipynb` | **0.946** (silver) | Single-backbone HGNet branch (hgnetv2_b4, 4-fold ONNX) rank-blended into Model_5 at weight 0.15 |
| v22 | `kaggle_notebooks/notebook_v22_hgnet3_ensemble.ipynb` | **0.947** (silver) | HGNet 3-backbone ONNX ensemble (b0 + b1 + b4-weak, `[1,1,3]`) rank-blended into Model_5 at total weight 0.20 |

> v20 and v22 share the **same inference code**; they differ only in HGNet branch
> configuration, selected via `BIRDCLEF_*` environment variables. See
> [`inference/README.md`](inference/README.md) for the exact commands. The
> distilled-SED branch backbone is **hgnetv2_b2** in both.

The Model_5 main pipeline alone scored **0.949 public LB**. The HGNet branch was
added for model diversity / private-LB robustness (it slightly lowers public LB
but increases ensemble diversity). The gold-medal 0.959 solution is *not* mine
and is not included here.

## Architecture overview

```
              60s soundscape (12 x 5s windows)
                          │
        ┌──────────────────┴──────────────────┐
        │                                      │
   Model_5 (main)                        HGNet branch
   Perch v2 + ProtoSSM/MLP               mel-spectrogram
   + site×hour prior fusion              image classifier
   + Distilled-SED (hgnetv2_b2)          (LSEModel, ONNX)
        │                                      │
        │  weight 0.85 (v20) / 0.80 (v22)      │  0.15 (v20) / 0.20 (v22)
        └──────────────► rank-blend ◄──────────┘
                              │
                       submission.csv
```

## Repository layout

```
BirdCLEF_2026_release/
├── README.md
├── LICENSE                     # MIT (integration work); upstream licenses apply per-module
├── .gitignore
├── docs/
│   ├── CREDITS.md              # ★ upstream attribution for every module
│   └── EXPERIMENTS.md          # honest experiment record (what helped / didn't)
├── training/
│   ├── distilled_sed/          # Perch-distilled SED, backbone=hgnetv2_b2 → sed_fold{0..4}.onnx
│   ├── distilled_sed_weak_labels/  # SED (hgnetv2_b2) on weak/pseudo labels (experiment)
│   ├── hgnet/                  # HGNetV2 branch training (standard labels)
│   ├── hgnet_weak_labels/      # HGNet branch training (weak / pseudo labels)
│   └── pseudo_runner/          # pseudo-label generation pipeline (feeds the weak-label models)
└── inference/                 # ★ Model_5 pipeline + HGNet branch (v20 / v22 via env vars)
    ├── scheme_model_5/         # Model_5 main pipeline → submission csv (LB 0.949)
    ├── hgnet_branch.py         # HGNet branch (single + multi-backbone ONNX)
    ├── SETTINGS.example.json   # template for data + weight paths
    └── kaggle_notebooks/       # Kaggle submission notebooks (v20 + v22)
```

Each subfolder keeps its own `README.md` with detailed, step-by-step instructions.

## Model weights are NOT in this repo

All `.pt / .onnx / .bin / .xml / .zip` and competition data are git-ignored.
They are large and should be hosted on GitHub Releases or as Kaggle datasets.
The required external assets (Perch v2, ttahara WAV datasets, public SED ONNX)
and their original sources are listed in [`docs/CREDITS.md`](docs/CREDITS.md).

## Quick start

### 1. Train the SED model (hgnetv2_b2, Perch-distilled)
```bash
cd training/distilled_sed
# see distilled_sed/README.md and REPRODUCE_LOCAL.md for the full flow
python main.py --mode train --backbone hgnetv2_b2.ssld_stage2_ft_in1k --folds 0,1,2,3,4
# → output/<backbone>_<timestamp>/sed_fold{0..4}.onnx
```

### 2. Train the HGNet branch
```bash
cd training        # hgnet / hgnet_weak_labels are packages under training/

# standard labels
python -m hgnet.train          # → best_model_fold{0..3}.pt → export to ONNX

# weak / pseudo labels (after generating pseudo-labels via pseudo_runner)
python -m hgnet_weak_labels.train
```

### 3. Run inference / build a submission
```bash
cd inference
# set BIRDCLEF_* env vars to point at data + ONNX weights (see inference/README.md)
python -m scheme_model_5.main   # Model_5 + HGNet branch → submission csv
```
v20 (0.946) and v22 (0.947) are the same code with different HGNet env vars —
see [`inference/README.md`](inference/README.md) and `inference/SETTINGS.example.json`
for the exact variable sets.

> **Paths default to `./data` and `./models`** (relative to the repo) and can be
> overridden with the documented `BIRDCLEF_*` environment variables (e.g.
> `BIRDCLEF_BASE`, `BIRDCLEF_TRAIN_ROOT`, `BIRDCLEF_SED_ROOT`) rather than editing
> the code. On Kaggle the pipeline auto-detects `/kaggle/input/...`. Each module
> README lists the relevant vars.

## Environment

- Python 3.10–3.11 (HGNet inference) / 3.10+ (SED training)
- PyTorch + torchaudio + torchvision (install the CUDA build for your machine)
- `timm`, `onnxruntime`, `openvino`, `librosa`, `soundfile`, `scikit-learn`, `pandas`, `numpy`
- See each module's `requirements.txt`.

## License

MIT for the integration and original modifications in this repo. Upstream
community components remain under their authors' rights — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`docs/CREDITS.md`](docs/CREDITS.md). When in doubt, credit the original authors.
