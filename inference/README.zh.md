# 推理 / 提交

> 语言: [English](README.md) | **中文**

两个银牌提交都构建在 **Model_5 流水线** (Perch v2 + ProtoSSM/MLP + site×hour
prior + distilled-SED) 之上, 再 rank-blend 一个 **HGNet 分支**。v20 (0.946) 和
v22 (0.947) 用**完全相同的代码** —— 只在 HGNet 分支配置上不同, 全部通过
`BIRDCLEF_*` 环境变量切换。

```
inference/
├── scheme_model_5/     # Model_5 流水线 (主) → submission csv
├── hgnet_branch.py     # HGNet 分支: 单 + 多 backbone ONNX 集成
├── SETTINGS.example.json   # 模板: 填数据 + 模型权重路径
└── kaggle_notebooks/   # 实际的 Kaggle 提交 notebook
```

> 早期的 "Model_2" 分支 (第二个 Perch/ProtoSSM 变体) 两个银牌提交都**没用到**,
> 已移除。HGNet 分支直接挂在 Model_5 上。

## ⚠️ 开箱即用跑不起来

流水线会加载若干**外部模型权重 / 数据集**, 它们*不*在本仓库里 (被 git-ignore,
见根 README)。你必须自己获取, 并把 `BIRDCLEF_*` 环境变量 (或 `SETTINGS.example.json`)
指向它们。没有这些, 什么都跑不了:

- Perch v2 (TF SavedModel) 和/或 Perch v2 ONNX
- Distilled-SED `sed_fold{0..4}.onnx` (用 `training/distilled_sed/` 训, backbone hgnetv2_b2; 或 Tucker 的公开 ONNX)
- HGNet 分支权重: `best_model_fold{0..3}.pt` (单 backbone, v20) 或每个 backbone 目录下的 `best_model_fold{0..3}.onnx` (集成, v22), 用 `training/hgnet/` 和 `training/hgnet_weak_labels/` 训练
- `birdclef-2026` 竞赛数据 (音频 + `sample_submission.csv` + `taxonomy.csv`)

来源见 `docs/CREDITS.md`。

## 配置

把 `SETTINGS.example.json` 复制成 `SETTINGS.json`, 填好本地路径, 然后把它们
export 成环境变量 (或直接设环境变量)。流水线读的是**环境变量**; JSON 只是一份
"需要设哪些变量" 的文档化清单。

| 环境变量 | 含义 |
|---|---|
| `BIRDCLEF_BASE` | 竞赛目录 (`sample_submission.csv` / `taxonomy.csv` / `test_soundscapes/`) |
| `BIRDCLEF_INPUT_ROOT` | 用于 rglob 找 ONNX / cache 的根 (Kaggle: `/kaggle/input`) |
| `BIRDCLEF_MODEL_DIR` | Perch v2 TF SavedModel 目录 (ONNX 缺失时的 fallback) |
| `BIRDCLEF_SED_DIR` | 含 `sed_fold*.onnx` 的目录 |
| `BIRDCLEF_MODE` | `submit` (默认) 或 `train` |
| `BIRDCLEF_MODEL5_SUBM` | 输出 csv 路径 (默认 `subm_5.csv`) |
| `BIRDCLEF_M5_USE_HGNET` | `1` 启用 HGNet 分支 |
| `BIRDCLEF_HGNET_W` | HGNet 分支在 rank-blend 中的总权重 |
| `BIRDCLEF_HGNET_BACKBONE` | (单 backbone) timm 名, 如 `hgnetv2_b4.ssld_stage2_ft_in1k` |
| `BIRDCLEF_HGNET_DIR` | (单 backbone) 含 `best_model_fold{0..3}.pt` 的目录 |
| `BIRDCLEF_HGNET_ENSEMBLE_DIRS` | (集成) os.pathsep 分隔的各 backbone ONNX 目录 |
| `BIRDCLEF_HGNET_ENSEMBLE_WEIGHTS` | (集成) 内部权重, 如 `1,1,3` |
| `BIRDCLEF_HGNET_BRANCH_PATH` | `hgnet_branch.py` 路径 (默认: 自动解析) |

> 路径默认指向我的环境 (Kaggle `/kaggle/input/...`)。用这些环境变量覆盖,
> 而不是改 `config.py`。

## 复现每个提交

### 仅 Model_5 (public LB 0.949)
```bash
cd inference
python -m scheme_model_5.main          # → subm_5.csv
```

### v20 — 单 backbone HGNet 分支 (private LB 0.946)
**不设** `BIRDCLEF_HGNET_ENSEMBLE_DIRS` → 回退单 backbone。
```bash
export BIRDCLEF_M5_USE_HGNET=1
export BIRDCLEF_HGNET_BACKBONE=hgnetv2_b4.ssld_stage2_ft_in1k
export BIRDCLEF_HGNET_DIR=/path/to/hgnetv2_b4_fold_pt   # best_model_fold{0..3}.pt
export BIRDCLEF_HGNET_W=0.15                            # 分支权重 (主 0.85)
python -m scheme_model_5.main          # → subm_5.csv (HGNet rank-blend 进去)
```

### v22 — 多 backbone HGNet ONNX 集成 (private LB 0.947)
设 `BIRDCLEF_HGNET_ENSEMBLE_DIRS` (os.pathsep 分隔) → 走多 backbone 路径。
```bash
export BIRDCLEF_M5_USE_HGNET=1
export BIRDCLEF_HGNET_ENSEMBLE_DIRS="/path/hgnetv2_b0_onnx:/path/hgnetv2_b1_onnx:/path/hgnetv2_b4_onnx_weak_labels"
export BIRDCLEF_HGNET_ENSEMBLE_WEIGHTS="1,1,3"          # b0:b1:b4 内部权重
export BIRDCLEF_HGNET_W=0.20                            # 总 (b0 4% + b1 4% + b4 12%)
python -m scheme_model_5.main          # → subm_5.csv (HGNet 集成 rank-blend 进去)
```
(Windows 上用 `;` 作分隔符, 用 `set VAR=...`。)

## HGNet 分支开关原理

`scheme_model_5/hgnet_addon.py` 根据 config 决定分支类型:

- `HGNET_ENSEMBLE_DIRS` 非空 → **多 backbone ONNX 集成** (v22)。mel 每个文件只算
  一次, 喂给每个 backbone×fold 的 ONNX session。
- `HGNET_ENSEMBLE_DIRS` 为空 → **单 backbone** (v20), 用 `HGNET_CKPT_DIR` +
  `HGNET_BACKBONE`。

两者最后都 `blend_with_hgnet(..., mode="rank")` 融进 Model_5 的提交。

## Kaggle notebook

`kaggle_notebooks/` 只含两个**银牌提交入口**:

| Notebook | 提交 | HGNet 分支 | `HGNET_W` |
|---|---|---|---|
| `notebook_v20_hgnet_b4.ipynb` | **0.946** | 单 backbone `hgnetv2_b4` (4 折 ONNX) | 0.15 |
| `notebook_v22_hgnet3_ensemble.ipynb` | **0.947** | 集成 `hgnetv2_b0 + b1 + b4(weak labels)`, 内部 `[1,1,3]` | 0.20 |

每个 notebook 在头几个 cell 设好环境变量, 然后 import `scheme_model_5.main`。

## 每个提交背后的模型配置

| 分支 | v20 (0.946) | v22 (0.947) |
|---|---|---|
| Perch v2 + SED 主 (Model_5) | 权重 0.85 | 权重 0.80 |
| SED backbone | hgnetv2_b2 (Perch 蒸馏) | hgnetv2_b2 (Perch 蒸馏) |
| HGNet 分支 | 单 hgnetv2_b4, 权重 0.15 | b0 + b1 + b4(weak), 共 0.20 (b0 4% / b1 4% / b4 12%) |

两个提交的 distilled-SED 分支 backbone 都是 **hgnetv2_b2** (我也试过 b0 / b4 /
b2-配弱标签 —— 见 `docs/EXPERIMENTS.md`)。
