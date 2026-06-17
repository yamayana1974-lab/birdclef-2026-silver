# BirdCLEF++ 2026 — 银牌方案 (75/4091, 前 1.83%, private LB 0.946 / 0.947)

> 语言: [English](README.md) | **中文**

[BirdCLEF++ 2026](https://www.kaggle.com/competitions/birdclef-2026) 研究竞赛
(从声景录音中做多标签物种识别, 评测指标为 macro-averaged ROC-AUC) 两个银牌
提交的可复现代码。

**最终排名: 75 / 4091 队 (前 1.83%, 🥈 银牌)。**

> **诚实定位**: 这**不是**一个全新算法的方案, 而是对社区公开代码的
> *集成 + 扩展* (Tucker Arrants 的 distilled-SED notebook、ttahara 的 HGNetV2
> baseline、Perch 系列等)。本仓库的贡献在于工程层面: 模块化、可本地复现的
> 流水线, 弱标签 / 伪标签训练扩展, 以及在 private 榜上稳住的集成调参。完整
> 归因见 [`docs/CREDITS.md`](docs/CREDITS.md)。

## 成绩

| 提交 | 入口 notebook | Private LB | 区别 |
|---|---|---|---|
| v20 | `kaggle_notebooks/notebook_v20_hgnet_b4.ipynb` | **0.946** (银牌) | 单 backbone 的 HGNet 分支 (hgnetv2_b4, 4 折 ONNX) 以权重 0.15 rank-blend 进 Model_5 |
| v22 | `kaggle_notebooks/notebook_v22_hgnet3_ensemble.ipynb` | **0.947** (银牌) | HGNet 3-backbone ONNX 集成 (b0 + b1 + b4-weak, `[1,1,3]`) 以总权重 0.20 rank-blend 进 Model_5 |

> v20 和 v22 **共用同一份推理代码**, 只在 HGNet 分支配置上不同, 通过
> `BIRDCLEF_*` 环境变量切换。具体命令见 [`inference/README.md`](inference/README.md)。
> 两者的 distilled-SED 分支 backbone 都是 **hgnetv2_b2**。

仅 Model_5 主流水线在 public LB 上是 **0.949**。加 HGNet 分支是为了模型多样性 /
private 榜稳健性 (它略微拉低 public LB, 但增加集成多样性)。金牌 0.959 方案**不是**
我的, 不包含在此。

## 架构概览

```
              60s 声景 (12 x 5s 窗口)
                          │
        ┌──────────────────┴──────────────────┐
        │                                      │
   Model_5 (主)                          HGNet 分支
   Perch v2 + ProtoSSM/MLP               梅尔谱图
   + site×hour prior 融合                图像分类器
   + Distilled-SED (hgnetv2_b2)          (LSEModel, ONNX)
        │                                      │
        │  权重 0.85 (v20) / 0.80 (v22)        │  0.15 (v20) / 0.20 (v22)
        └──────────────► rank-blend ◄──────────┘
                              │
                       submission.csv
```

## 仓库结构

```
BirdCLEF_2026_release/
├── README.md                   # 英文说明 (本文件为中文版 README.zh.md)
├── LICENSE                     # MIT (集成工作); 上游各模块适用其各自 license
├── .gitignore
├── docs/
│   ├── CREDITS.md              # ★ 每个模块的上游归因
│   └── EXPERIMENTS.md          # 诚实的实验记录 (什么有效 / 无效)
├── training/
│   ├── distilled_sed/          # Perch 蒸馏 SED, backbone=hgnetv2_b2 → sed_fold{0..4}.onnx
│   ├── distilled_sed_weak_labels/  # 弱/伪标签上的 SED (hgnetv2_b2) (实验)
│   ├── hgnet/                  # HGNetV2 分支训练 (标准标签)
│   ├── hgnet_weak_labels/      # HGNet 分支训练 (弱 / 伪标签)
│   └── pseudo_runner/          # 伪标签生成流水线 (喂给弱标签模型)
└── inference/                 # ★ Model_5 流水线 + HGNet 分支 (v20 / v22 通过环境变量)
    ├── scheme_model_5/         # Model_5 主流水线 → submission csv (LB 0.949)
    ├── hgnet_branch.py         # HGNet 分支 (单 + 多 backbone ONNX)
    ├── SETTINGS.example.json   # 数据 + 权重路径模板
    └── kaggle_notebooks/       # Kaggle 提交 notebook (v20 + v22)
```

每个子目录都有自己的 `README.md`, 含详细的分步说明。

## 模型权重不在本仓库

所有 `.pt / .onnx / .bin / .xml / .zip` 和竞赛数据都被 git-ignore。它们体积大,
应托管在 GitHub Releases 或作为 Kaggle 数据集。所需的外部资源 (Perch v2、ttahara
的 WAV 数据集、公开 SED ONNX) 及其原始来源列在 [`docs/CREDITS.md`](docs/CREDITS.md)。

## 快速上手

### 1. 训练 SED 模型 (hgnetv2_b2, Perch 蒸馏)
```bash
cd training/distilled_sed
# 完整流程见 distilled_sed/README.md 和 REPRODUCE_LOCAL.md
python main.py --mode train --backbone hgnetv2_b2.ssld_stage2_ft_in1k --folds 0,1,2,3,4
# → output/<backbone>_<timestamp>/sed_fold{0..4}.onnx
```

### 2. 训练 HGNet 分支
```bash
cd training        # hgnet / hgnet_weak_labels 是 training/ 下的包

# 标准标签
python -m hgnet.train          # → best_model_fold{0..3}.pt → 导出为 ONNX

# 弱 / 伪标签 (先用 pseudo_runner 生成伪标签)
python -m hgnet_weak_labels.train
```

### 3. 推理 / 构建提交
```bash
cd inference
# 设置 BIRDCLEF_* 环境变量指向数据 + ONNX 权重 (见 inference/README.md)
python -m scheme_model_5.main   # Model_5 + HGNet 分支 → submission csv
```
v20 (0.946) 和 v22 (0.947) 是同一份代码、不同的 HGNet 环境变量 ——
具体变量集见 [`inference/README.md`](inference/README.md) 和
`inference/SETTINGS.example.json`。

> **路径默认指向 Kaggle 的 `/kaggle/input/...`** (在 Kaggle 上运行时流水线会自动
> 探测)。本地运行时用文档里的 `BIRDCLEF_*` 环境变量覆盖 (如 `BIRDCLEF_BASE`、
> `BIRDCLEF_MODEL_DIR`、`BIRDCLEF_SED_DIR`), 不要直接改代码。各模块 README 列出了
> 相关变量。

## 环境

- Python 3.10–3.11 (HGNet 推理) / 3.10+ (SED 训练)
- PyTorch + torchaudio + torchvision (按你机器的 CUDA 装对应版本)
- `timm`、`onnxruntime`、`openvino`、`librosa`、`soundfile`、`scikit-learn`、`pandas`、`numpy`
- 见各模块的 `requirements.txt`。

## 许可证

本仓库的集成工作和原创修改采用 MIT。上游社区组件归其作者所有 —— 见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 和
[`docs/CREDITS.md`](docs/CREDITS.md)。有疑问时, 请归功于原作者。
