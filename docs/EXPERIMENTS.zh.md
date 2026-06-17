# 实验记录 (诚实版)

> 语言: [English](EXPERIMENTS.md) | **中文**

我实际试过什么、什么有效、什么无效。开放竞赛 write-up 的意义在于可复现和诚实,
而不是声称发明。

## 关键数字

| 配置 | Public LB | Private LB | 说明 |
|---|---|---|---|
| 仅 Model_5 | 0.949 | — | Perch v2 + ProtoSSM/MLP + site×hour prior + distilled-SED |
| v20 = M5 + 单个 HGNet b4 | — | **0.946** | 银牌; HGNet b4 分支权重 0.15 |
| v22 = M5 + HGNet 3-backbone 集成 | — | **0.947** | 银牌; b0+b1+b4(weak) 总权重 0.20 |

## 分支权重 (最终提交)

| 分支 | v20 (0.946) | v22 (0.947) |
|---|---|---|
| Perch v2 + distilled-SED 主 (Model_5) | 0.85 | 0.80 |
| HGNet 分支 | 单 `hgnetv2_b4`, 0.15 | `b0 + b1 + b4(weak)` `[1,1,3]`, 共 0.20 (b0 4% / b1 4% / b4 12%) |

## SED 分支 backbone

两个提交里的 distilled-SED 分支都用 **hgnetv2_b2** (Perch v2 蒸馏)。竞赛期间我试过
几种 SED backbone:

- `hgnetv2_b0`
- `hgnetv2_b2`  ← 银牌提交所用
- `hgnetv2_b4`
- `hgnetv2_b2` 配 **弱 / 伪标签** (训练代码: `training/distilled_sed_weak_labels/`)

SED backbone 通过 `training/distilled_sed/config.py` 的 `BIRDCLEF_BACKBONE` 环境变量切换。

## HGNet 分支模型

- v20: 单个 `hgnetv2_b4` (4 折) —— 用 `training/hgnet/` 训练。
- v22: `hgnetv2_b4` (**弱/伪标签**, `training/hgnet_weak_labels/`) + `hgnetv2_b0`
  (无弱标签) + `hgnetv2_b1` (无弱标签) 的集成, 都是 4 折 ONNX, 内部权重 `[1,1,3]`
  (b4 占主导)。

## 关键观察

- **HGNet 分支拉低 public LB, 但为多样性保留。** 纯 Model_5 (public 0.949) 在
  public split 上比 v20/v22 强。HGNet 分支 (梅尔谱图像分类器, 跟 Perch embedding
  线差异很大) 以小权重 rank-blend 进来, 是为了增加集成多样性、降低 private 榜抖动
  风险。在 private 榜上两个融合提交落在 0.946 / 0.947 —— 稳稳的银牌。

- **多 backbone HGNet > 单 backbone。** v22 (b0 + b1 + b4-weak, 3-backbone ONNX
  集成, mel 只算一次喂给所有 session) 比 v20 (单 b4) private 高约 0.001。内部权重
  `[1,1,3]` 重度偏向弱标签 b4 模型。(还试过加 repvit_m1_1 的 4-backbone 变体, 但
  最终 0.947 提交用的是 3-backbone 版。)

- **SED backbone 替换。** distilled-SED 用 **hgnetv2_b2** 重训, 替代上游的
  EfficientNet-B0。训练保持 Perch v2 作为蒸馏 teacher (在 1536 维 embedding 上做
  MSE), 蒸馏头和 SED 分类头之间用 stop-gradient。

- **弱 / 伪标签。** 伪标签流水线 (`training/pseudo_runner/`) 生成高置信度软标签,
  训练了 v22 集成里用到的额外 HGNet (b4) 模型。

## 推理工程 (对 code 竞赛很重要)

- 一切都跑在 **CPU 上, 通过 ONNX / OpenVINO**, 以适配 notebook 运行时限制。
- HGNet 分支: 12×5s 窗口, 4 折 (或多 backbone) ONNX, 在概率空间 sigmoid 平均,
  高斯 σ=0.65 时间平滑。
- 最终融合用 **rank-percentile** 平均 (对尺度鲁棒), 而不是原始概率平均。

## 上游归因

完整的逐模块归因见 [`docs/CREDITS.md`](CREDITS.md)。简而言之:

- `scheme_model_5` (Perch v2 + ProtoSSM/MLP + prior + distilled-SED) 源自
  **Derek** 的 *"birdclef 2026 exp019 eos4 rank power 06"* notebook, 其
  Perch + ProtoSSM + ResidualSSM 核心源自 **yukiZ (hideyukizushi)**。
- distilled-SED 线: **Tucker Arrants** 的 `bc2026-distilled-sed`。
- HGNet 分支: **ttahara (Tawara)** 的 HGNetV2-B0 baseline。
