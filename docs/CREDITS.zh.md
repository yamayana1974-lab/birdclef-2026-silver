# 致谢与归因

> 语言: [English](CREDITS.md) | **中文**

本方案是对 **BirdCLEF 2026 社区公开代码的集成与扩展**, 不声称有全新的算法贡献。
本仓库的价值在于*工程集成*、*弱标签 / 伪标签扩展*、*模块化的本地可复现重构*,
以及共同达到银牌 private LB 的*集成调参*。

下表每个模块都列出了它的上游来源和具体改动。

---

## 模块归因

| 模块 (本仓库) | 上游来源 | 我的改动 / 新增 |
|---|---|---|
| `training/distilled_sed/` | Tucker Arrants 的公开 notebook **`bc2026-distilled-sed`** (Perch 蒸馏 SED) | backbone 换成 **hgnetv2_b2** (通过 `BIRDCLEF_BACKBONE` 环境变量; 银牌提交用 b2), 调了训练超参, 把单 notebook 拆成模块化可本地复现的包, 加了 5 折 ONNX 导出 + 一致性校验。也试过 b0 / b4。 |
| `training/distilled_sed_weak_labels/` | 同上 Tucker distilled-SED base | **我的扩展**: 在全量数据的弱 / 伪标签上训练 distilled-SED (hgnetv2_b2)。一条实验线, 不是最终提交的 SED。 |
| `training/hgnet/` | **ttahara (Tawara)** 的公开 notebook **"BirdCLEF+2026: HGNetV2-B0 Baseline [Training]"** (社区 baseline, Notebooks Grandmaster; 用 LSEModel + LSEHead + BCEWithLogitsLoss) | 模块化成 Python 包, 修了原 notebook 几个隐藏 bug (见模块 README), 支持换 backbone 到 **hgnetv2_b4** 和其它 timm backbone, 加了 OpenVINO 导出 + CPU OOF 校验。 |
| `training/hgnet_weak_labels/` | 同上 ttahara HGNetV2-B0 baseline | **我的扩展**: 在 baseline 之上的弱标签 / 伪标签训练流程。 |
| `training/pseudo_runner/` | — | **我的代码**: 伪标签生成流水线 (构建 base、高置信度过滤、类别平衡、软标签融合), 喂给 `hgnet_weak_labels`。 |
| `inference/scheme_model_5/` | **Derek** 的 Kaggle notebook **"birdclef 2026 exp019 eos4 rank power 06"** (public LB 0.949; 社区集成榜的 "Model_5" 条目)。其 **Perch v2 + ProtoSSM + ResidualSSM** 核心源自 **yukiZ (hideyukizushi)** 的 notebook **"Bird26.REPRODUCE.Perch+ProtoSSM+ResSSM.INF/TRAIN"**。Derek 的 notebook 加了 **Karnakbayev** 的 "Power Optimization" rank / 后处理链 (LB 0948) + EOS-4 / Model_6 修正网格, 并融入 Tucker Arrants 的 distilled-SED ONNX 分支。 | 把单 `cell_07` notebook 拆成模块化、可本地复现的 Python 包 (config / data / mapping / perch_inference / probes / prior_fusion / protossm_pipeline / sed_pipeline / blend), 让所有写死路径都能用 `BIRDCLEF_*` 环境变量覆盖, 并加了 HGNet 分支 rank-blend (v20 / v22)。核心算法和分数路径保持不变。 |
| `inference/hgnet_branch.py` | 基于上面的 HGNet baseline | 我的代码: 4 折 / 多 backbone ONNX 集成推理 + rank-blend 工具, 把 HGNet 分支融进 Model_5 的提交。 |

> `scheme_model_5` 的 Perch + ProtoSSM/MLP + site×hour prior + distilled-SED 流水线
> 源自 Derek 的公开 notebook *"birdclef 2026 exp019 eos4 rank power 06"* (社区
> "Model_5" 线, LB 0.949)。其 Perch + ProtoSSM + ResidualSSM 核心源自
> **yukiZ (hideyukizushi)** 的 *"Bird26.REPRODUCE.Perch+ProtoSSM+ResSSM.INF/TRAIN"*;
> Derek 的 notebook 在其上叠加了 Karnakbayev 的 power-optimization 后处理。这套
> 代码曾被广泛 fork, 这是能确认的最接近上游。如果你是其中任何部分的原作者、希望
> 不同 / 补充的归因, 请提 issue。
>
> *(原 notebook 里还有第二个 "Model_2" Perch/ProtoSSM 变体 (yukiZ 复现), 但两个
> 银牌提交都没用到, 已从本仓库移除。)*

---

## 预训练模型 / 外部资源 (本仓库不转发)

这些在训练 / 推理时需要, 但**不**包含在本仓库里。请从原始来源下载:

| 资源 | 来源 | 用途 |
|---|---|---|
| Perch v2 (鸟鸣分类器) | Google / Kaggle `google/bird-vocalization-classifier` | SED 蒸馏 teacher + Model_5 推理 |
| Perch v2 ONNX (`perch_v2_no_dft.onnx`) | Kaggle 社区数据集 | Perch 的 CPU 推理 |
| Distilled-SED 公开 ONNX | `tuckerarrants/bc2026-distilled-sed-public` | SED 分支可选的现成替代 |
| ttahara train-audio WAV 数据集 (00–03) | Kaggle `ttahara/birdclef2026-train-audio-wav-0{0..3}` | HGNet 用的预转换训练音频 |

---

## 如何引用原竞赛

BirdCLEF++ 2026, Kaggle / LifeCLEF。基于本工作时, 请同时致谢竞赛主办方和上面列出
的原 notebook 作者。
