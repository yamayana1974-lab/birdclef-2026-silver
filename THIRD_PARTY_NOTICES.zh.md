# 第三方声明

> 语言: [English](THIRD_PARTY_NOTICES.md) | **中文**

**在认为 MIT `LICENSE` 覆盖本仓库全部内容之前, 请先读这份。**

本仓库的 MIT `LICENSE` **仅适用于仓库所有者原创的集成工作和修改** —— 即模块化
重构、`BIRDCLEF_*` 环境变量的接线、弱标签 / 伪标签扩展、集成 / rank-blend 的连接,
以及文档。

本项目**集成并改编了 BirdCLEF++ 2026 社区公开的代码** (Kaggle notebook 和社区
baseline)。这些上游组件仍归各自作者所有, 受**其各自的 license 和 Kaggle 规则**
约束, 而非本仓库的 MIT license。上游来源若声明了 license, 则对应代码以该 license
为准。

本仓库**不**转发竞赛数据或预训练模型权重。请从原始来源获取 (见下文及
`docs/CREDITS.md`)。

## 上游组件

| 本仓库中的组件 | 上游作者 / 来源 | 备注 |
|---|---|---|
| `inference/scheme_model_5/` (Perch v2 + ProtoSSM + ResidualSSM 核心) | **yukiZ (hideyukizushi)** — *"Bird26.REPRODUCE.Perch+ProtoSSM+ResSSM.INF/TRAIN"* (Kaggle) | 核心算法。这里重构成了包。 |
| `inference/scheme_model_5/` (rank power-optimization 后处理 + SED 融合) | **Derek** — *"birdclef 2026 exp019 eos4 rank power 06"* (Kaggle); 引用 **Karnakbayev** 的 power-optimization 链 | 社区 "Model_5" 线 (LB 0.949)。 |
| `training/distilled_sed/`, `training/distilled_sed_weak_labels/` | **Tucker Arrants** — *`bc2026-distilled-sed`* (Kaggle, Perch 蒸馏 SED) | backbone 换成 hgnetv2_b2; 模块化。 |
| `training/hgnet/`, `training/hgnet_weak_labels/`, `inference/hgnet_branch.py` | **ttahara (Tawara)** — *"BirdCLEF+2026: HGNetV2-B0 Baseline [Training]"* (Kaggle) | 模块化; 加了多 backbone ONNX 集成。 |
| `training/pseudo_runner/` | 仓库所有者 (原创) | 围绕上述模型的伪标签生成。 |

## 外部资源 (本仓库不转发)

| 资源 | 来源 |
|---|---|
| Perch v2 鸟鸣分类器 (+ ONNX) | Google / Kaggle `google/bird-vocalization-classifier` |
| Distilled-SED 公开 ONNX | `tuckerarrants/bc2026-distilled-sed-public` |
| 预转换的 train-audio WAV 数据集 | Kaggle `ttahara/birdclef2026-train-audio-wav-0{0..3}` |
| BirdCLEF++ 2026 竞赛数据 | Kaggle 竞赛 `birdclef-2026` (受竞赛规则约束) |

## 如果你是上游作者

这套代码曾被广泛 fork, 确切的出处是从代码注释和公开集成榜重建的。如果你撰写了
其中任何部分、希望不同 / 补充 / 移除归因, 请提 issue。

每个模块改了什么的详细分解见 [`docs/CREDITS.md`](docs/CREDITS.md)。
