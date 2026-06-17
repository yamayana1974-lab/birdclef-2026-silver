# Tucker Distilled SED — 模块化拆分

把 `bc2026-distilled-sed.ipynb` (14 个 code cell, ~1200 行) 拆成 9 个 `.py` 文件,
让你能逐个模块学习这套**自带 Perch 蒸馏的 SED 训练 + ONNX 推理 pipeline**.

跑出来的 `sed_fold{0..4}.onnx` 就是你在 0.946 v1 方案里用到的 Tucker SED ensemble.

---

## 文件结构

| 文件 | 对应 cell | 内容 |
|---|---|---|
| `config.py` | S0 / S1 | 全局超参 (路径 / 模型 / 蒸馏 / 训练 / 增强 / MixUp / batch 组成) |
| `data.py` | S2 | 加载 waveform cache + 标签矩阵 + StratifiedKFold/GroupKFold + 稀有类上采样 |
| `models.py` | S3 | `MelSpecTransform` / `SpecAugment` / `PerchTeacher` / `DistillHead` / `GeMFreqPool` / `BirdSEDModel` |
| `dataset.py` | S4 | `FocalDS` / `ScDS` + **3 种 MixUp** + `MixSamp` 多源采样 + 波形增强 |
| `train.py` | S5 | `train_fold` 函数 (Loss / AMP / SWA-cosine / 早停 / 双 best 跟踪) |
| `export_onnx.py` | S6 | `SEDExportWrapper` (Linear→Conv1d) + `export_fold_to_onnx` (含一致性验证) |
| `eval.py` | S3+S7 | `compute_macro_auc` / `full_eval` / `print_oof_summary` |
| `inference.py` | Inference cells | librosa mel + ONNX 5-fold ensemble + 高斯平滑 + 写 submission |
| `main.py` | (新增) | 入口, 串起来全流程, MODE=train / infer 切换 |

---

## 完整流程图

```
                          MODE = "train"
                                ↓
                       data.load_data()
                                ↓
       ┌──────────────────── Fold k loop (k=0..4) ────────────────────┐
       │                                                                │
       │   train.train_fold(k, data_bundle)                             │
       │     ├── dataset.FocalDS + ScDS  (ConcatDataset)                │
       │     ├── dataset.MixSamp  (按 SHARES=0.9/0.1 组 batch)          │
       │     ├── models.BirdSEDModel + MelSpec + SpecAugment            │
       │     ├── models.PerchTeacher  (frozen ONNX teacher)             │
       │     └── 每 epoch:                                              │
       │            forward (auto_cast)                                 │
       │              ├── h = backbone(mel)                             │
       │              ├── distill_emb = DistillHead(h)  ←★ grad to bb   │
       │              ├── h.detach() → GeM → bottleneck → att, cla      │
       │              ├── cls_loss = 0.5*BCE(clip) + 0.5*BCE(frame_max) │
       │              └── total = cls_loss + 1.0 * MSE(distill, perch)  │
       │            backward + grad_clip + LR schedule                  │
       │            validate: pool=blend, 算 5 个 AUC, 跟踪 best        │
       │                                                                │
       │   export_onnx.export_fold_to_onnx(k, best_state, ...)          │
       │     ├── 构造 SEDExportWrapper (Linear → Conv1d)                │
       │     ├── load_and_remap_state (丢 distill_head)                 │
       │     └── torch.onnx.export → sed_fold{k}.onnx + 一致性验证       │
       │                                                                │
       └──────────────────────────────────────────────────────────────┘
                                ↓
                  eval.print_oof_summary (5 个指标)


                          MODE = "infer"
                                ↓
                  inference.load_sed_sessions
                    (加载 OUT_DIR 或 Kaggle dataset 的 ONNX)
                                ↓
                  inference.find_test_files
                                ↓
                  inference.run_inference  (对每个 60s 文件)
                    ├── file_to_chunks (60s → 12 段 5s)
                    ├── audio_to_mel (librosa, 128 mels)
                    ├── 每 fold ONNX: clip_logits + framewise_logits
                    ├── fold 内: 0.5*clip + 0.5*frame_max (logit 空间)
                    ├── 5 fold 平均 (logit 空间)
                    └── 高斯平滑 + sigmoid 一次 → probs
                                ↓
                  inference.write_submission → submission.csv ✓
```

---

## ★ 这个 notebook 的 5 个关键创新点 (跟普通 SED 比)

### 1. **Perch v2 知识蒸馏** (★ 最关键)
backbone 输出经过 `DistillHead` 投到 1536-d, MSE 回归 Perch v2 的 emb. 让小 backbone (EfficientNet B0, 4M 参数) 学会 Perch (大 14k 物种 pretrained) 的通用鸟类特征.

> 论文里说 LB 从 **0.876 → 0.898**, +0.022 提升.

### 2. **Stop-Gradient** (跟蒸馏配套)
SED head 用 `h.detach()` 喂, 不更新 backbone. 这样:
- backbone 只被蒸馏 loss 更新 → 学 Perch 知识
- SED head 只被 cls_loss 更新 → 在 frozen-like 特征上训分类

两个 loss 互不干扰, 收敛更稳.

### 3. **GeMFreqPool**
频率维池化用可学的 generalized mean pool (p=3.0 起步, p∈[1,∞)). 比 GAP/max pool 都强:
- p=1 → mean (跟 GAP 一样)
- p→∞ → max
- p=3 → 自适应介于中间, 锐化峰值不丢平均信号

### 4. **Focal-Soundscape MixUp** (★ 域漂移修复)
训练时 50% 概率把 focal clip 跟一个 labeled soundscape 段叠加. 让模型在训练时就**直接见过 soundscape 域**, 显著缩小 focal → soundscape 的域漂移.

### 5. **多源 batch 严格组成**
`MixSamp` 让每个 batch 严格 90% focal + 10% sc, 不是随机抽. 这让训练信号在每个 batch 都稳定 (随机抽样在小 batch 上会有 sc=0 的情况).

---

## 用法

### 推理模式 (默认, 直接用别人/自己导出的 ONNX)

```bash
cd distilled_sed_model
python main.py
```

需要 attach:
- `birdclef-2026` (比赛数据)
- `tuckerarrants/bc2026-distilled-sed-public` (5 个 ONNX, Tucker 公开的) **或** 你自己跑训练导出的 ONNX
- `tuckerarrants/perch-v2-no-dft-onnx` (推理时 **不需要**, 仅训练时蒸馏用)

输出: `submission.csv`

### 训练模式 (自己训出 5 个 ONNX)

```python
# 改 config.py:
MODE = "train"
```

```bash
python main.py
```

需要 attach:
- `birdclef-2026`
- `tuckerarrants/bc2026-waveform-cache` (预切的 int16 .pt waveform + metadata) **必须**
- `tuckerarrants/perch-v2-no-dft-onnx` (Perch teacher) **必须**

输出:
- `OUT_DIR/fold{0..4}_best.pt`  (PyTorch ckpt)
- `OUT_DIR/sed_fold{0..4}.onnx`  (推理用)

### 训练时间预估

每 fold 25 epoch, T4 GPU 大约 1-1.5 小时. 5 fold ≈ 6-8 小时. 全跑完接近 Kaggle 9 小时 GPU 上限, 谨慎.

---

## 推荐研究顺序

按这个顺序看, 信息梯度最平滑:

1. **`README.md`** (本文件) — 总览 + 流程图
2. **`config.py`** — 看一遍超参, 理解每个参数的作用
3. **`data.py`** — 数据加载, 重点看 fold 分配 (Stratified vs Group) 和稀有类上采样
4. **`models.py`** — ★ 重点看 `BirdSEDModel`:
   - `init` 里的 GeMFreqPool + bottleneck + att/cla
   - `forward` 里的 `h.detach()` (stop gradient) + DistillHead
5. **`dataset.py`** — ★ 重点看 `FocalDS.__getitem__` 里的 3 种 MixUp 分支
6. **`train.py`** — ★ 重点看 `train_fold` 里:
   - Loss = cls_loss + α·distill_loss
   - Mel + SpecAugment 在 no_grad 里 (省显存)
   - validate 同时算 3 种 head (clip/fmax/blend) 的 AUC
7. **`export_onnx.py`** — 看 `load_and_remap_state` 怎么把训练 ckpt 重映射到 export wrapper
8. **`inference.py`** — 看 5 fold ensemble 在 logit 空间累加 + 高斯平滑
9. **`eval.py`** — 看 macro AUC 的实现 (跳过全 0 / 全 1 类) 和 non_S22 主指标
10. **`main.py`** — 看怎么串起来

---

## 跟你 0.946 v1 项目 (bird_0.946_project) 里的 sed.py 的关系

`bird_0.946_project/sed.py` 里的 `run_sed_inference` **加载的就是这里训出来的 ONNX**.
区别:

| | bird_0.946_project/sed.py | distilled_sed_model (本目录) |
|---|---|---|
| 作用 | **推理** (加载 ONNX 跑 test) | **训练 + 推理** |
| Mel 库 | librosa | librosa |
| Mel 输入 | 256 mels | **★ ONNX 里是 128 mels** (训练 256, 推理 128, 节省体积) |
| Ensemble | 0.5×sigmoid(clip) + 0.5×sigmoid(frame_max) 在 prob 空间 | 在 **logit 空间**累加, 最后 sigmoid 一次 |
| 平滑 | gaussian σ=0.65 | 5-tap 固定 kernel [0.1, 0.2, 0.4, 0.2, 0.1] |
| 输出 fold 数 | 5 | 5 |

两边等价 (输出概率几乎一样), 实现细节略不同.

---

## 主要 Kaggle 依赖 (训练时)

| 数据集 | 用途 | 必须? |
|---|---|---|
| `birdclef-2026` | 比赛主数据 + taxonomy + sample_submission | ✓ |
| `bc2026-waveform-cache` | 预切的 int16 .pt waveform | ✓ (训练) |
| `perch-v2-no-dft-onnx` | Perch v2 ONNX + onnxruntime wheel | ✓ (蒸馏 teacher) |

推理时只需要前两个 (Perch teacher 用不到).
