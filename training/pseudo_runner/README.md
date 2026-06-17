# 打伪标签 pipeline (单套 Model_5)

把 `train_soundscapes/` 中**未在 train_soundscapes_labels.csv 出现**的文件,
当成 test 跑过 Model_5 (Perch + ProtoSSM + SED),合并成软标签 csv,作为
SED / HGNet 弱标签训练的伪标签源。

> **历史说明**: 早期版本跑 v6 / v13 / v17 三套 scheme 再取平均。但三套之间
> **只有 HGNet 分支的 backbone 不同**,而打伪标签阶段 HGNet 分支根本不参与
> (只用 `submission_protossm.csv` + `submission_sed.csv`,且最终 blend 被
> `runtime_patches` 跳过),三套输出**完全一致**。因此现已简化为**单套 Model_5**,
> 复用发布版的 `inference/scheme_model_5`,不再维护三份副本。

## 设计要点

### 1. 复用单份 scheme_model_5
`run_one_scheme.py` 在子进程里把 `inference/scheme_model_5` 的父目录加进
`sys.path`,然后 `import scheme_model_5` 跑。走子进程是为了让 monkey-patch
和 `BIRDCLEF_*` 环境变量在干净解释器里生效,不污染父进程。

### 2. 用 fake_base 把 unlabeled soundscape 当 test
`main.py` 写死了 `test_paths = sorted((base_dir / "test_soundscapes").glob("*.ogg"))`,
所以建一个 fake_base 目录:

```
fake_base/
  ├── sample_submission.csv → 链接到真实 birdclef-2026/sample_submission.csv
  ├── taxonomy.csv → ...
  ├── train_soundscapes_labels.csv → ...
  ├── train_soundscapes/ → 链接到真实目录
  └── test_soundscapes/ → 含 unlabeled .ogg 的硬链接
```

设环境变量 `BIRDCLEF_BASE=fake_base`,原 pipeline 就以为这是 test。

### 3. 关闭最终阈值化,保留软概率
`runtime_patches` 把 `apply_per_class_thresholds` 换成 identity,让 protossm
输出 [0,1] 软概率而不是二值。同时跳过最终 blend / write_final_submission
(伪标签场景只要 protossm + sed 两个 csv)。

### 4. CUDA 加速 (可选)
- Perch ONNX → CUDAExecutionProvider (可用 `BIRDCLEF_PERCH_FORCE_CPU=1` 回 CPU)
- SED ONNX → CUDAExecutionProvider
- ProtoSSM / ResidualSSM → torch.cuda

### 5. 软概率融合
最终 csv = `0.6 * protossm + 0.4 * sed` (跟 SOLUT.xSED 对齐),
然后 `primary_label_prob > 0.5` 过滤 + `< 0.1` trim + 5 折 GroupKFold 分组。

## 运行 (在 `training/` 目录下)

```bash
cd training

# 一键: build fake_base + 跑 Model_5 + 融合
python -m pseudo_runner.run_all \
    --comp-dir ./data/birdclef-2026 \
    --output-dir ./data/pseudo_output \
    --perch-onnx ./models/perch_v2_no_dft.onnx \
    --sed-dir ./models/distilled_sed_onnx
```

产物: `./data/pseudo_output/pseudo_filtered_grouped.csv` —— 直接喂给
`hgnet_weak_labels` / `distilled_sed_weak_labels` 训练。

文件多时可用 `run_batches.py` 分批跑、`merge_batches.py` 跨批合并,见
`RUN_GUIDE.md`。
