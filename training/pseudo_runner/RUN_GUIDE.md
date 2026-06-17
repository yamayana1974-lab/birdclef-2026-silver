# 服务器运行指南 (单卡 4090, 24GB)

> **单套说明**: 现已简化为单套 Model_5 (`--schemes m5`,默认值)。早期的
> v6/v13/v17 三套在打伪标签阶段输出完全一致 (HGNet 分支不参与),故合并为一套。

## 一、数据规模 + 时间预估

实测 train_soundscapes 总共 **10658 个文件** (66 labeled + 10592 unlabeled).

要做:
- Perch v2 ONNX 推理: 10592 文件 × 12 段 = **127,104 段**
- ProtoSSM TTA forward: 上面 127k 段 × 6 次 (5 shift + 1 flip)
- ResidualSSM forward: 127k 段 × 1 次
- Distilled-SED ONNX: 127k 段 × 5 fold

预估单卡 4090 耗时:
| 阶段 | 时间 |
|---|---|
| Perch ONNX 推理 (CUDA) | ~25 min |
| ProtoSSM 训练 (66 文件训练数据, CPU) | ~2 min |
| ProtoSSM TTA + Residual forward | ~5 min |
| Distilled-SED 5-fold ONNX (CUDA) | ~30 min |
| **合计** | **~60 min** |

## 二、依赖准备

服务器上需要:

```
./data/
├── birdclef-2026/                          # 比赛数据 (~25GB)
├── perch_v2_no_dft.onnx                    # Perch v2 ONNX
├── perch_v2_cpu/                           # (可选) Perch TF SavedModel
└── distilled_sed_onnx/                     # 含 sed_fold0..4.onnx
```

环境包:

```bash
pip install onnxruntime-gpu==1.22.0  # GPU 加速; 没 GPU 用 onnxruntime 也能跑(慢)
pip install torch torchaudio
pip install timm scikit-learn pandas tqdm soundfile librosa scipy
# tensorflow 仅在用 Perch TF SavedModel 后端时才需要, ONNX 后端不需要
```

## 三、快速验证 (5 文件冒烟测试)

正式跑之前先用 5 个文件验证 pipeline 通畅:

```bash
cd /path/to/repo/training

python -m pseudo_runner.run_all \
    --comp-dir ./data/birdclef-2026 \
    --output-dir ./data/pseudo_smoke \
    --perch-onnx ./data/perch_v2_no_dft.onnx \
    --sed-dir ./data/distilled_sed_onnx \
    --max-files 5
```

正常会看到:
- `pseudo_smoke/fake_base/test_soundscapes/` 含 5 个 .ogg
- `pseudo_smoke/m5/submission_protossm.csv` 60 行 (5 文件 × 12 段)
- `pseudo_smoke/m5/submission_sed.csv` 60 行
- `pseudo_smoke/pseudo_m5.csv` 60 行
- `pseudo_smoke/pseudo_filtered_grouped.csv` 取决于阈值过滤后剩多少行

总耗时应该在 5 分钟内 (Perch cache 第一次构建慢一点).

## 四、正式跑 (全量)

```bash
cd /path/to/repo/training

# 后台运行, 输出写到 log
nohup python -m pseudo_runner.run_all \
    --comp-dir ./data/birdclef-2026 \
    --output-dir ./data/pseudo_output \
    --perch-onnx ./data/perch_v2_no_dft.onnx \
    --sed-dir ./data/distilled_sed_onnx \
    > pseudo_run.log 2>&1 &

# 监控
tail -f pseudo_run.log
```

文件多时可用 `run_batches.py` 分批跑 + `merge_batches.py` 跨批合并 (中断可恢复):

```bash
python -m pseudo_runner.run_batches \
    --comp-dir ./data/birdclef-2026 \
    --output-dir ./data/pseudo_output \
    --perch-onnx ./data/perch_v2_no_dft.onnx \
    --sed-dir ./data/distilled_sed_onnx \
    --num-batches 10
# 全部跑完后:
python -m pseudo_runner.merge_batches --output-dir ./data/pseudo_output
```

## 五、产物

```
pseudo_output/
├── fake_base/                              # 临时(可删)
├── _work/m5/                               # Perch cache (可删)
├── m5/
│   ├── submission_protossm.csv
│   └── submission_sed.csv
├── pseudo_m5.csv                           # 合并 (proto 0.6 + sed 0.4)
├── pseudo_ensemble.csv                     # 单套时 = pseudo_m5
└── pseudo_filtered_grouped.csv             # ★★ 直接喂训练分支
```

`pseudo_filtered_grouped.csv` 字段:
- `row_id`: `{soundscape_stem}_{end_sec}`
- 234 列 primary_label 概率 (软标签, < 0.1 已置零)
- `primary_label`: argmax 类名
- `primary_label_prob`: 最大概率 (> 0.5)
- `fold_id`: 0~4, 用于训练时排除当前 fold 防泄漏

## 六、常见问题排查

### 1. 报 `No module named 'pseudo_runner'`
确保 cwd 是 `training/` 目录, 用 `python -m pseudo_runner.xxx` 方式运行.

### 2. ONNX 没用上 GPU
检查 `pip list | grep onnxruntime`, 应该是 `onnxruntime-gpu`.
log 里能看到:
```
[perch] ONNX backend: perch_v2_no_dft.onnx  providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
[sed] sed_fold0.onnx providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 3. 显存不够 (OOM)
降 `batch_files` (改 `inference/scheme_model_5/config.py` 的 `CFG['batch_files']`).

### 4. 跳过部分阶段
- `--skip-build-base`: 已经建过 fake_base, 跳过链接
- `--skip-inference`: 推理 csv 已经有, 直接做 merge
- `--no-merge`: 只跑推理不合并 (调试用)

## 七、训练侧消费

把 `pseudo_filtered_grouped.csv` 复制到 SED / HGNet 弱标签训练分支的数据目录,
参考各分支 README 接入 (默认走 `HGNET_PSEUDO_CSV` / SED config 的 PSEUDO_CSV 路径).
