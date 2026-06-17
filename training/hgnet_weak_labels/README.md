# BirdCLEF+2026 · HGNetV2-B0 Baseline (多文件复现版)

本目录是把原 Kaggle notebook 拆分而成的多文件 Python 包，方便后续在本地/服务器上复现训练 + 推理。

源 notebook：

- `birdclef-2026-hgnetv2-b0-baseline-training.ipynb`
- `birdclef-2026-hgnetv2-b0-baseline-inference.ipynb`

## 目录结构

```
hgnet/
├── __init__.py        # 包入口 + 模块速览
├── config.py          # 全局配置 / 路径 / CFG / INFER_CFG
├── utils.py           # 随机种子 / to_device / rank_normalize / sigmoid / device
├── data.py            # 标签加载 + soundscape 5s 区间合并切片 + 多标签分层分组 K-Fold
├── dataset.py         # BirdTrainDataset / BirdValidDataset / get_data_loader
├── transforms.py      # LogMelSpectrogramTransform / MixUp / dummy_mixup
├── models.py          # GeM/AttnSED/LSE 全部模型 + CustomBCEWithLogitsLoss
├── prepare_wav.py     # 官方 train_audio/*.ogg -> wav 批量预转换 (复刻 ttahara 的 4 份数据集)
├── train.py           # 单折训练 train_one_fold + 主入口 main()
├── export.py          # torch -> ONNX -> OpenVINO + CPU OOF AUC 校验
├── infer.py           # 推理 (±2.5s TTA) + 生成 submission.csv
├── requirements.txt   # Python 依赖
└── README.md          # 本文档
```

## 与原 notebook 的差异（修了几处隐性 bug）

1. **`with torch.no_grad() and torch.autocast(...)`**：原 notebook 用了 `and`，相当于把
   `no_grad()` 的结果当布尔短路求值，**`no_grad` 并不会真正生效**。已改成正确的多上下文：
   `with torch.no_grad(), torch.autocast(...):`。
2. **`f"{fn.split(".")[0]}.wav"`**：PEP 701 的写法，**只在 Python 3.12+ 才合法**。已改成
   `f"{fn.split('.')[0]}.wav"`，兼容 3.10 / 3.11。
3. **soundscape 末尾片段命名**：原 notebook 最后一个挂起片段写成 `f"{ai}_{start}-{end}.wav"`，
   实际应使用合并起点 `tmp_start / tmp_end`。已修正。
4. **路径与超参从全局散落变量收敛到 `config.py`**：所有 `RANDOM_SEED / N_FOLDS / 路径 / CFG`
   都从 `config.py` import，不再依赖 notebook 顶部全局变量。

逻辑/超参与原 notebook 严格一致，可直接复现 baseline 结果。

## 数据 / 模型路径假设

默认按 Kaggle Kernel 布局，`Path.cwd().parent` 同级有 `input/` 目录：

```
<ROOT>/
├── working/           # 你的 cwd
└── input/
    ├── competitions/birdclef-2026/{train.csv, taxonomy.csv, train_soundscapes_labels.csv, train_audio/, train_soundscapes/, test_soundscapes/, sample_submission.csv}
    └── datasets/ttahara/birdclef2026-train-audio-wav-{00..03}/<primary_label>/<XCxxx>.wav
```

本地复现时只需修改 `config.py` 中的 `ROOT / INPUT / DATA / TRAIN_AUDIO_WAVS` 几个常量，或者
直接 export 环境变量后在 `config.py` 头部读取也可以。

预转好的 wav 数据集来自：

- <https://www.kaggle.com/datasets/ttahara/birdclef2026-train-audio-wav-00>
- <https://www.kaggle.com/datasets/ttahara/birdclef2026-train-audio-wav-01>
- <https://www.kaggle.com/datasets/ttahara/birdclef2026-train-audio-wav-02>
- <https://www.kaggle.com/datasets/ttahara/birdclef2026-train-audio-wav-03>

如果不想从 Kaggle 下载（或要在本地从官方 ogg 自己预转），用 `prepare_wav.py`
一行命令就能复刻。核心函数和 ttahara 在 Kaggle 讨论区给的官方转换脚本完全一致
（`soundfile.read(dtype="float32")` → `soundfile.write(format="wav", subtype="FLOAT")`），
外层加了并行 / 断点续跑 / 错误隔离 / 可选切 4 份等批处理逻辑。

```bash
cd training

# 方案 A: 输出到一个目录 (路径最简单, 然后改 config.TRAIN_AUDIO_WAVS)
python -m hgnet_weak_labels.prepare_wav \
    --output-root ./data/wav-output --split 1 --n-jobs 16

# 方案 B: 严格复刻 ttahara 的 4 份布局 (config 不用改)
python -m hgnet_weak_labels.prepare_wav \
    --output-root <INPUT>/datasets/ttahara --split 4 --n-jobs 16
```

详细参数说明见 `python -m hgnet_weak_labels.prepare_wav --help`。

## 环境

```bash
# 建议 Python 3.10 ~ 3.11
pip install -r requirements.txt
```

`torch / torchaudio / torchvision` 需要根据 CUDA 版本去 [pytorch.org](https://pytorch.org) 安装对应版本，
`requirements.txt` 里只写了名字，没有钉死版本。

## 复现流程

> 下列命令都默认 `cd` 到 `training/` 目录（也就是 `hgnet_weak_labels` 的父目录），
> 用 `python -m` 把模块当包来跑，这样 `from .config import ...` 这种相对 import 才会生效。

### 1. 训练 4 折

```bash
cd training
python -m hgnet_weak_labels.train
```

会做：

1. 读 `train.csv` / `train_soundscapes_labels.csv` / `taxonomy.csv`；
2. 对 `train_soundscapes` 的连续 5s 标注区间做合并 + 切 wav，落到
   `<ROOT>/processed_data/train_soundscapes_split/`；
3. `MultiLabelStratifiedGroupKFold(K=4)`；
4. 4 折分别训练 20 epoch (AdamW + OneCycleLR + AMP + MixUp，`hgnetv2_b0` + LSEHead，BCEWithLogitsLoss)；
5. 输出 OOF 的 macro ROC-AUC（原始 + rank-normalized）。

训练产物（落在 cwd）：

```
best_model_fold{0..3}.pt
best_val_pred_fold{0..3}.npy
result_df_fold{0..3}.csv
log_mel_spectrogram.jb
```

每折约 30 分钟（T4×1），全部跑完不到 2 小时。

### 2. 导出 OpenVINO 模型 + CPU 校验

```bash
python -m hgnet_weak_labels.export
```

会做：

1. 把 4 折 `.pt` -> ONNX (opset 11, dynamic batch) -> OpenVINO `.xml` / `.bin`；
2. 用 OpenVINO 在 CPU 上重新跑一遍 OOF，打印 raw / rank 两种 AUC，校验数值一致性。

产物：

```
best_model_fold{0..3}.xml
best_model_fold{0..3}.bin
```

### 3. 推理 + 生成 `submission.csv`

```bash
python -m hgnet_weak_labels.infer
```

会做：

1. 读 `sample_submission.csv` 判断当前环境是隐藏测试集还是本地 debug；
2. 对每条 60s 音频构造 normal 12 段 + shifted (±2.5s) 13 段两套切片；
3. CPU 上 joblib 并行计算 log-mel；
4. 用 4 折 OpenVINO 异步推理 + sigmoid；
5. TTA 融合：`0.25·shifted[0:12] + 0.5·normal + 0.25·shifted[1:13]`，再按 fold 取平均（可
   切换 `INFER_CFG.rank_avg=True` 用 rank 平均）；
6. 按 `sample_submission` 的 `row_id` 顺序输出 `submission.csv`。

## 常用调参点

集中在 `config.py` 里：

- `CFG.max_epoch / warmup_epoch / batch_size / lr / weight_decay`
- `CFG.head_dropout / lse_temperature / mixup / use_amp / use_dp`
- `CFG.num_workers`（**本地内存不够时务必下调，默认 16**）
- `INFER_CFG.batch_size / lms_n_jobs / num_requests / ov_compile_config / rank_avg`

## 提示

- 训练阶段需要的 wav 总量很大（4 份 wav 数据集），跑通前请确认硬盘空间够用。
- soundscape 切片会写到 `<ROOT>/processed_data/train_soundscapes_split/`，首次运行需要写盘
  几百到几千个 wav 片段，**第二次运行不会自动跳过**，如不需要重新切片可以注释掉 `train.py`
  里 `split_train_soundscapes(...)` 那一行（前提是 `processed_data` 已存在），自行 reload。
- `LSEModel` / `AttnSEDModel` / `CustomBCEWithLogitsLoss` 都保留在 `models.py`，方便做对比实验。
  Baseline 用的就是 `LSEModel` + `nn.BCEWithLogitsLoss`。
