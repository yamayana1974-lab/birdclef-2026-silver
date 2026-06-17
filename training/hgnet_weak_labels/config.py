"""全局配置 / 路径 / 常量
=================================

把 notebook 里散落的 RANDOM_SEED / 路径 / N_FOLDS / N_CLASSES / CFG
集中管理, 训练、推理脚本统一从这里 import.

如果迁到本地或者其它 Kaggle 数据集, 只需要修改 ``ROOT``/``INPUT``/``TRAIN_AUDIO_WAVS``
等少量常量即可.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path


# ============================================================
#                       0. 随机种子
# ============================================================
RANDOM_SEED: int = 1086
# 注意: PYTHONHASHSEED 必须在导入 numpy / torch 之前设置, 这里在 import 顶部生效.
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))


# ============================================================
#                       1. 路径常量
# ============================================================
# 默认按照 Kaggle Kernel 的目录布局: /kaggle/working 为 cwd, 同级有 input/.
# 在本地复现时, 把 ROOT 改成你存放 input/ 的目录即可.
# Linux 服务器 / Windows 本地切换可用环境变量 BIRDCLEF_TRAIN_ROOT 覆盖,
# 例如:  export BIRDCLEF_TRAIN_ROOT=/data/birdclef
ROOT: Path = Path(os.environ.get(
    "BIRDCLEF_TRAIN_ROOT",
    "./data",
))
INPUT: Path = ROOT
DATA: Path = ROOT / "birdclef-2026"
TRAIN_AUDIO: Path = DATA / "train_audio"
TRAIN_SS: Path = DATA / "train_soundscapes"
TEST_SS: Path = DATA / "test_soundscapes"


# 4 份预转好的 wav 数据集 (加速训练加载).
# https://www.kaggle.com/datasets/ttahara/birdclef2026-train-audio-wav-00..03
#
# 想直接用官方 .ogg 跳过转 wav, 把 USE_OGG_DIRECT=True 即可:
# - TRAIN_AUDIO_WAVS 会在 data.py 里被替换成 [TRAIN_AUDIO] 单目录,
# - 文件后缀使用 .ogg 而不是 .wav.
USE_OGG_DIRECT: bool = True       # ★ True = 不转 wav, 直接读 train_audio/*.ogg
TRAIN_AUDIO_WAVS = (
    [TRAIN_AUDIO]                          # 单目录, ogg 模式
    if USE_OGG_DIRECT
    else [
        INPUT / "datasets" / f"ttahara/birdclef2026-train-audio-wav-{i:02}"
        for i in range(4)
    ]
)

# soundscape 切片落盘位置.
PROC: Path = ROOT / "processed_data"
TRAIN_SS_SPLIT: Path = PROC / "train_soundscapes_split"

# 伪标签 csv (merge_batches.py 产物). 训练时 dataset 加载这份做 40% 同类替换.
# 默认指向 pseudo_runner 跑出来的最终结果,
# 也支持 HGNET_PSEUDO_CSV 环境变量覆盖.
PSEUDO_CSV: Path = Path(os.environ.get(
    "HGNET_PSEUDO_CSV",
    str(ROOT / "pseudo_output" / "pseudo_filtered_grouped.csv"),
))
# 伪标签段对应的 soundscape 原始 ogg 目录 (用 row_id 还原 filename + start_sec 后定位).
PSEUDO_SC_DIR: Path = TRAIN_SS

# 训练完产物的默认根目录: ``<cwd>/runs/<model_short>/<timestamp>/``.
# - ``RUNS_ROOT``     : 所有 run 的父目录, 默认是 ``<cwd>/runs``.
# - ``make_run_dir``  : 训练时调一次, 新建带时间戳的子目录, 用于落盘当前训练产物.
# - ``resolve_run_dir``: 推理 / 导出时调, 默认返回 "当前 model 的最新 run 目录";
#                       也可以通过环境变量 ``HGNET_RUN_DIR`` 强制指定某个具体目录,
#                       用于回挑历史模型.
RUNS_ROOT: Path = Path.cwd() / "runs"


def _model_short(model_name: str) -> str:
    """从 timm model_name 取前缀做目录名, 例如 hgnetv2_b1.ssld_xxx -> hgnetv2_b1."""
    return model_name.split(".")[0]


def make_run_dir(model_name: str, runs_root: Path = RUNS_ROOT) -> Path:
    """新建一个带时间戳的训练产物目录并返回. 多次训练互不覆盖."""
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = runs_root / _model_short(model_name) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_run_dir(model_name: str, runs_root: Path = RUNS_ROOT) -> Path:
    """推理 / 导出时使用. 解析顺序:

    1. 环境变量 ``HGNET_RUN_DIR`` 指向具体的 run 目录 (绝对或相对均可);
    2. 否则取 ``<runs_root>/<model_short>/`` 下最新的子目录;
    3. 都没有就回退到 ``cwd``, 兼容旧的"直接落 cwd"用法.
    """
    env_dir = os.environ.get("HGNET_RUN_DIR")
    if env_dir:
        p = Path(env_dir)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p

    model_root = runs_root / _model_short(model_name)
    if model_root.exists():
        candidates = sorted([p for p in model_root.iterdir() if p.is_dir()])
        if candidates:
            return candidates[-1]

    return Path.cwd()


# 兼容字段: 老代码若直接 import TRAINED_MODEL_DIR 仍可用,
# 但推荐改用 resolve_run_dir(CFG.model_name).
TRAINED_MODEL_DIR: Path = Path.cwd()


# ============================================================
#                       2. 数据集常量
# ============================================================
N_FOLDS: int = 4
N_CLASSES: int = 234

SAMPLING_RATE: int = 32_000
SEGMENT_SEC: int = 5


# ============================================================
#                       3. 训练超参 CFG
# ============================================================
class CFG:
    """训练阶段的超参数集中配置."""

    # ---- 训练循环 ----
    # 4090 24GB 单卡, hgnetv2_b2:
    #   batch=96 实测 ~17-18 GB,留余量给 mixup/amp 缓冲;想再激进可拉到 128.
    #   batch 从 128(b0) 减到 96(b2),lr 按 linear scaling 同步下调 (1e-3 → 7.5e-4).
    max_epoch: int = int(os.environ.get("HGNET_MAX_EPOCH", "20"))
    warmup_epoch: int = int(os.environ.get("HGNET_WARMUP_EPOCH", "5"))
    batch_size: int = int(os.environ.get("HGNET_BATCH_SIZE", "96"))
    lr: float = float(os.environ.get("HGNET_LR", "7.5e-4"))
    weight_decay: float = float(os.environ.get("HGNET_WEIGHT_DECAY", "1e-4"))
    num_workers: int = int(os.environ.get("HGNET_NUM_WORKERS", "12"))  # 4090 平台 CPU 通常 12 核+, 设 12 让 IO 不卡 GPU.

    # ---- 模型 ----
    # 支持环境变量覆盖, 方便同机多终端并行跑不同 backbone:
    #   HGNET_MODEL_NAME=hgnetv2_b1.ssld_stage2_ft_in1k python -m hgnet_weak_labels.train ...
    # 同时也可通过 HGNET_BATCH_SIZE / HGNET_LR / HGNET_DROP_PATH 微调,
    # 不传就用下面的默认值 (b2 baseline).
    model_name: str = os.environ.get(
        "HGNET_MODEL_NAME", "hgnetv2_b2.ssld_stage2_ft_in1k"
    )
    pretrained: bool = True
    drop_path_rate: float = float(os.environ.get("HGNET_DROP_PATH", "0.15"))  # b2 比 b1 再大一档, 略增 drop_path 防过拟合.
    head_dropout: float = 0.5
    lse_temperature: float = 1.0

    # ---- log-mel ----
    mel_spectrogram_params = dict(
        sample_rate=32_000,
        n_fft=2048,
        win_length=626,
        hop_length=313,
        f_min=20,
        n_mels=256,
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="htk",
    )
    lms_shape = (256, 256)
    top_db: float = 80.0

    # ---- 其它 ----
    mixup = dict(alpha=1.0, theta=0.8)
    use_amp: bool = True
    # AMP 精度: "fp16" (默认, 兼容性好) 或 "bf16" (数值更稳, ConvNeXt V2/GRN 必选).
    amp_dtype: str = os.environ.get("HGNET_AMP_DTYPE", "fp16")
    use_dp: bool = False  # 多卡 DataParallel.

    # ---- 伪标签 (weak labels) ----
    # 用 unlabeled soundscape 的伪标签替换 XC 单物种数据, 缩小域漂移.
    # 替换规则 (复刻 2nd place 同类替换思路):
    #   1) 仅对 XC train_audio 行 (.ogg) 做替换判断, soundscape 5s 真标签行不动
    #   2) 仅对 primary_label 在伪标签覆盖类集合 (pseudo_birds) 内的行替换
    #   3) 以 PSEUDO_REPLACE_PROB 概率把当前 XC 样本替换成同类的伪标签 5s 段
    # 替换后 wave = 伪标签段, label = 234 维软概率 (< 0.1 已 trim 为 0)
    use_pseudo_replace: bool = bool(int(os.environ.get("HGNET_USE_PSEUDO", "1")))
    pseudo_replace_prob: float = float(os.environ.get("HGNET_PSEUDO_PROB", "0.4"))


# ============================================================
#                       4. 推理超参 INFER_CFG
# ============================================================
class INFER_CFG:
    """推理 / 提交阶段的可调参数."""

    # 推理 batch (送给 OpenVINO 的 batch). 训练 ONNX 导出时用了 64,
    # 但 ONNX 设了 dynamic batch, 12 / 13 这种小 batch 也能跑.
    batch_size: int = 12

    # log-mel 并行计算的 joblib 进程数 (CPU).
    lms_n_jobs: int = 4

    # OpenVINO AsyncInferQueue 的并行请求数.
    num_requests: int = 16

    # OpenVINO 编译参数.
    ov_compile_config = {
        "PERFORMANCE_HINT": "THROUGHPUT",
        "INFERENCE_NUM_THREADS": 4,
        "NUM_STREAMS": 2,
    }

    # 是否对每折的输出做 rank-normalize 再做 fold 平均.
    rank_avg: bool = False

    # 本地调试 / 提交模式控制.
    # DEBUG = True 时, 没有 sample_submission 的环境会用 600 条 train_soundscapes 试跑.
    debug: bool = True
