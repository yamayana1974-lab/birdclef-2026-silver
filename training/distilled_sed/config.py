# -*- coding: utf-8 -*-
"""
config.py — 全局配置 + 超参
=============================

对应 notebook 的 S0 / S1 cell.

包含所有可调参数, 改一处生效所有地方:
  - SEED / DEVICE / MODE 开关
  - 路径 (本地 / Kaggle 自动适配)
  - 音频常量 (SR, TRAIN_DURATION, NUM_CLASSES)
  - 梅尔频谱参数 (n_mels=256, hop=512, ...)
  - 模型 backbone (默认 EfficientNet B0 ImageNet-NS pretrained)
  - 训练超参 (lr, batch, epochs, fold, ...)
  - 增强参数 (gain / noise / MixUp / SpecAugment)
  - 多源 batch 组成 (focal 9 : sc 1)

注意:
  ★ INF_N_MELS = 128 (推理时), 而 N_MELS = 256 (训练时).
    ONNX 导出 wrapper 用的是 128 (这是 v1 v2 的小细节, 看 export_onnx.py).
"""
import os
import random
from pathlib import Path

# ★ HuggingFace 镜像 (autodl 等国内服务器连不上 hf 官方时启用)
#   timm 加载 backbone 预训练权重会从 hf 拉, 设了这个就走镜像.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch


# =============================================================================
# 1. 模式开关
# =============================================================================
# "train" : 训练 + ONNX 导出 + OOF 评估
# "infer" : 跳过训练, 直接加载 ONNX 跑 test → submission.csv
#
# 优先级: 命令行 --mode > 环境变量 BIRDCLEF_MODE > 文件默认值
MODE = os.environ.get("BIRDCLEF_MODE", "train")

# DEBUG 时 epoch=1, fold=[0], 数据各取一小部分 (Kaggle 上调试用)
# 第一次跑通流程建议 DEBUG=True, 跑通后改 False 正式训练
DEBUG = os.environ.get("BIRDCLEF_DEBUG", "0") in ("1", "true", "True")


# =============================================================================
# 2. 路径配置 (本地 layout)
# =============================================================================
# 项目根目录: 默认 ./data, 用环境变量 BIRDCLEF_SED_ROOT 覆盖.
# 例如:  export BIRDCLEF_SED_ROOT=/data/birdclef
_PROJECT_ROOT      = Path(os.environ.get("BIRDCLEF_SED_ROOT", "./data"))
COMP_DIR           = _PROJECT_ROOT / "birdclef-2026"
WAVEFORM_CACHE_DIR = _PROJECT_ROOT / "waveform_cache"        # ← build_cache.py 产出
PERCH_ONNX_PATH    = _PROJECT_ROOT / "perch_v2.onnx"         # ← convert_perch_onnx.py 产出
PERCH_CACHE_DIR    = _PROJECT_ROOT / "perch_cache_v2"        # ← precompute_perch_emb.py 产出

LABELS_PATH        = COMP_DIR / "train_soundscapes_labels.csv"            # ★ 全 66 个 (59 full + 7 partial)
TRAIN_SC_LABELS    = LABELS_PATH                                          # ★ 训练用全 66 个 (与 notebook / 公开 ONNX 一致)
LABELS_FULL_PATH   = COMP_DIR / "train_soundscapes_labels_full.csv"     # 59 个完全标注文件
LABELS_PART_PATH   = COMP_DIR / "train_soundscapes_labels_partial.csv"  # 7 个部分标注文件
TAXONOMY_PATH      = COMP_DIR / "taxonomy.csv"
SAMPLE_SUB_PATH    = COMP_DIR / "sample_submission.csv"
TEST_DIR           = COMP_DIR / "test_soundscapes"

# 输出目录 (所有训练产物的根目录)
OUT_DIR = _PROJECT_ROOT / "distilled_sed_model" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# get_run_dir() 在 6 节 BACKBONE_NAME 定义之后, 见下文.


# =============================================================================
# 3. 随机种子 + 设备
# =============================================================================
SEED = 42


def seed_everything(seed=SEED):
    """跨 numpy / torch / python 设种子."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # benchmark=True 比 deterministic=True 快, 训练用
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 4. 音频参数 (训练 + 推理共享)
# =============================================================================
SR             = 32_000                         # 采样率 (BirdCLEF 标准)
NUM_CLASSES    = 234                            # 物种数

TRAIN_DURATION = 5                              # 训练 clip 时长 (秒)
TRAIN_SAMPLES  = TRAIN_DURATION * SR            # 160000

VAL_DURATION   = 5                              # 验证 clip 时长 (跟训练一致)
VAL_SAMPLES    = VAL_DURATION * SR


# =============================================================================
# 5. 梅尔频谱参数
# =============================================================================
# ★ 训练时用 256 mels
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 256
FMIN       = 20
FMAX       = 16000


# =============================================================================
# 6. 模型 backbone
# =============================================================================
# 可选 backbone(改这一行就生效, 其他代码会自动适配 backbone_dim):
#
#   "tf_efficientnet_b0.ns_jft_in1k"      4.0M  out C=1280  ← ★ 当前选 (跟 notebook 完全一致)
#   "tf_efficientnet_b2.ns_jft_in1k"      7.7M  out C=1408
#   "tf_efficientnet_b3.ns_jft_in1k"     10.7M  out C=1536
#   "hgnetv2_b0.ssld_stage2_ft_in1k"      4.0M  out C=2048  ← HGNet 同量级对比
#   "hgnetv2_b3.ssld_stage2_ft_in1k"     14.2M  out C=2048
#   "hgnetv2_b5.ssld_stage2_ft_in1k"     37.5M  out C=2048
#   "edgenext_small.usi_in1k"             5.6M  hybrid CNN+Transformer
#   "edgenext_base.in21k_ft_in1k"        18.5M  hybrid, IN21k 预训练
#   "convnext_small.fb_in22k_ft_in1k"    50.0M  hybrid 风格
#
# 优先级: 环境变量 BIRDCLEF_BACKBONE > 文件默认值
# 例如:
#   set BIRDCLEF_BACKBONE=hgnetv2_b3.ssld_stage2_ft_in1k
#   python main.py
#
# HGNetV2 是 PaddlePaddle 团队的高性能 GPU backbone, ssld 是它们的两阶段半监督蒸馏权重,
# 在 BirdCLEF / mel-spec 任务上比同尺寸 EfficientNet 通常更强一点.
BACKBONE_NAME = os.environ.get(
    "BIRDCLEF_BACKBONE", "tf_efficientnet_b0.ns_jft_in1k",
)


# =============================================================================
# 6b. 单次训练子目录 (★ 每次跑 main.py 自动建一个目录, 不覆盖旧产物)
# =============================================================================
# 输出结构:
#   output/
#     ├── hgnetv2_b3_20260520_143022/         ← <backbone_short>_<时间戳>
#     │   ├── fold0_best.pt
#     │   ├── sed_fold0.onnx
#     │   └── ...
#     ├── hgnetv2_b3_20260520_165511_debug/   ← 加 tag 区分 (可选)
#     └── ...
def _backbone_short(name: str) -> str:
    """把 'hgnetv2_b3.ssld_stage2_ft_in1k' 之类的 backbone 名简化:
        hgnetv2_b3.ssld_stage2_ft_in1k    → hgnetv2_b3
        tf_efficientnet_b0.ns_jft_in1k    → efficientnet_b0
    """
    n = name.split(".")[0]                              # 去掉 . 后面
    if n.startswith("tf_"):
        n = n[3:]                                       # 去 'tf_' 前缀
    return n


def get_run_dir(tag: str = "") -> Path:
    """生成本次训练子目录: output/<backbone>_<时间戳>[_<tag>]/."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone_part = _backbone_short(BACKBONE_NAME)
    name = f"{backbone_part}_{ts}"
    if tag:
        name += f"_{tag}"
    rd = OUT_DIR / name
    rd.mkdir(parents=True, exist_ok=True)
    return rd


# =============================================================================
# 7. Perch 蒸馏 (★ 核心 trick)
# =============================================================================
# 思路: 用 Perch v2 当 teacher, 训 backbone 输出 1536-d emb 去 MSE Perch emb.
# 同时 SED head 用 detached backbone feature 训分类 (stop gradient).
# 两个 loss 互不干扰: backbone 学 Perch knowledge, head 学分类.
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM   = 1536                         # Perch v2 emb 维度
ALPHA_DISTILL     = 1.0                          # 蒸馏 loss 权重 (相对 cls_loss)

# ★ Perch teacher emb 离线预缓存 (大幅加速训练):
#   USE_PERCH_CACHE=True 时, 训练读 PERCH_CACHE_DIR 里的 .pt 而不是每 step 跑 ONNX.
#   开启前必须先跑: python precompute_perch_emb.py
#   一致性折中: 每个 focal/sc 文件只缓存中心段 (随机 crop 时 emb 略偏, 影响 < 0.001 AUC)
#
#   False = 每 step 在线跑 Perch ONNX (跟 notebook 100% 一致, 慢 ~40%)
USE_PERCH_CACHE   = False


# =============================================================================
# 8. 训练超参
# =============================================================================
N_FOLDS            = 5                           # 5-fold ensemble
# FOLDS: 这次要跑哪些 fold (其他保持 OOF 缺失). 可被环境变量覆盖.
#   set BIRDCLEF_FOLDS=0,1   →  只跑 fold 0 和 1 (多终端并行场景)
#   不设   →  默认全跑 [0,1,2,3,4]
def _parse_folds_env(default):
    raw = os.environ.get("BIRDCLEF_FOLDS")
    if not raw:
        return default
    parsed = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not parsed:
        return default
    return parsed
FOLDS              = _parse_folds_env([0, 1, 2, 3, 4])
EPOCHS             = int(os.environ.get("BIRDCLEF_EPOCHS", 25))
BATCH              = 16 if DEBUG else int(os.environ.get("BIRDCLEF_BATCH", 64))
LR                 = float(os.environ.get("BIRDCLEF_LR", 5e-4))
MIN_LR             = 1e-5
WD                 = 1e-4                        # weight decay
WARMUP_EPOCHS      = 2                           # warmup 2 epoch (linear) + 之后 cosine

# 早停 (★ 防过拟合 + 节省时间)
#   连续 EARLY_STOP_PATIENCE 个 epoch 主指标 (ns22) 没刷新, 提前结束.
#   设 0 = 关闭早停, 跑满 EPOCHS.
EARLY_STOP_PATIENCE = 8                          # 0 关闭; 8 = 连续 8 epoch 没涨就停

# DataLoader (RTX PRO 6000 + NVMe 配置)
NUM_WORKERS        = 16                          # 0 = 单线程 (Kaggle); 8-16 = 服务器
PIN_MEMORY         = True
PERSISTENT_WORKERS = True                        # epoch 间 worker 不重启

# AMP 精度
#   "fp16" = 跟 notebook 一致 (用 GradScaler, T4/P100 也支持)
#   "bf16" = Blackwell/H100 数值稳定, 不需要 GradScaler (速度差不多, 训练更稳)
AMP_DTYPE          = "fp16"                      # "bf16" | "fp16" | "fp32"

# Soundscape 训练 / CV 策略 (★ 跟 notebook 公开 ONNX 完全一致):
#   - 全 66 个 soundscape 文件 (59 full + 7 partial) 全部进 cache
#   - 5-fold GroupKFold by filename → 每折训练时自动留 ~13 个文件做 OOF 验证
#   - 5 折跑完 train_fold 拼起来 = 全 66 个文件的 OOF 预测 (eval.print_oof_summary)
#   - CV 指标 = OOF macro AUC (non_S22), 跟 LB 通常差 ±0.01


# =============================================================================
# 9. 稀有类上采样
# =============================================================================
# 样本数 < MIN_SAMPLE 的物种, 复制到至少有 MIN_SAMPLE 个样本
MIN_SAMPLE = 20


# =============================================================================
# 10. 波形增强 (training data augmentation, S4 里的 apply_aug)
# =============================================================================
AUG_PROB                = 0.5
AUG_GAIN_DB_RANGE       = (-6.0, 6.0)            # 增益抖动 ±6 dB
AUG_NOISE_SNR_DB_RANGE  = (10.0, 30.0)           # 加噪 SNR 10~30 dB


# =============================================================================
# 11. MixUp (★ 域漂移修复的关键)
# =============================================================================
# Focal-Focal MixUp: 两个 focal clip 叠加, 模拟多物种共存
USE_FOCAL_MIXUP    = True
MIXUP_PROB         = 0.5
MIXUP_ALPHA        = 0.4                         # Beta(α, α) 采样 λ
MIXUP_HARD         = True                        # True = max(l1, l2) 硬 union, False = λ*l1+(1-λ)*l2 软加权

# Focal-Soundscape MixUp: focal + 真实 soundscape 段叠加, 缩小域漂移 (跟测试分布对齐)
USE_FOCAL_SC_MIXUP   = True
FOCAL_SC_MIXUP_PROB  = 0.5
FOCAL_SC_MIXUP_ALPHA = 0.4

# FreqMixStyle (默认关掉, 仅供参考)
FREQ_MIXSTYLE_PROB  = 0.0
FREQ_MIXSTYLE_ALPHA = 0.1


# =============================================================================
# 12. SpecAugment (在 mel 上做的随机 mask)
# =============================================================================
FREQ_MASK_PARAM = 10                              # 单次 mask 最大频率宽度
TIME_MASK_PARAM = 10                              # 单次 mask 最大时间宽度
NUM_FREQ_MASKS  = 1                               # 每张 mel 加几条 freq mask
NUM_TIME_MASKS  = 2                               # 每张 mel 加几条 time mask


# =============================================================================
# 13. 多源 batch 组成 (focal + labeled soundscape)
# =============================================================================
USE_FOCAL           = True                        # 启用 focal 训练数据
USE_FOCAL_SECONDARY = True                        # 用 train.csv 的 secondary_labels (多标签)
USE_LABELED_SC      = True                        # 用 labeled soundscape 当训练数据

# Batch 内组成比例 (focal 9 : sc 1)
ACTIVE_SOURCES = ["focal", "sc"]
SHARES = {"focal": 0.9, "sc": 0.1}

# 不同数据源的 sample-level loss 权重
SOURCE_WEIGHTS = {
    "focal":         1.0,
    "focal_missing": 0.0,                         # cache 缺失文件 → 不算 loss
    "sc":            1.0,
}


# =============================================================================
# 14. 推理参数 (仅 inference 用)
# =============================================================================
# ★ 注意: 训练时用 256 mels, 推理时 ONNX 导出用 128 mels
#         (这是 v1/v2 优化, ONNX 输入小一半, 推理快)
INF_N_MELS    = 256                               # 推理 mel bins
INF_N_FFT     = 2048
INF_HOP       = 512
INF_FMIN      = 20
INF_FMAX      = 16000
INF_TOP_DB    = 80
INF_SR        = 32000
INF_CHUNK_S   = 5
INF_CHUNK_N   = INF_SR * INF_CHUNK_S              # 160000
INF_N_FRAMES  = INF_CHUNK_N // INF_HOP + 1        # 313

N_WINDOWS     = 12                                # 60s / 5s = 12 个推理段
