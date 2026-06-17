# -*- coding: utf-8 -*-
"""config.py — Model_5 全局配置 (Karnakbayev_PowerOptimization_LB0948).

完全对齐 ``bird_0.952_kaggle.ipynb`` cell_07 第 ~118-211 行 ``CFG`` 字典 + 第
46-58 行 ``solut`` 字典 + 第 5-19 行 exp019 power=0.6 设定.

所有路径默认指向 Kaggle ``/kaggle/input/...``, 也支持环境变量本地覆盖.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


# =============================================================================
# 路径
# =============================================================================
BASE = Path(os.environ.get(
    "BIRDCLEF_BASE",
    "/kaggle/input/competitions/birdclef-2026",
))

MODEL_DIR = Path(os.environ.get(
    "BIRDCLEF_MODEL_DIR",
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
))

INPUT_ROOT = Path(os.environ.get("BIRDCLEF_INPUT_ROOT", "/kaggle/input"))

WORK_DIR = Path(os.environ.get("BIRDCLEF_WORK_DIR", "/kaggle/working/cache"))
WORK_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 音频常量
# =============================================================================
SR             = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES   = 60 * SR
N_WINDOWS      = 12

DEVICE = "cpu"


# =============================================================================
# MODE
# =============================================================================
MODE = os.environ.get("BIRDCLEF_MODE", "submit")
assert MODE in {"train", "submit"}


# =============================================================================
# 内部 ensemble 配置 (cell_07 第 46-58 行)
# =============================================================================
SOLUT = {
    "type_add": "direct",
    "Models": [
        {
            "Model":  "Karnakbayev_PowerOptimization_LB0948",
            "subm":   "subm_karnakbayev_power_optimization.csv",
            "weight": 1.0,
            "xSED":   [0.60, 0.40],
            "LB":     "0.948",
        },
    ],
}

# Model_5 最终输出文件名 (跟 cell_02 solutions.Models[1].subm 对齐)
OUTPUT_SUBM = os.environ.get("BIRDCLEF_MODEL5_SUBM", "subm_5.csv")
INTERMEDIATE_SUBM = SOLUT["Models"][0]["subm"]   # subm_karnakbayev_power_optimization.csv


# =============================================================================
# HGNet 附加分支 (默认关闭). 复用 hgnet_branch.py 的 LSEModel,
# 在 subm_5.csv 之外再叠一刀 rank blend, HGNet 默认占 8%.
# =============================================================================
USE_HGNET = os.environ.get("BIRDCLEF_M5_USE_HGNET", "0") not in {"0", "false", "False"}

# HGNet backbone (timm 名称). 默认 hgnetv2_b4 (跟 hgnetv2_b4_pt 4-fold 权重对齐).
# 想换 b0/b2/repvit 等就把环境变量 BIRDCLEF_HGNET_BACKBONE 改成对应名字.
HGNET_BACKBONE = os.environ.get(
    "BIRDCLEF_HGNET_BACKBONE", "hgnetv2_b4.ssld_stage2_ft_in1k",
)

# best_model_fold0..3.pt 所在目录, Kaggle 上挂 dataset 后用环境变量
# BIRDCLEF_HGNET_DIR 覆盖即可.
HGNET_CKPT_DIR = os.environ.get(
    "BIRDCLEF_HGNET_DIR",
    "/kaggle/input/datasets/czastu/hgnetv2-b4-fold-pt",
)

# HGNet 在最终 blend 中的权重 (跟主流水线 rank-space 加权).
# subm_5.csv 占 1 - HGNET_W, HGNet 占 HGNET_W. 默认 0.20 = 20%.
# (b0 4% + b1 4% + b4 12% = 20%, 配合下面 HGNET_ENSEMBLE_WEIGHTS=[1,1,3])
HGNET_W = float(os.environ.get("BIRDCLEF_HGNET_W", "0.20"))

# hgnet_branch.py 所在目录 (Kaggle 上推荐放进 scheme-model dataset 顶部,
# 默认按工程相对路径找; 找不到就 fallback 到 INPUT_ROOT 下 rglob).
HGNET_BRANCH_PATH = os.environ.get("BIRDCLEF_HGNET_BRANCH_PATH", "")


# =============================================================================
# HGNet 附加分支的多 backbone ONNX ensemble (dian_pt 系列).
#
# 每个目录含 best_model_fold{0..3}.onnx (统一的 LSEModel 导出, 输入
# (B,1,256,256) → 输出 (B,234)). mel 只算一次喂给所有 session.
#
# 设了 HGNET_ENSEMBLE_DIRS (非空) 时, hgnet_addon 走多 backbone ensemble;
# 留空则退回单 backbone (HGNET_CKPT_DIR + HGNET_BACKBONE) 的老逻辑.
#
# 默认留空 → 退回单 backbone (v20, HGNET_CKPT_DIR + HGNET_BACKBONE).
# 要跑 v22 三 backbone ensemble, 显式设 BIRDCLEF_HGNET_ENSEMBLE_DIRS
# (os.pathsep 分隔: Windows ';' / Linux ':'), 例如 b0 / b1 / b4(weak) 三个 onnx 目录:
#   set BIRDCLEF_HGNET_ENSEMBLE_DIRS=<b0_onnx>;<b1_onnx>;<b4_weak_onnx>
# 配合 HGNET_W=0.20 + HGNET_ENSEMBLE_WEIGHTS=1,1,3:
#   b0 = 20% * 0.2 = 4%,  b1 = 20% * 0.2 = 4%,  b4 = 20% * 0.6 = 12%.
# =============================================================================
_DEFAULT_HGNET_ENSEMBLE_DIRS = ""

HGNET_ENSEMBLE_DIRS = [
    d for d in os.environ.get(
        "BIRDCLEF_HGNET_ENSEMBLE_DIRS", _DEFAULT_HGNET_ENSEMBLE_DIRS,
    ).split(os.pathsep) if d.strip()
]

# 跨 backbone 权重 (逗号分隔, 长度需等于 HGNET_ENSEMBLE_DIRS). 留空=等权.
# 默认 [1, 1, 3] → 归一化 0.2 / 0.2 / 0.6, 对应 b0:b1:b4 = 4%:4%:12%.
HGNET_ENSEMBLE_WEIGHTS = [
    float(w) for w in os.environ.get(
        "BIRDCLEF_HGNET_ENSEMBLE_WEIGHTS", "1,1,3",
    ).split(",") if w.strip()
] or None


# =============================================================================
# CFG (cell_07 第 170-211 行)
# =============================================================================
CFG = {
    "batch_files":  16,
    "oof_n_splits": 5  if MODE == "train" else 3,
    "dryrun_n_files": 20 if MODE == "train" else 0,
    "run_oof":   MODE == "train",
    "verbose":   MODE == "train",

    "proto_ssm_train": {
        "n_epochs":         80  if MODE == "train" else 40,
        "lr":               8e-4,
        "weight_decay":     1e-3,
        "val_ratio":        0.15,
        "patience":         20  if MODE == "train" else 8,
        "pos_weight_cap":   25.0,
        "distill_weight":   0.15,
        "proto_margin":     0.15,
        "label_smoothing":  0.03,
        "oof_n_splits":     5  if MODE == "train" else 3,
        "mixup_alpha":      0.4,
        "focal_gamma":      2.5,
        "swa_start_frac":   0.65,
        "swa_lr":           4e-4,
        "use_cosine_restart": True,
        "restart_period":   20,
    },

    "residual_ssm": {
        "d_model":          128,
        "d_state":          16,
        "n_ssm_layers":     2,
        "dropout":          0.1,
        "correction_weight": 0.35,
        "n_epochs":         40 if MODE == "train" else 20,
        "lr":               8e-4,
        "patience":         12 if MODE == "train" else 6,
    },

    "mlp_params": {
        "hidden_layer_sizes":  (256, 128),
        "activation":          "relu",
        "max_iter":            500 if MODE == "train" else 200,
        "early_stopping":      True,
        "validation_fraction": 0.15,
        "n_iter_no_change":    20 if MODE == "train" else 10,
        "random_state":        42,
        "learning_rate_init":  5e-4,
        "alpha":               0.005,
    },

    # exp019: rank_aware_scaling power=0.5 → 0.6
    "rank_aware_power": 0.6,

    # exp017: apply_prior lambda_prior 0.4 → 0.5
    "lambda_prior": 0.5,

    # 第二阶段权重
    "ensemble_w_per_class_mapped":   0.60,
    "ensemble_w_per_class_unmapped": 0.35,

    # 后处理
    "file_conf_top_k":       2,
    "file_conf_power":       0.4,
    "delta_alpha":           0.20,
    "tta_shifts":            [0, 1, -1, 2, -2],
    "correction_grid":       [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
}


# =============================================================================
# Distilled-SED 参数 (cell_07 第 1326-1334 行)
# =============================================================================
SED_CFG = {
    "n_mels":  256,
    "n_fft":   2048,
    "hop":     512,
    "fmin":    20,
    "fmax":    16000,
    "top_db":  80,
    "smooth_sigma": 0.65,
}


# =============================================================================
# 计时
# =============================================================================
WALL_START = time.time()


def elapsed_min() -> float:
    return (time.time() - WALL_START) / 60.0
