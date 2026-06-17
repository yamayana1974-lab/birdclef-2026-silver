# -*- coding: utf-8 -*-
"""hgnet_addon.py — 把 ``hgnet_branch.py`` 当外部 utility 使,
在 Model_5 已经产出 ``subm_5.csv`` 之后再叠一刀 HGNet rank-blend.

设计:
  * 不重新实现 HGNet 推理 / 融合, 完全复用现成的 ``hgnet_branch.py``.
  * 把 ``hgnet_branch`` 模块用 ``importlib.util`` 按文件路径加载, 这样无论
    它打包成什么 dataset 名, 都能找到.
  * Kaggle 默认查找路径:
      1. ``BIRDCLEF_HGNET_BRANCH_PATH`` (env var)
      2. ``bird_0.952_project/hgnet_branch.py`` (推荐, 自包含)
      3. ``/kaggle/input/**/hgnet_branch.py`` rglob 第一个
      4. 项目内 ``../bird_0.949_project/hgnet_branch.py`` (本地调试)

不调本模块时 (USE_HGNET=False), 不会 import HGNet 依赖 (timm / torchvision /
torchaudio), 失败也不影响主流水线.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import (
    HGNET_BACKBONE,
    HGNET_BRANCH_PATH,
    HGNET_CKPT_DIR,
    HGNET_ENSEMBLE_DIRS,
    HGNET_ENSEMBLE_WEIGHTS,
    HGNET_W,
    INPUT_ROOT,
)


def _resolve_hgnet_branch_path() -> Path:
    """优先级:
        1. env var ``BIRDCLEF_HGNET_BRANCH_PATH``
        2. 项目内同级 (``bird_0.952_project/hgnet_branch.py``) — 推荐, 自包含
        3. Kaggle 上 ``/kaggle/input/`` 下 rglob
        4. 兜底: 项目内 ``bird_0.949_project/hgnet_branch.py``
    """
    if HGNET_BRANCH_PATH:
        p = Path(HGNET_BRANCH_PATH)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"BIRDCLEF_HGNET_BRANCH_PATH={HGNET_BRANCH_PATH} 不存在"
        )

    here = Path(__file__).resolve()
    cand_local = here.parent.parent / "hgnet_branch.py"
    if cand_local.exists():
        return cand_local

    if INPUT_ROOT.exists():
        hits = sorted(INPUT_ROOT.rglob("hgnet_branch.py"))
        if hits:
            return hits[0]

    for parent in (here.parent.parent.parent, here.parent.parent.parent.parent):
        cand = parent / "bird_0.949_project" / "hgnet_branch.py"
        if cand.exists():
            return cand

    raise FileNotFoundError(
        "hgnet_branch.py 找不到. 请把它放在 bird_0.952_project/ 顶层, "
        "或设置 BIRDCLEF_HGNET_BRANCH_PATH, "
        "或放进 /kaggle/input/<some_dataset>/hgnet_branch.py."
    )


def _load_hgnet_branch_module():
    path = _resolve_hgnet_branch_path()
    # 把 backbone 名字塞到环境变量, hgnet_branch.py 在 import 时读取
    # ``BIRDCLEF_HGNET_BACKBONE`` 决定 MODEL_NAME (默认 b0, 我们这里默认 b4).
    os.environ["BIRDCLEF_HGNET_BACKBONE"] = HGNET_BACKBONE
    spec = importlib.util.spec_from_file_location("_hgnet_branch_external", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法 import {path} as module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    print(f"[hgnet_addon] loaded hgnet_branch from {path}  "
          f"backbone={getattr(mod, 'MODEL_NAME', '?')}")
    return mod


def append_hgnet_to_submission(
    final_subm_path: str,
    test_paths: List[Path],
    primary_labels: List[str],
    out_path: Optional[str] = None,
) -> Path:
    """跑 HGNet → rank-blend 进 ``final_subm_path``, 默认覆盖原文件.

    后端二选一 (看 config):
      * ``HGNET_ENSEMBLE_DIRS`` 非空 → 多 backbone ONNX ensemble
        (dian_pt 的 efficientnetv2_b0 / hgnetv2_b0 / hgnetv2_b1 等),
        mel 只算一次喂给所有 session;
      * 否则 → 单 backbone (``HGNET_CKPT_DIR`` + ``HGNET_BACKBONE``) 老逻辑.

    Dry-run 容错: 如果 base 跟 hgnet csv 的 row_id 完全不重叠 (Save & Run All
    阶段 hidden test 没挂载, base 走 sample_submission stub, hgnet 走
    train_soundscapes dry-run), ``blend_with_hgnet`` 内部会直接退化成只用
    base 写出, commit 不会崩.

    Args:
        final_subm_path:  Model_5 已经写好的 subm_5.csv 路径.
        test_paths:       跟 sed_pipeline 同一份 (含 dry-run fallback).
        primary_labels:   234 个 label, 用于 sample_submission 列对齐.
        out_path:         默认 None = 覆盖 final_subm_path.

    Returns:
        最终 csv 路径.
    """
    out_path = out_path or final_subm_path
    hgb = _load_hgnet_branch_module()

    hgnet_csv = "submission_hgnet.csv"

    if HGNET_ENSEMBLE_DIRS:
        # ── 多 backbone ONNX ensemble (dian_pt 系列) ──
        print(f"[hgnet_addon] multi-backbone ONNX ensemble  "
              f"base_w={1 - HGNET_W:.4f}  hgnet_w={HGNET_W:.4f}")
        print(f"[hgnet_addon]   dirs={HGNET_ENSEMBLE_DIRS}")
        print(f"[hgnet_addon]   dir_weights={HGNET_ENSEMBLE_WEIGHTS or 'equal'}")
        hgb.run_onnx_ensemble_inference(
            test_paths     = test_paths,
            onnx_dirs      = HGNET_ENSEMBLE_DIRS,
            primary_labels = list(primary_labels),
            out_csv        = hgnet_csv,
            dir_weights    = HGNET_ENSEMBLE_WEIGHTS,
        )
    else:
        # ── 单 backbone (HGNET_CKPT_DIR + HGNET_BACKBONE) 老逻辑 ──
        print(f"[hgnet_addon] base={final_subm_path}  weight base={1 - HGNET_W:.4f} "
              f"hgnet={HGNET_W:.4f}  backbone={HGNET_BACKBONE}  "
              f"ckpt_dir={HGNET_CKPT_DIR}")
        hgb.run_hgnet_inference(
            test_paths     = test_paths,
            ckpt_dir       = HGNET_CKPT_DIR,
            primary_labels = list(primary_labels),
            out_csv        = hgnet_csv,
        )

    hgb.blend_with_hgnet(
        base_csv  = final_subm_path,
        hgnet_csv = hgnet_csv,
        out_csv   = out_path,
        w_base    = 1.0 - HGNET_W,
        w_hgnet   = HGNET_W,
        mode      = "rank",
    )
    return Path(out_path)
