# -*- coding: utf-8 -*-
"""setup_env.py — wheel install + seed + ONNX/TF availability for Model_5.

对应 cell_07 第 78-119 行 (autodetect wheels + global seed = 4).
"""
from __future__ import annotations

import os
import random
import subprocess
import sys

import numpy as np

from .config import INPUT_ROOT


def _find_optional_wheel(pattern: str):
    hits = sorted(INPUT_ROOT.rglob(pattern)) if INPUT_ROOT.exists() else []
    return hits[0] if hits else None


def _install_optional_wheel(pattern: str) -> bool:
    whl = _find_optional_wheel(pattern)
    if whl is None:
        return False
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(whl)],
        check=True,
    )
    return True


def detect_onnx() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        if _install_optional_wheel("onnxruntime-1.24.4-*.whl"):
            try:
                import onnxruntime  # noqa: F401
                return True
            except ImportError:
                return False
        return False


def ensure_tensorflow() -> bool:
    """尝试让 TensorFlow 可用 (仅 Perch TF SavedModel 后端需要).

    返回 True 表示 TF 可导入. **不会抛异常**: 如果既没装 TF 也没有本地 wheel,
    只打印一条提示并返回 False —— 此时应走 Perch ONNX 后端 (CPU 提交的默认路径).
    """
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        pass
    # 本地有离线 wheel 就装 (Kaggle no-internet 场景)
    _install_optional_wheel("tensorboard-2.20.0-*.whl")
    if _install_optional_wheel("tensorflow-2.20.0-*.whl"):
        try:
            import tensorflow  # noqa: F401
            return True
        except ImportError:
            pass
    print("[setup_env] TensorFlow unavailable — falling back to Perch ONNX backend "
          "(install tensorflow only if you need the TF SavedModel teacher).")
    return False


def seed_everything(seed: int = 4) -> None:
    """全局 seed (cell_07 默认 seed=4, 跟 notebook 行为一致)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bootstrap(seed: int = 4) -> bool:
    """``main`` 在最开始调一次, 返回 ONNX 是否可用."""
    onnx_ok = detect_onnx()
    tf_ok = ensure_tensorflow()
    seed_everything(seed)
    print(f"[setup_env] seed={seed}  onnxruntime_available={onnx_ok}  tensorflow_available={tf_ok}")
    return onnx_ok
