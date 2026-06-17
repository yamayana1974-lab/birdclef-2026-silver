# -*- coding: utf-8 -*-
"""
convert_perch_onnx.py — 把本地 Perch v2 SavedModel 转成 ONNX
==============================================================

输入 SavedModel 签名(已验证):
    inputs ['inputs']:    (B, 160000)  float32   ← 5s 波形
    outputs['embedding']: (B, 1536)    float32   ← 蒸馏目标

只导出 embedding 这个输出,体积更小、推理更快.

依赖:
    pip install tf2onnx tensorflow onnx onnxruntime
"""
from pathlib import Path
import os
import subprocess
import sys

# 路径: 默认 ./data, 用环境变量 BIRDCLEF_SED_ROOT 覆盖根目录.
_ROOT           = Path(os.environ.get("BIRDCLEF_SED_ROOT", "./data"))
SAVED_MODEL_DIR = Path(os.environ.get("PERCH_SAVED_MODEL", str(_ROOT / "perch_v2_model")))
OUT_ONNX        = Path(os.environ.get("PERCH_OUT_ONNX", str(_ROOT / "perch_v2.onnx")))


def main():
    OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", str(SAVED_MODEL_DIR),
        "--output",      str(OUT_ONNX),
        "--opset",       "17",
        # 只保留 embedding,丢掉 label/spatial_embedding/spectrogram
        "--rename-outputs", "embedding",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # 验证
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(str(OUT_ONNX), providers=["CPUExecutionProvider"])
    print("\nONNX inputs :", [(i.name, i.shape) for i in sess.get_inputs()])
    print("ONNX outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
    out = sess.run(None, {sess.get_inputs()[0].name:
                          np.zeros((1, 160000), dtype=np.float32)})
    embed = next(o for o in out if o.shape[-1] == 1536)
    print(f"Test forward OK: embedding shape = {embed.shape}")
    print(f"\n✓ Saved to {OUT_ONNX}")


if __name__ == "__main__":
    main()
