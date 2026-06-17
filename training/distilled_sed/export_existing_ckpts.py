# -*- coding: utf-8 -*-
"""
export_existing_ckpts.py — 把已有的 fold*_best.pt 单独导出成 ONNX
================================================================

用途: 训练已经存了 .pt 但 ONNX 导出失败 (比如 onnxscript 缺失 / 阈值过严),
直接对已有 ckpt 重新跑 ONNX 导出, 不用重训.

用法:
    python export_existing_ckpts.py <run_dir>

示例:
    python export_existing_ckpts.py output/efficientnet_b0_20260520_191336
    # 会扫该目录下所有 fold*_best.pt, 导出对应 sed_fold*.onnx
"""
import sys
from pathlib import Path

import torch

from config import device
from models import make_model
from export_onnx import export_fold_to_onnx


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"[ERROR] run_dir not found: {run_dir}")
        sys.exit(1)
    print(f"run_dir = {run_dir}")

    ckpts = sorted(run_dir.glob("fold*_best.pt"))
    if not ckpts:
        print(f"[ERROR] no fold*_best.pt found in {run_dir}")
        sys.exit(1)
    print(f"found {len(ckpts)} ckpts: {[c.name for c in ckpts]}")

    for ckpt_path in ckpts:
        # 文件名 fold{k}_best.pt → 抠出 k
        fold_k = int(ckpt_path.stem.replace("fold", "").replace("_best", ""))
        onnx_path = run_dir / f"sed_fold{fold_k}.onnx"
        if onnx_path.exists():
            print(f"\n  [skip] fold {fold_k}: {onnx_path.name} already exists")
            continue
        print(f"\n  → fold {fold_k}: loading {ckpt_path.name}")
        state = torch.load(ckpt_path, map_location="cpu")

        # 构造模型拿 backbone_dim 然后导出
        m = make_model()
        m.load_state_dict(state, strict=False)
        export_fold_to_onnx(
            fold_k        = fold_k,
            trained_state = state,
            backbone_dim  = m.backbone_dim,
            onnx_path     = onnx_path,
            verify        = True,
        )
        del m

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
