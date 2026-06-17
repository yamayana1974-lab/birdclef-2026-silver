# -*- coding: utf-8 -*-
"""
check_backbones.py — 快速验证 backbone 能否跑通本套 SED pipeline
=================================================================

不碰 waveform cache / 训练数据, 只验证:
  1. timm.create_model(name, in_chans=1, num_classes=0, global_pool="") 能建
  2. BirdSEDModel 能 forward, 拿到 backbone_dim (训练路径)
  3. SEDExportWrapper 能 forward (ONNX 导出路径)

用法 (在 distilled_sed_model_0524 目录):
    python check_backbones.py

通过的 backbone 就可以直接:
    python main.py --mode train --backbone <name> --debug    # 先 1 fold 1 epoch 验证全流程
    python main.py --mode train --backbone <name>            # 正式 5 fold
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import timm

from models import BirdSEDModel
from export_onnx import SEDExportWrapper
from config import (N_MELS, INF_N_MELS, INF_N_FRAMES, HOP_LENGTH,
                    TRAIN_SAMPLES, NUM_CLASSES)


BACKBONES = [
    "hgnetv2_b3.ssld_stage2_ft_in1k",
    "tf_efficientnetv2_b2.in1k",
    "convnextv2_pico.fcmae_ft_in1k",
]


def check_one(name):
    print("=" * 64)
    print(f"backbone: {name}")
    try:
        # ── 1. 训练路径: BirdSEDModel forward (CPU, 快速) ──
        m = BirdSEDModel(name)          # __init__ 会 print backbone 输出 shape + C
        m.eval()
        n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
        mel  = torch.randn(1, 1, N_MELS, n_tf)
        with torch.no_grad():
            clip, fw, dist = m(mel, return_framewise=True, return_distill=True)
        print(f"  [train ] clip={tuple(clip.shape)} "
              f"framewise={tuple(fw.shape)} distill={tuple(dist.shape)}")

        # ── 2. 导出路径: SEDExportWrapper forward (推理 mel 尺寸) ──
        w = SEDExportWrapper(name, NUM_CLASSES, m.backbone_dim)
        w.eval()
        infer_mel = torch.randn(1, 1, INF_N_MELS, INF_N_FRAMES)
        with torch.no_grad():
            c2, f2 = w(infer_mel)
        print(f"  [export] clip={tuple(c2.shape)} framewise={tuple(f2.shape)}")

        print(f"  ✓ PASS  (backbone_dim={m.backbone_dim})")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {type(e).__name__}: {e}")
        # 找相近的可用名字, 方便改正
        stem = name.split(".")[0]
        cands = timm.list_models(f"*{stem}*", pretrained=True)
        if cands:
            print(f"    相近可用名字: {cands[:8]}")
        return False


def main():
    print(f"timm version: {timm.__version__}")
    results = {name: check_one(name) for name in BACKBONES}
    print("=" * 64)
    print("SUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()
