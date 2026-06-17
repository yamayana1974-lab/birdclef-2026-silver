# -*- coding: utf-8 -*-
"""
main.py — Tucker distilled SED 完整 pipeline 入口
====================================================

把 7 个模块串起来跑完整 train / infer 流程.

CLI 用法 (推荐, 不用改 config.py):

  # 训练 efficientnet b0, 全 5 fold
  python main.py --mode train --backbone tf_efficientnet_b0.ns_jft_in1k

  # 训练 hgnetv2_b3, 只跑 fold 0,1 (多终端并行场景)
  python main.py --mode train --backbone hgnetv2_b3.ssld_stage2_ft_in1k --folds 0,1

  # 训练 edgenext_base, 单卡 96GB 上 batch 调大
  python main.py --mode train --backbone edgenext_base.in21k_ft_in1k --batch 128 --epochs 30

  # 续训某个目录 (不重新建 RUN_DIR)
  python main.py --mode train --run-dir output/hgnetv2_b3_20260524_120000

  # 推理: 默认从最新 RUN_DIR 加载 ONNX
  python main.py --mode infer


环境变量也支持 (跟 CLI 等价, 优先级低于 CLI):
  BIRDCLEF_MODE / BIRDCLEF_BACKBONE / BIRDCLEF_FOLDS / BIRDCLEF_EPOCHS / BIRDCLEF_BATCH / BIRDCLEF_LR
  RUN_DIR (续训用)


对应 notebook 的 cell 顺序:
  S0/S1 → config.seed_everything
  S2     → data.load_data
  S3     → models.* (lazy import 在 train.py 里)
  S4     → dataset.* (同上)
  S5/S6  → train.train_fold + export_onnx.export_fold_to_onnx
  S7     → eval.print_oof_summary
  Inference cells → inference.run_inference + write_submission
"""
import argparse
import os
import sys


def _parse_args():
    p = argparse.ArgumentParser(description="distilled-SED train/infer entrypoint")
    p.add_argument("--mode", choices=["train", "infer"], default=None,
                   help="train or infer; defaults to BIRDCLEF_MODE / config.MODE")
    p.add_argument("--backbone", default=None,
                   help="timm backbone name, e.g. hgnetv2_b3.ssld_stage2_ft_in1k")
    p.add_argument("--folds", default=None,
                   help="comma-separated folds, e.g. '0,1' or '0'; default = all 5")
    p.add_argument("--epochs", type=int, default=None,
                   help="override EPOCHS")
    p.add_argument("--batch", type=int, default=None,
                   help="override BATCH")
    p.add_argument("--lr", type=float, default=None,
                   help="override LR")
    p.add_argument("--run-dir", default=None,
                   help="reuse an existing run dir (resume); default = new timestamped dir")
    p.add_argument("--debug", action="store_true",
                   help="DEBUG=True (epoch=1, fold=[0])")
    return p.parse_args()


def _apply_cli_overrides(args):
    """把 CLI 参数转成环境变量, 必须在 import config 之前调用."""
    if args.mode:     os.environ["BIRDCLEF_MODE"]     = args.mode
    if args.backbone: os.environ["BIRDCLEF_BACKBONE"] = args.backbone
    if args.folds:    os.environ["BIRDCLEF_FOLDS"]    = args.folds
    if args.epochs is not None: os.environ["BIRDCLEF_EPOCHS"] = str(args.epochs)
    if args.batch  is not None: os.environ["BIRDCLEF_BATCH"]  = str(args.batch)
    if args.lr     is not None: os.environ["BIRDCLEF_LR"]     = str(args.lr)
    if args.run_dir:  os.environ["RUN_DIR"]                   = args.run_dir
    if args.debug:    os.environ["BIRDCLEF_DEBUG"]            = "1"


# ★ argparse 必须在 import config 之前生效, 否则 config.py 里读 os.environ 时拿不到
_ARGS = _parse_args() if __name__ == "__main__" else None
if _ARGS is not None:
    _apply_cli_overrides(_ARGS)


import numpy as np
import torch
from pathlib import Path

from config import (MODE, FOLDS, N_FOLDS, OUT_DIR, NUM_CLASSES, BACKBONE_NAME,
                    EPOCHS, BATCH, LR,
                    USE_PERCH_DISTILL, DEBUG, seed_everything, get_run_dir)


def main():
    seed_everything(42)
    print(f"[config] MODE={MODE}  BACKBONE={BACKBONE_NAME}")
    print(f"[config] FOLDS={FOLDS}  EPOCHS={EPOCHS}  BATCH={BATCH}  LR={LR}")
    print(f"[config] USE_PERCH_DISTILL={USE_PERCH_DISTILL}  DEBUG={DEBUG}")

    # ★ 每次跑都建一个 <backbone>_<时间戳> 子目录, 不覆盖旧产物.
    #   也可以通过 RUN_DIR 环境变量 / --run-dir 指定已有目录, 续训用.
    forced = os.environ.get("RUN_DIR")
    if forced:
        run_dir = Path(forced).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CONTINUE] RUN_DIR = {run_dir} (from env)")
    else:
        run_dir = get_run_dir(tag="debug" if DEBUG else "")
        print(f"RUN_DIR = {run_dir}")

    if MODE == "train":
        # =========================================================
        # ── 训练模式 ──
        # =========================================================
        from data import load_data
        from train import train_fold, _load_val_waveforms, _predict_from_waveforms
        from models import MelSpecTransform, make_model
        from export_onnx import export_fold_to_onnx
        from eval import print_oof_summary

        # ── Step 1. 加载数据 ─────────────────────────────────────
        print("\n[Step 1] Loading data...")
        data_bundle = load_data()
        sc_cache_meta   = data_bundle["sc_cache_meta"]
        Y_SC            = data_bundle["Y_SC"]
        non_s22_mask_sc = data_bundle["non_s22_mask_sc"]
        TAXON_MASKS     = data_bundle["TAXON_MASKS"]
        PRIMARY_LABELS  = data_bundle["PRIMARY_LABELS"]

        # ── Step 2. Fold loop ────────────────────────────────────
        oof_preds = np.full((len(sc_cache_meta), NUM_CLASSES),
                             np.nan, dtype=np.float32)
        all_hist  = {}

        for fold_k in FOLDS:
            print(f"\n{'='*60}\nFOLD {fold_k}\n{'='*60}")
            vm = sc_cache_meta["fold"].values == fold_k
            val_sc_df_k = sc_cache_meta[vm].reset_index(drop=True)

            # ── 训一个 fold ──
            best_ns22_state, best_macro_state, hist = train_fold(fold_k, data_bundle)
            all_hist[fold_k] = hist

            if best_macro_state is not None:
                # ── 1. 存 PyTorch ckpt 到 RUN_DIR ──
                ckpt_path = run_dir / f"fold{fold_k}_best.pt"
                torch.save(best_macro_state, ckpt_path)
                print(f"  Saved {ckpt_path}")

                # ── 2. OOF 推理 (用 best_macro_state) ──
                m = make_model()
                m.load_state_dict(best_macro_state, strict=False)
                mel_tf = MelSpecTransform().to(m.parameters().__next__().device)
                val_wavs_k = _load_val_waveforms(val_sc_df_k)
                oof_preds[vm] = _predict_from_waveforms(m, mel_tf, val_wavs_k)["blend"]

                # ── 3. ONNX 导出到 RUN_DIR ──
                export_fold_to_onnx(
                    fold_k        = fold_k,
                    trained_state = best_macro_state,
                    backbone_dim  = m.backbone_dim,
                    onnx_path     = run_dir / f"sed_fold{fold_k}.onnx",
                    verify        = True,
                )
                del m

        # ── Step 3. OOF 汇总 ─────────────────────────────────────
        print_oof_summary(oof_preds, Y_SC, non_s22_mask_sc, TAXON_MASKS,
                          all_hist, N_FOLDS, sc_cache_meta)

    elif MODE == "infer":
        # =========================================================
        # ── 推理模式 ──
        # =========================================================
        from inference import (load_sed_sessions, find_test_files,
                                run_inference, write_submission)

        # 这里需要 PRIMARY_LABELS, 但不要全跑 load_data (慢, 还要 cache).
        # 直接从 sample_submission 拿
        import pandas as pd
        from config import SAMPLE_SUB_PATH
        sample_sub     = pd.read_csv(SAMPLE_SUB_PATH)
        PRIMARY_LABELS = sample_sub.columns[1:].tolist()

        # ── Step 1. 加载 5 个 ONNX session ───────────────────────
        print("\n[Step 1] Loading ONNX sessions...")
        fold_sessions, fold_ids, sed_dir = load_sed_sessions()

        # ── Step 2. 找 test 文件 ─────────────────────────────────
        print("\n[Step 2] Finding test files...")
        test_files = find_test_files(debug_n=5)

        # ── Step 3. 推理 ────────────────────────────────────────
        print("\n[Step 3] Running inference...")
        all_rows, all_preds_arr = run_inference(test_files, fold_sessions, fold_ids)

        # ── Step 4. 写 submission ───────────────────────────────
        print("\n[Step 4] Writing submission.csv...")
        write_submission(all_rows, all_preds_arr, PRIMARY_LABELS,
                          out_csv=run_dir / "submission.csv")

    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
