r"""训练入口
==============

包含:

- ``train_one_fold`` : 训练并保存单折模型权重 / OOF 预测 / 训练曲线 csv.
- ``main``           : 完整 pipeline (4 折训练 + OOF AUC 计算 + 保存 log_mel jb).

使用:

    python -m hgnet.train

或者:

    cd /path/to/birdclef-2026-repo/training/hgnet
    python train.py

训练产物默认落到 ``cwd``:
    best_model_fold{0..3}.pt
    best_val_pred_fold{0..3}.npy
    result_df_fold{0..3}.csv
    log_mel_spectrogram.jb

后续用 ``export.py`` 把 .pt 转 OpenVINO .xml / .bin.
"""

from __future__ import annotations

import typing as tp
from pathlib import Path
from time import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import trange

from .config import CFG, N_CLASSES, N_FOLDS, RANDOM_SEED, make_run_dir
from .data import (
    MultiLabelStratifiedGroupKFold,
    build_train_dataframe,
    load_label_files,
    split_train_soundscapes,
)
from .dataset import get_data_loader
from .models import LSEModel
from .transforms import LogMelSpectrogramTransform, MixUp, dummy_mixup
from .utils import device, rank_normalize, set_random_seed, to_device


# ============================================================
#                  1. 单折训练
# ============================================================
def train_one_fold(
    train_df: pd.DataFrame,
    fold_id: int,
    device: torch.device,
    classes: tp.List[str],
    output_dir: tp.Optional[Path] = None,
) -> None:
    """训练并保存某一折模型 / OOF 预测 / 训练曲线 csv."""
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir.mkdir(exist_ok=True)

    n_train = len(train_df.query("fold != @fold_id"))
    n_valid = len(train_df.query("fold == @fold_id"))
    print(f"[training fold {fold_id}]")
    print(f"train: {n_train}, valid: {n_valid}")

    set_random_seed(RANDOM_SEED)
    trn_loader, val_loader = get_data_loader(train_df, fold_id, classes=classes, cfg=CFG)

    # log-mel 模块放在 GPU, 与训练 / 推理同 device.
    lms_transform = LogMelSpectrogramTransform(
        CFG.mel_spectrogram_params, CFG.top_db, CFG.lms_shape,
    ).eval().to(device)
    mixup = MixUp(**CFG.mixup)

    model = LSEModel(
        CFG.model_name, CFG.pretrained,
        drop_path_rate=CFG.drop_path_rate,
        num_classes=N_CLASSES,
        head_dropout=CFG.head_dropout,
        lse_temperature=CFG.lse_temperature,
    ).to(device)

    num_devices = torch.cuda.device_count()
    if CFG.use_dp and num_devices > 1:
        model = nn.DataParallel(model)

    optimizer = AdamW(params=model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    optimizer.zero_grad()
    scheduler = OneCycleLR(
        optimizer=optimizer,
        epochs=CFG.max_epoch,
        pct_start=CFG.warmup_epoch / CFG.max_epoch,
        max_lr=CFG.lr,
        div_factor=25,
        final_div_factor=4.0,
        steps_per_epoch=len(trn_loader),
    )
    loss_func = nn.BCEWithLogitsLoss()
    # AMP dtype: fp16 (默认, 需要 GradScaler) 或 bf16 (数值更稳, ConvNeXt V2 / GRN 必选).
    _amp_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    amp_dtype = _amp_map.get(getattr(CFG, "amp_dtype", "fp16"), torch.float16)
    use_gradscaler = CFG.use_amp and amp_dtype == torch.float16
    grad_scaler = torch.GradScaler(enabled=use_gradscaler)
    print(f"[amp] dtype={amp_dtype}, gradscaler={use_gradscaler}")

    result_list: tp.List[tp.List[float]] = []
    best_val_score = 0.0
    best_state: tp.Optional[tp.Dict[str, torch.Tensor]] = None
    best_val_pred: tp.Optional[np.ndarray] = None

    for epoch in trange(CFG.max_epoch):
        epoch_start = time()
        trn_loss = 0.0

        # 前 warmup_epoch 个 epoch 关 mixup, 之后开启.
        tmp_mixup = mixup if epoch >= CFG.warmup_epoch else dummy_mixup

        # ---------- train ----------
        model.train()
        for batch in trn_loader:
            batch = to_device(batch, device, non_blocking=True)
            wave, label = batch["wave"], batch["label"]

            lms = lms_transform(wave)
            lms, label = tmp_mixup(lms, label)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=CFG.use_amp):
                logits = model(lms.detach())
                loss = loss_func(logits, label)

            if use_gradscaler:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            trn_loss += loss.item()
            del batch, wave, label, lms, logits, loss

        trn_loss /= len(trn_loader)

        # ---------- valid ----------
        model.eval()
        logit_list, label_list = [], []
        # 原 notebook 写的 `with torch.no_grad() and torch.autocast(...)` 是 `and`
        # 短路求值, no_grad 不会真正生效. 这里修正成正确的多上下文写法.
        for batch in val_loader:
            lms = lms_transform(to_device(batch["wave"], device, non_blocking=True))
            with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=CFG.use_amp):
                logits = model(lms.detach())
            logit_list.append(logits.detach().cpu())
            label_list.append(batch["label"].detach())
            del batch, lms, logits

        logits = torch.cat(logit_list, axis=0)
        labels = torch.cat(label_list, axis=0)
        val_loss = F.binary_cross_entropy_with_logits(logits, labels).item()

        logits = logits.numpy()
        labels = labels.numpy()

        # 只对验证集中至少出现过一次的类计算 macro-AUC, 避免全零列报错.
        mask = labels.sum(axis=0) > 0
        val_score = roc_auc_score(labels[:, mask], logits[:, mask], average="macro")

        # ---------- 记录 best ----------
        if val_score > best_val_score:
            best_val_score = val_score
            module = model.module if (CFG.use_dp and num_devices > 1) else model
            best_state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
            best_val_pred = logits

        epoch_end = time()
        result_list.append([
            epoch, scheduler.get_last_lr()[0], trn_loss, val_loss, val_score, epoch_end - epoch_start,
        ])
        print(
            "[epoch {}] lr={:.6f}, trn_loss={:.5f}, val_loss={:.5f}, val_score={:.5f}. elapsed={:.2f}".format(
                *result_list[-1]
            )
        )

    # ---------- 落盘 ----------
    result_df = pd.DataFrame(
        result_list,
        columns=["epoch", "lr", "trn_loss", "val_loss", "val_score", "elapsed_time"],
    )
    result_df.to_csv(output_dir / f"result_df_fold{fold_id}.csv", index=False)
    torch.save(best_state, output_dir / f"best_model_fold{fold_id}.pt")
    np.save(output_dir / f"best_val_pred_fold{fold_id}.npy", best_val_pred)


# ============================================================
#                  2. 主入口
# ============================================================
def main(fold_ids: tp.Optional[tp.List[int]] = None,
         out_dir: tp.Optional[Path] = None) -> None:
    set_random_seed(RANDOM_SEED)

    # ---- 1) 加载标签 + soundscape 切片 ----
    train_labels, train_ss_labels, taxonomy, label2idx, _ = load_label_files()
    classes = taxonomy.primary_label.values.tolist()

    train_ss_labels_merged = split_train_soundscapes(train_ss_labels)
    print("train_ss_labels_merged.shape:", train_ss_labels_merged.shape)

    # ---- 2) 合并成 train_df + multi-hot 标签矩阵 ----
    train_df, labels_arr = build_train_dataframe(
        train_labels, train_ss_labels_merged, label2idx,
    )
    print("train_df.shape:", train_df.shape)

    # ---- 3) 多标签分层分组 K 折 ----
    mlsgkf = MultiLabelStratifiedGroupKFold(n_splits=N_FOLDS, random_state=RANDOM_SEED)
    train_val_splits = list(mlsgkf.split(labels_arr, train_df["audio_id"].values))

    train_df.insert(5, "fold", -1)
    for fold_id, (_, val_idx) in enumerate(train_val_splits):
        train_df.loc[val_idx, "fold"] = fold_id

    # 各折样本数检查.
    for fold_id in range(N_FOLDS):
        is_ogg = train_df["filename"].str.contains("ogg")
        is_wav = train_df["filename"].str.contains("wav")
        print(f"[fold {fold_id}]")
        print(
            "train_audio      :",
            ((train_df["fold"] != fold_id) & is_ogg).sum(),
            ((train_df["fold"] == fold_id) & is_ogg).sum(),
        )
        print(
            "train_soundscapes:",
            ((train_df["fold"] != fold_id) & is_wav).sum(),
            ((train_df["fold"] == fold_id) & is_wav).sum(),
        )

    # ---- 4) 训练 (默认 4 折; 也可指定子集 fold_ids 来分进程并行) ----
    print(list(filter(lambda x: x[0][:2] != "__", CFG.__dict__.items())))
    if out_dir is None:
        out_dir = make_run_dir(CFG.model_name)
    print(f"[run dir] {out_dir}")

    fold_ids = fold_ids if fold_ids is not None else list(range(N_FOLDS))
    for fold_id in fold_ids:
        train_one_fold(train_df, fold_id, device, classes=classes, output_dir=out_dir)

    # ---- 5) 计算 OOF AUC (仅当 4 fold 全跑完才做) ----
    all_pt_present = all((out_dir / f"best_val_pred_fold{k}.npy").exists()
                         for k in range(N_FOLDS))
    if all_pt_present:
        oof_pred_trn = np.zeros((len(train_df), N_CLASSES))
        oof_pred_trn_rank = np.zeros((len(train_df), N_CLASSES))
        for fold_id, (_, val_idxs) in enumerate(train_val_splits):
            val_pred = np.load(out_dir / f"best_val_pred_fold{fold_id}.npy")
            oof_pred_trn[val_idxs] = val_pred
            oof_pred_trn_rank[val_idxs] = rank_normalize(val_pred)
        print("auc for raw pred :", roc_auc_score(labels_arr, oof_pred_trn))
        print("auc for rank pred:", roc_auc_score(labels_arr, oof_pred_trn_rank))

        # ---- 6) 保存 log-mel jb, 推理脚本通过 joblib.load 复用 ----
        lms_transform_cpu = LogMelSpectrogramTransform(
            CFG.mel_spectrogram_params, top_db=CFG.top_db, lms_shape=CFG.lms_shape,
        ).eval()
        joblib.dump(lms_transform_cpu, out_dir / "log_mel_spectrogram.jb")
    else:
        print(f"[main] skipped OOF AUC: not all 4 folds present in {out_dir}")
        print(f"       run remaining folds with HGNET_RUN_DIR={out_dir} to resume.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=str, default="",
                        help="comma-separated fold ids (e.g. '0,1' or '0'); empty = all 4")
    parser.add_argument("--run-dir", type=str, default="",
                        help="explicit output dir; empty = make a new timestamped one")
    args = parser.parse_args()

    fold_ids = None
    if args.folds:
        fold_ids = [int(x) for x in args.folds.split(",") if x.strip()]

    out_dir = None
    if args.run_dir:
        out_dir = Path(args.run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    main(fold_ids=fold_ids, out_dir=out_dir)
