"""用 dian_pt/repvit_m_1_1_pt 4 折 .pt 直接做 PyTorch 推理
========================================================

跟 ``infer.py`` 的区别:
  - 模型走 PyTorch (LSEModel + repvit_m1_1), 不再依赖 OpenVINO IR;
  - 数据集是任意目录下的 60s 音频 (默认 ``--input-dir`` 指定);
  - 每个文件输出 12 行 (按 5..60s end-sec), 带 ±2.5s shifted TTA;
  - 输出列与 taxonomy.csv 的 234 个 primary_label 对齐, row_id = "{file_id}_{end_sec}".

用法 (在仓库根目录 BirdCLEF++2026/ 下跑):

    python -m hgnet.infer_repvit_pt \
        --input-dir /path/to/your/60s_clips \
        --model-dir /path/to/repo/sed_model/dian_pt/repvit_m_1_1_pt \
        --output    /path/to/repo/submission_repvit.csv \
        --batch-size 16
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import soundfile
import torch
from tqdm import tqdm

from .config import CFG, DATA, N_CLASSES, N_FOLDS, RANDOM_SEED, SAMPLING_RATE
from .models import LSEModel
from .transforms import LogMelSpectrogramTransform
from .utils import device, rank_normalize, set_random_seed, sigmoid


# ============================================================
#  1. 读音频 -> normal 12 段 + shifted 13 段切片
# ============================================================
def build_wave_arrays(audio_paths, max_sec: int = 60):
    """返回 (normal_waves [num_audios * 12, 5*sr], shifted_waves [num_audios * 13, 5*sr])."""
    sr = SAMPLING_RATE
    duration = sr * max_sec
    shift = int(sr * 2.5)

    normal_list, shifted_list = [], []
    for path in tqdm(audio_paths, desc="loading waves"):
        with soundfile.SoundFile(path) as f:
            n_frames = f.frames
            wave = np.zeros(shift + duration + shift, dtype="float32")
            if n_frames < duration + shift:
                # 短于 60 + 2.5 s, 直接 read 完丢前面 shift 偏移位置.
                wave[shift : shift + n_frames] = f.read(dtype="float32")
            else:
                wave[shift : shift + duration + shift] = f.read(
                    frames=duration + shift, dtype="float32"
                )

        # normal 12 段: [0..5), [5..10), ..., [55..60)
        for s in range(0, max_sec, 5):
            normal_list.append(wave[shift + s * sr : shift + (s + 5) * sr])
        # shifted 13 段: [-2.5..2.5), [2.5..7.5), ..., [57.5..62.5)
        for s in range(0, max_sec + 5, 5):
            shifted_list.append(wave[s * sr : (s + 5) * sr])

    return (
        np.stack(normal_list, axis=0).astype(np.float32),
        np.stack(shifted_list, axis=0).astype(np.float32),
    )


# ============================================================
#  2. PyTorch 推理 (CPU/GPU 自动)
# ============================================================
@torch.no_grad()
def run_inference(model, lms_transform, waves: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """waves: (N, 5*sr)  ->  preds: (N, N_CLASSES) sigmoid 概率."""
    n = waves.shape[0]
    out = np.zeros((n, N_CLASSES), dtype=np.float32)
    for i in tqdm(range(0, n, batch_size), desc="infer", leave=False):
        wb = torch.from_numpy(waves[i : i + batch_size]).to(device, non_blocking=True)
        lms = lms_transform(wb)            # (B, 1, H, W)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(lms)
        else:
            logits = model(lms)
        out[i : i + batch_size] = logits.float().cpu().numpy()
    return sigmoid(out)


# ============================================================
#  3. 主入口
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path,
                   help="存放 60s 音频的目录 (会递归读 *.ogg / *.wav / *.flac).")
    p.add_argument("--filename-csv", type=Path, default=None,
                   help="可选, csv 必须有 filename 列, 只对该列出现过的文件做推理 "
                        "(用于在大目录里筛子集, 例如 train_soundscapes_labels.csv).")
    p.add_argument("--model-dir", default=Path("./models/repvit_m1_1_pt"),
                   type=Path, help="包含 best_model_fold0.pt ... fold3.pt 的目录.")
    p.add_argument("--model-name", default="repvit_m1_1",
                   help="timm backbone 名称, 默认 repvit_m1_1.")
    p.add_argument("--output", default=Path("submission_repvit.csv"),
                   type=Path, help="输出 csv 路径.")
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--rank-avg", action="store_true",
                   help="折间用 rank-normalize 后再平均 (默认: 直接均值).")
    p.add_argument("--n-folds", default=N_FOLDS, type=int,
                   help="参与融合的折数 (默认 4).")
    p.add_argument("--label-csv", type=Path, default=None,
                   help="可选, 形如 train_soundscapes_labels.csv. 给了就在推理结束后"
                        "用 sklearn 算 macro/micro ROC-AUC (只用 csv 里出现过的 segment).")
    return p.parse_args()


def collect_audio_paths(root: Path) -> list[Path]:
    exts = {".ogg", ".wav", ".flac", ".mp3"}
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"no audio files under {root}")
    return paths


def main() -> None:
    args = parse_args()
    set_random_seed(RANDOM_SEED)

    # ---- 类别表 ----
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    classes = taxonomy.primary_label.values.tolist()
    assert len(classes) == N_CLASSES, f"taxonomy 类别数={len(classes)}, 期望 {N_CLASSES}"

    # ---- 音频列表 + row_id ----
    audio_paths = collect_audio_paths(args.input_dir)
    if args.filename_csv is not None:
        wanted = set(pd.read_csv(args.filename_csv)["filename"].astype(str).unique().tolist())
        audio_paths = [p for p in audio_paths if p.name in wanted]
        missing = wanted - {p.name for p in audio_paths}
        if missing:
            print(f"[warn] {len(missing)} files in csv not found under {args.input_dir}, e.g. {list(missing)[:3]}")
        print(f"[filter] {len(audio_paths)} / {len(wanted)} files matched from csv")
    print(f"[input] {len(audio_paths)} files under {args.input_dir}")
    seg_row_ids: list[str] = []
    for p in audio_paths:
        for end_sec in range(5, 65, 5):
            seg_row_ids.append(f"{p.stem}_{end_sec}")
    num_audios = len(audio_paths)
    num_normal_segs = num_audios * 12
    num_shifted_segs = num_audios * 13

    # ---- log-mel ----
    lms_transform = LogMelSpectrogramTransform(
        CFG.mel_spectrogram_params, top_db=CFG.top_db, lms_shape=CFG.lms_shape,
    ).eval().to(device)

    # ---- 切片 ----
    normal_waves, shifted_waves = build_wave_arrays(audio_paths)
    print(f"[shape] normal {normal_waves.shape}, shifted {shifted_waves.shape}")

    # ---- 4 折 PyTorch 推理 ----
    print(f"[device] {device}")
    pred_normal = np.zeros((args.n_folds, num_normal_segs, N_CLASSES), dtype=np.float32)
    pred_shifted = np.zeros((args.n_folds, num_shifted_segs, N_CLASSES), dtype=np.float32)

    for fold_id in range(args.n_folds):
        ck_path = args.model_dir / f"best_model_fold{fold_id}.pt"
        print(f"\n[fold {fold_id}] {ck_path}")
        model = LSEModel(
            args.model_name, pretrained=False, drop_path_rate=0.0,
            num_classes=N_CLASSES,
            head_dropout=CFG.head_dropout,
            lse_temperature=CFG.lse_temperature,
        )
        state = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval().to(device)

        t0 = time()
        pred_normal[fold_id] = run_inference(model, lms_transform, normal_waves, args.batch_size)
        print(f"  normal done in {time() - t0:.1f}s")
        t0 = time()
        pred_shifted[fold_id] = run_inference(model, lms_transform, shifted_waves, args.batch_size)
        print(f"  shifted done in {time() - t0:.1f}s")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ---- TTA 融合 ----
    pn = pred_normal.reshape(args.n_folds, num_audios, 12, N_CLASSES)
    ps = pred_shifted.reshape(args.n_folds, num_audios, 13, N_CLASSES)
    tta = (
        0.25 * ps[..., 0:12, :]
        + 0.50 * pn
        + 0.25 * ps[..., 1:13, :]
    ).reshape(args.n_folds, num_normal_segs, N_CLASSES)

    if args.rank_avg:
        tta_rank = np.zeros_like(tta)
        for fold_id in range(args.n_folds):
            tta_rank[fold_id] = rank_normalize(tta[fold_id])
        avg = tta_rank.mean(axis=0)
    else:
        avg = tta.mean(axis=0)

    # ---- 落盘 ----
    sub_df = pd.DataFrame(
        avg, columns=classes,
        index=pd.Series(seg_row_ids, name="row_id"),
    ).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(args.output, index=False)
    print(f"\n[done] wrote {sub_df.shape} -> {args.output}")
    print(sub_df.head())

    # ---- 可选: 算 AUC ----
    if args.label_csv is not None:
        evaluate_auc(sub_df, classes, args.label_csv)


def evaluate_auc(sub_df: pd.DataFrame, classes: list[str], label_csv: Path) -> None:
    """用 train_soundscapes_labels.csv 风格的标签算 macro/micro ROC-AUC.

    标签 csv schema: filename, start, end ('HH:MM:SS'), primary_label (';' 分隔多标签).
    实际只用 (filename, end) 与 sub_df 的 row_id="{stem}_{end_sec}" 对齐.
    """
    from sklearn.metrics import roc_auc_score

    labels_raw = pd.read_csv(label_csv)
    # end 'HH:MM:SS' -> 总秒数
    end_sec = labels_raw["end"].str.split(":").apply(
        lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2])
    )
    labels_raw["row_id"] = (
        labels_raw["filename"].str.replace(".ogg", "", regex=False) + "_" + end_sec.astype(str)
    )
    # 同一 row_id 多行 -> 合并多标签 (虽然实际是重复, 合并后等价).
    labels_grp = (
        labels_raw.groupby("row_id")["primary_label"]
        .apply(lambda s: ";".join(s.dropna().astype(str)))
        .reset_index()
    )

    # 构造 multi-hot.
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_true = np.zeros((len(labels_grp), len(classes)), dtype=np.float32)
    miss_labels = set()
    for i, lab in enumerate(labels_grp["primary_label"].fillna("")):
        if not lab:
            continue
        for c in lab.split(";"):
            c = c.strip()
            if not c:
                continue
            j = cls_to_idx.get(c)
            if j is None:
                miss_labels.add(c)
            else:
                y_true[i, j] = 1.0

    if miss_labels:
        print(f"[eval] {len(miss_labels)} labels not in taxonomy, ignored "
              f"(e.g. {sorted(miss_labels)[:5]}).")

    # 取交集 row_id, 并按 labels_grp 顺序排.
    merged = labels_grp.merge(sub_df, on="row_id", how="inner")
    if len(merged) != len(labels_grp):
        print(f"[eval] {len(labels_grp) - len(merged)} labeled rows not found in submission, "
              "they are skipped.")
    # 按 merged 重新取 y_true.
    y_true = y_true[labels_grp["row_id"].isin(merged["row_id"]).values]
    y_pred = merged[classes].values

    # 只对有正例的类算 macro AUC.
    has_pos = y_true.sum(axis=0) > 0
    print(f"[eval] segments={len(merged)}, classes_with_pos={int(has_pos.sum())}/{len(classes)}, "
          f"avg labels/seg={y_true.sum(axis=1).mean():.2f}")

    macro_auc = roc_auc_score(y_true[:, has_pos], y_pred[:, has_pos], average="macro")
    micro_auc = roc_auc_score(y_true[:, has_pos], y_pred[:, has_pos], average="micro")
    # rank-normalized 后再算一遍.
    y_pred_rank = rank_normalize(y_pred)
    macro_auc_r = roc_auc_score(y_true[:, has_pos], y_pred_rank[:, has_pos], average="macro")
    print("[eval] macro ROC-AUC          :", round(macro_auc, 5))
    print("[eval] macro ROC-AUC (rank)   :", round(macro_auc_r, 5))
    print("[eval] micro ROC-AUC          :", round(micro_auc, 5))


if __name__ == "__main__":
    main()
