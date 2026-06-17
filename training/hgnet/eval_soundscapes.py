"""在 66 个有标签的 train_soundscapes 上做 OpenVINO 推理 + 算 macro AUC.

流程:
    1. 读 train_soundscapes_labels.csv, 取出涉及到的 ogg 文件;
    2. 对每个文件做 normal (12 段) + shifted (13 段) 切片, ±2.5s TTA;
    3. CPU log-mel + 4 折 OpenVINO 异步推理, sigmoid;
    4. TTA 融合 + fold 平均, 得到每个 5s 段的 234 类概率;
    5. 跟 labels_csv 的真实标签 join 起来 (按 filename + start_sec), 算 macro ROC-AUC.

使用:
    # 先把 .pt 转成 .xml/.bin
    python -m hgnet.export --skip-oof
    # 再跑评估
    python -m hgnet.eval_soundscapes

可选环境变量:
    HGNET_RUN_DIR        : 指向 .xml/.bin 所在目录, 默认走 resolve_run_dir.
    HGNET_MODEL_NAME     : timm 模型名 (export.py 时已经定型, 这里只用于 resolve_run_dir).

输出:
    eval_soundscapes_pred.csv : 每段一行, 列 = filename, start_sec, end_sec, 234 类概率
    控制台打印 macro AUC / per-class AUC top10 / bottom10.
"""

from __future__ import annotations

import gc
from pathlib import Path
from time import time

import numpy as np
import openvino as ov
import pandas as pd
import soundfile
import torch
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange

from .config import (
    CFG,
    DATA,
    INFER_CFG,
    N_CLASSES,
    N_FOLDS,
    RANDOM_SEED,
    SAMPLING_RATE,
    TRAIN_SS,
    resolve_run_dir,
)
from .transforms import LogMelSpectrogramTransform
from .utils import set_random_seed, sigmoid


# ============================================================
#                  1. 准备文件 + 标签
# ============================================================
def load_eval_data():
    """读 train_soundscapes_labels.csv, 转出:
    - file_paths: list[Path], 涉及的 ogg 路径 (按字母序去重)
    - labels_df : 包含 filename / start_sec / end_sec / multi-hot 标签 的 DataFrame
    - classes   : 长度 234 的类名列表 (与训练保持一致)
    """
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    classes = taxonomy.primary_label.values.tolist()
    label2idx = {label: idx for idx, label in enumerate(classes)}

    df = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    df = df.drop_duplicates().reset_index(drop=True)

    # HH:MM:SS -> 秒
    df["start"] = pd.to_datetime(df["start"], format="%H:%M:%S")
    df["end"] = pd.to_datetime(df["end"], format="%H:%M:%S")
    df["start_sec"] = df["start"].dt.minute * 60 + df["start"].dt.second
    df["end_sec"] = df["end"].dt.minute * 60 + df["end"].dt.second

    # primary_label 是 ";" 分隔的多标签, 转成 one-hot.
    multi_hot = np.zeros((len(df), N_CLASSES), dtype=np.float32)
    for i, raw in enumerate(df["primary_label"].values):
        for lab in str(raw).split(";"):
            lab = lab.strip()
            if lab in label2idx:
                multi_hot[i, label2idx[lab]] = 1.0
    label_cols = [f"_y_{c}" for c in classes]
    df_labels = pd.DataFrame(multi_hot, columns=label_cols)
    labels_df = pd.concat([df[["filename", "start_sec", "end_sec"]], df_labels], axis=1)

    file_paths = sorted({TRAIN_SS / fn for fn in df["filename"].unique()})
    return file_paths, labels_df, classes, label_cols


# ============================================================
#                  2. 切片 + log-mel
# ============================================================
def build_wave_batches(file_paths, batch_size: int):
    """对每条音频 pad 头尾 2.5s, 切 normal (12 段) + shifted (13 段), 拼 batch.

    并不强制 60s, 实际长度根据文件 frames 计算, 多于 60s 的部分忽略, 不足按 segments
    数量做最大 floor (start + 5s <= 总长 + 头尾各 2.5s pad).
    """
    sample_rate = SAMPLING_RATE
    max_sec = 60
    duration = sample_rate * max_sec
    shift = int(sample_rate * 2.5)

    wave_list, shifted_wave_list = [], []
    seg_meta_list, shifted_seg_meta_list = [], []  # (filename, start_sec)

    for path in tqdm(file_paths, desc="loading eval waves"):
        with soundfile.SoundFile(path) as f:
            n_frames = f.frames
            wave = np.zeros(shift + duration + shift, dtype="float32")
            n_to_read = min(n_frames, duration + shift)
            data = f.read(frames=n_to_read, dtype="float32")
            wave[shift : shift + len(data)] = data

        fn = path.name
        # normal: [0..5), [5..10), ..., [55..60) -> 12 段
        for seg_sec in range(0, max_sec, 5):
            wave_list.append(
                wave[shift + seg_sec * sample_rate : shift + (seg_sec + 5) * sample_rate]
            )
            seg_meta_list.append((fn, seg_sec))
        # shifted: [-2.5..2.5), [2.5..7.5), ..., [57.5..62.5) -> 13 段
        for seg_sec in range(0, max_sec + 5, 5):
            shifted_wave_list.append(
                wave[seg_sec * sample_rate : (seg_sec + 5) * sample_rate]
            )
            shifted_seg_meta_list.append((fn, seg_sec))

    num_audios = len(file_paths)
    print(f"normal segs : {len(wave_list)} ({len(wave_list) / 12:.0f} files * 12)")
    print(f"shifted segs: {len(shifted_wave_list)} ({len(shifted_wave_list) / 13:.0f} files * 13)")

    def _batch(seq, bs):
        return [np.stack(seq[i : i + bs], axis=0) for i in range(0, len(seq), bs)]

    return (
        _batch(wave_list, batch_size),
        _batch(shifted_wave_list, batch_size),
        num_audios,
        seg_meta_list,
        shifted_seg_meta_list,
    )


def compute_lms_batches(wave_batches, lms_transform, n_jobs: int):
    lms_batches = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(lms_transform)(torch.from_numpy(waves)) for waves in wave_batches
    )
    return [np.ascontiguousarray(lms, dtype=np.float32) for lms in lms_batches]


# ============================================================
#                  3. OpenVINO 异步推理
# ============================================================
def compile_ov_model(ov_model_path):
    return ov.compile_model(str(ov_model_path), "CPU", INFER_CFG.ov_compile_config)


def async_infer_with_order(model, lms_batches, num_requests: int):
    idx_batches = []
    tmp_idx = 0
    for b in lms_batches:
        b_size = len(b)
        idx_batches.append(np.arange(tmp_idx, tmp_idx + b_size))
        tmp_idx += b_size
    n_records = tmp_idx

    infer_queue = ov.AsyncInferQueue(model, num_requests)
    logit_arr = np.zeros((n_records, N_CLASSES), dtype=np.float32)

    def callback(request, userdata):
        logit_arr[userdata] = request.get_output_tensor().data

    infer_queue.set_callback(callback)
    input_name = model.inputs[0].get_any_name()

    start_time = time()
    for idxs, lms in zip(idx_batches, lms_batches):
        infer_queue.start_async({input_name: lms}, userdata=idxs)
    infer_queue.wait_all()
    print(f"... Done by {time() - start_time:.2f} sec")
    return logit_arr


# ============================================================
#                  4. 主入口
# ============================================================
def main() -> None:
    set_random_seed(RANDOM_SEED)

    # ---- 0) 数据 + 标签 ----
    file_paths, labels_df, classes, label_cols = load_eval_data()
    print(f"#files: {len(file_paths)}, #labeled segs: {len(labels_df)}")

    # ---- 1) log-mel transform ----
    lms_transform = LogMelSpectrogramTransform(
        CFG.mel_spectrogram_params, top_db=CFG.top_db, lms_shape=CFG.lms_shape,
    ).eval()

    # ---- 2) 加载 4 折 OpenVINO ----
    run_dir = resolve_run_dir(CFG.model_name)
    print(f"[run dir] {run_dir}")
    ov_models = [
        compile_ov_model(run_dir / f"best_model_fold{k}.xml") for k in range(N_FOLDS)
    ]

    # ---- 3) 切片 + log-mel ----
    wave_batches, shifted_wave_batches, num_audios, seg_meta, shifted_seg_meta = (
        build_wave_batches(file_paths, batch_size=INFER_CFG.batch_size)
    )

    lms_batches = compute_lms_batches(wave_batches, lms_transform, INFER_CFG.lms_n_jobs)
    shifted_lms_batches = compute_lms_batches(shifted_wave_batches, lms_transform, INFER_CFG.lms_n_jobs)
    del wave_batches, shifted_wave_batches
    gc.collect()

    # ---- 4) 4 折推理 ----
    num_segs = len(seg_meta)
    num_shifted = len(shifted_seg_meta)
    test_preds = np.zeros((N_FOLDS, num_segs, N_CLASSES), dtype=np.float32)
    shifted_preds = np.zeros((N_FOLDS, num_shifted, N_CLASSES), dtype=np.float32)
    for k, m in enumerate(ov_models):
        print(f"[fold {k}] normal")
        test_preds[k] = sigmoid(async_infer_with_order(m, lms_batches, INFER_CFG.num_requests))
        print(f"[fold {k}] shifted")
        shifted_preds[k] = sigmoid(async_infer_with_order(m, shifted_lms_batches, INFER_CFG.num_requests))

    # ---- 5) TTA + fold 平均 ----
    test_preds = test_preds.reshape(N_FOLDS, num_audios, 12, N_CLASSES)
    shifted_preds = shifted_preds.reshape(N_FOLDS, num_audios, 13, N_CLASSES)
    tta = (
        0.25 * shifted_preds[..., 0:12, :]
        + 0.50 * test_preds
        + 0.25 * shifted_preds[..., 1:13, :]
    )
    tta = tta.reshape(N_FOLDS, num_segs, N_CLASSES).mean(axis=0)  # (num_segs, N_CLASSES)

    # ---- 6) join 真实标签 + 算 AUC ----
    pred_df = pd.DataFrame(tta, columns=classes)
    pred_df.insert(0, "start_sec", [m[1] for m in seg_meta])
    pred_df.insert(0, "filename", [m[0] for m in seg_meta])
    pred_df["end_sec"] = pred_df["start_sec"] + 5

    merged = pd.merge(
        labels_df, pred_df,
        on=["filename", "start_sec", "end_sec"],
        how="inner",
    )
    print(f"merged rows = {len(merged)} / labeled rows = {len(labels_df)}")

    y_true = merged[label_cols].values.astype(np.int32)
    y_pred = merged[classes].values.astype(np.float32)

    mask = y_true.sum(axis=0) > 0
    macro_auc = roc_auc_score(y_true[:, mask], y_pred[:, mask], average="macro")
    print(f"macro ROC-AUC: {macro_auc:.5f}  (over {mask.sum()} active classes)")

    # per-class AUC top/bottom
    per_class = []
    for i, c in enumerate(classes):
        if mask[i]:
            try:
                auc_i = roc_auc_score(y_true[:, i], y_pred[:, i])
                per_class.append((c, auc_i, int(y_true[:, i].sum())))
            except ValueError:
                pass
    per_class.sort(key=lambda x: x[1])
    print("\nbottom-10 classes by AUC:")
    for c, a, n in per_class[:10]:
        print(f"  {c}: AUC={a:.4f}  n_pos={n}")
    print("\ntop-10 classes by AUC:")
    for c, a, n in per_class[-10:]:
        print(f"  {c}: AUC={a:.4f}  n_pos={n}")

    # ---- 7) 落盘预测 ----
    out_csv = run_dir / "eval_soundscapes_pred.csv"
    pred_df.to_csv(out_csv, index=False)
    print(f"\nsaved predictions -> {out_csv}")


if __name__ == "__main__":
    main()
