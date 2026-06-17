"""推理脚本: 生成 submission.csv (带 ±2.5s 偏移 TTA)
============================================================

来源: ``birdclef-2026-hgnetv2-b0-baseline-inference.ipynb``.

流程:
    1. 读 sample_submission, 解析出本环境的 ``test_ss_paths`` (官方测试 / 本地 debug);
    2. 对每条 60s 音频构造两套切片:
         - normal  : [0..5), [5..10), ... [55..60)        共 12 段;
         - shifted : [-2.5..2.5), [2.5..7.5), ... [57.5..62.5)  共 13 段
                     (头尾各 pad 2.5s 0);
    3. 用 ``LogMelSpectrogramTransform`` (CPU, joblib 并行) 把波形 batch 转成 log-mel;
    4. 用 4 折 OpenVINO 模型分别异步推理, 取 sigmoid;
    5. TTA 融合 (shifted 头 12 段 * 0.25 + normal 12 段 * 0.5 + shifted 尾 12 段 * 0.25);
    6. 按 sample_submission 的 row_id 顺序写出 ``submission.csv``.

使用:

    python -m hgnet.infer

模型路径默认是 ``cwd`` (与 ``train.py`` / ``export.py`` 落盘位置一致);
如果模型放在 Kaggle 输入挂载目录, 修改 ``TRAINED_MODEL_DIR`` 即可.
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
from tqdm import tqdm, trange

from .config import (
    DATA,
    INFER_CFG,
    N_CLASSES,
    N_FOLDS,
    RANDOM_SEED,
    SAMPLING_RATE,
    TEST_SS,
    TRAIN_SS,
    resolve_run_dir,
)
from .transforms import LogMelSpectrogramTransform
from .utils import rank_normalize, set_random_seed, sigmoid


# ============================================================
#                  1. 准备测试音频路径
# ============================================================
def collect_test_paths(debug: bool = INFER_CFG.debug):
    """返回 (sample_sub, test_ss_paths, is_test_env).

    - is_test_env=True : 提交时的隐藏测试集, 用 sample_submission 列表;
    - is_test_env=False:
        * debug=True   时, 用 train_soundscapes 的前 600 条本地试跑;
        * debug=False  时, 用前 10 条 (fast submit).
    """
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    is_test_env = len(sample_sub) > 10

    if is_test_env:
        test_ss_paths = []
        added = set()
        for row_id in sample_sub["row_id"].values:
            file_id = "_".join(row_id.split("_")[:-1])
            if file_id in added:
                continue
            added.add(file_id)
            test_ss_paths.append(TEST_SS / f"{file_id}.ogg")
    else:
        if debug:
            test_ss_paths = sorted(TRAIN_SS.iterdir())[:600]
        else:
            test_ss_paths = sorted(TRAIN_SS.iterdir())[:10]

    return sample_sub, test_ss_paths, is_test_env


def make_test_ss_segs(test_ss_paths) -> list:
    """每个 60s 音频展开成 12 个 5s 段的 row_id 列表."""
    segs = []
    for p in test_ss_paths:
        for i in range(0, 60, 5):
            segs.append(f"{p.stem}_{i + 5}")
    return segs


# ============================================================
#         2. 构造 normal / shifted 两套切片 (TTA)
# ============================================================
def build_wave_batches(test_ss_paths, batch_size: int = INFER_CFG.batch_size):
    """对每条 60s 音频, 切出 normal (12 段) + shifted (13 段) 两套 5s 切片, 拼成 batch.

    每条音频会先 pad 头尾各 2.5s 的 0, 这样 [-2.5..2.5) 也能直接取得到.
    """
    sample_rate = SAMPLING_RATE
    max_sec = 60
    duration = sample_rate * max_sec
    shift = int(sample_rate * 2.5)

    wave_list = []
    shifted_wave_list = []

    for path in tqdm(test_ss_paths, desc="loading test waves"):
        with soundfile.SoundFile(path) as f:
            n_frames = f.frames
            wave = np.zeros(shift + duration + shift, dtype="float32")
            if n_frames < duration + shift:
                wave[shift : shift + n_frames] = f.read(dtype="float32")
            else:
                wave[shift : shift + duration + shift] = f.read(
                    frames=duration + shift, dtype="float32"
                )

        # normal: [0..5), [5..10), ..., [55..60)
        for seg_sec in range(0, max_sec, 5):
            wave_list.append(
                wave[shift + seg_sec * sample_rate : shift + (seg_sec + 5) * sample_rate]
            )
        # shifted: [-2.5..2.5), [2.5..7.5), ..., [57.5..62.5)
        for seg_sec in range(0, max_sec + 5, 5):
            shifted_wave_list.append(
                wave[seg_sec * sample_rate : (seg_sec + 5) * sample_rate]
            )

    num_audios = len(test_ss_paths)
    num_segs = len(wave_list)
    num_shifted_segs = len(shifted_wave_list)
    print(num_segs, num_segs / 12)
    print(num_shifted_segs, num_shifted_segs / 13)

    wave_batches = []
    shifted_wave_batches = []
    for i in trange(0, len(wave_list), batch_size, desc="batching normal"):
        wave_batches.append(np.stack(wave_list[i : i + batch_size], axis=0))
    for i in trange(0, len(shifted_wave_list), batch_size, desc="batching shifted"):
        shifted_wave_batches.append(np.stack(shifted_wave_list[i : i + batch_size], axis=0))

    return wave_batches, shifted_wave_batches, num_audios, num_segs, num_shifted_segs


def compute_lms_batches(wave_batches, lms_transform, n_jobs: int = INFER_CFG.lms_n_jobs):
    """用 joblib 并行把波形 batch 转 log-mel batch, 返回 (B, 1, 256, 256) float32 列表."""
    lms_batches = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(lms_transform)(torch.from_numpy(waves)) for waves in wave_batches
    )
    return [np.ascontiguousarray(lms, dtype=np.float32) for lms in lms_batches]


# ============================================================
#                  3. OpenVINO 推理工具
# ============================================================
def compile_ov_model(ov_model_path, cfg=INFER_CFG):
    return ov.compile_model(str(ov_model_path), "CPU", cfg.ov_compile_config)


def async_infer_with_order(model, lms_batches, num_requests: int = INFER_CFG.num_requests):
    """利用 ``ov.AsyncInferQueue`` 异步并行推理, 并按原顺序写回结果."""
    idx_batches = []
    tmp_idx = 0
    for b in lms_batches:
        b_size = len(b)
        idx_batches.append(np.arange(tmp_idx, tmp_idx + b_size))
        tmp_idx += b_size
    n_records = tmp_idx

    infer_queue = ov.AsyncInferQueue(model, num_requests)
    logit_arr = np.zeros((n_records, N_CLASSES), dtype=np.float32)

    start_time = time()

    def callback(request, userdata):
        input_idxs = userdata
        logit_arr[input_idxs] = request.get_output_tensor().data

    infer_queue.set_callback(callback)
    input_name = model.inputs[0].get_any_name()

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

    # ---- 0) 类别表 ----
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    CLASSES = taxonomy.primary_label.values.tolist()

    # ---- 1) 测试音频路径 + row_id 列表 ----
    sample_sub, test_ss_paths, _ = collect_test_paths()
    test_ss_segs = make_test_ss_segs(test_ss_paths)

    # ---- 2) log-mel transform (CPU) ----
    from .config import CFG  # 复用训练时一致的 mel 参数.
    lms_transform = LogMelSpectrogramTransform(
        CFG.mel_spectrogram_params, top_db=CFG.top_db, lms_shape=CFG.lms_shape,
    ).eval()

    # ---- 3) 加载 4 折 OpenVINO 模型 ----
    # 默认找当前模型最新的 run 目录; 也可通过 HGNET_RUN_DIR 指定具体目录.
    run_dir = resolve_run_dir(CFG.model_name)
    print(f"[run dir] {run_dir}")
    ov_model_list = []
    for fold_id in range(N_FOLDS):
        ov_model_list.append(
            compile_ov_model(run_dir / f"best_model_fold{fold_id}.xml")
        )

    # ---- 4) 构造 wave batches + 计算 log-mel ----
    wave_batches, shifted_wave_batches, num_audios, num_segs, num_shifted_segs = build_wave_batches(
        test_ss_paths
    )

    lms_batches = compute_lms_batches(wave_batches, lms_transform)
    shifted_lms_batches = compute_lms_batches(shifted_wave_batches, lms_transform)

    del wave_batches, shifted_wave_batches
    gc.collect()

    # ---- 5) 4 折 OpenVINO 推理 ----
    test_preds_arr = np.zeros((N_FOLDS, num_segs, N_CLASSES), dtype=np.float32)
    for fold_id, ov_model in enumerate(ov_model_list):
        print(f"[fold: {fold_id}]")
        test_pred = async_infer_with_order(ov_model, lms_batches)
        test_preds_arr[fold_id] = sigmoid(test_pred)

    shifted_test_preds_arr = np.zeros((N_FOLDS, num_shifted_segs, N_CLASSES), dtype=np.float32)
    for fold_id, ov_model in enumerate(ov_model_list):
        print(f"[fold: {fold_id}] (shifted)")
        test_pred = async_infer_with_order(ov_model, shifted_lms_batches)
        shifted_test_preds_arr[fold_id] = sigmoid(test_pred)

    # ---- 6) TTA 融合 ----
    test_preds_arr_by_record = test_preds_arr.reshape(N_FOLDS, num_audios, 12, N_CLASSES)
    shifted_by_record = shifted_test_preds_arr.reshape(N_FOLDS, num_audios, 13, N_CLASSES)
    # 每条记录: 0.25 * shifted[:12]  +  0.5 * normal[:12]  +  0.25 * shifted[1:13].
    tta_by_records = (
        0.25 * shifted_by_record[..., 0:12, :]
        + 0.50 * test_preds_arr_by_record
        + 0.25 * shifted_by_record[..., 1:13, :]
    )
    tta = tta_by_records.reshape(N_FOLDS, num_segs, N_CLASSES)

    # 可选 rank-normalize 后再融合.
    if INFER_CFG.rank_avg:
        tta_rank = np.zeros_like(tta)
        for fold_id in range(N_FOLDS):
            tta_rank[fold_id] = rank_normalize(tta[fold_id])
        test_pred_avg = tta_rank.mean(axis=0)
    else:
        test_pred_avg = tta.mean(axis=0)

    # ---- 7) 写 submission.csv ----
    sub_df = pd.DataFrame(
        test_pred_avg,
        columns=CLASSES,
        index=pd.Series(test_ss_segs, name="row_id"),
    ).reset_index()

    sub_df = pd.merge(sample_sub[["row_id"]], sub_df, on="row_id", how="left")
    sub_df.to_csv("submission.csv", index=False)
    print(sub_df.shape)
    print(sub_df.head())


if __name__ == "__main__":
    main()
