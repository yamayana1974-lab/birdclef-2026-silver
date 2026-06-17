"""把训练好的 torch 模型导出成 ONNX -> OpenVINO IR
=================================================================

包含:

- ``convert_torch_to_onnx_to_ov`` : 单个 .pt -> 中间 .onnx -> OpenVINO .xml / .bin.
- ``async_infer_with_order``      : 利用 ``ov.AsyncInferQueue`` 异步并行推理,
                                    并按原顺序写回结果数组.
- ``main``                        : 把 4 折 .pt 全部转 OpenVINO, 然后在 CPU 上重新跑一遍
                                    OOF, 用 AUC 校验数值一致性.

使用:

    python -m hgnet.export

假设训练产物在 ``cwd``: ``best_model_fold{0..3}.pt``,
转出后产物: ``best_model_fold{0..3}.xml`` / ``.bin``.
"""

from __future__ import annotations

import gc
from pathlib import Path
from time import time

import numpy as np
import openvino as ov
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .config import CFG, N_CLASSES, N_FOLDS, RANDOM_SEED, resolve_run_dir
from .data import (
    MultiLabelStratifiedGroupKFold,
    build_train_dataframe,
    load_label_files,
    split_train_soundscapes,
)
from .dataset import get_data_loader
from .models import LSEModel
from .transforms import LogMelSpectrogramTransform
from .utils import device, rank_normalize, set_random_seed, to_device


# ============================================================
#                  1. torch -> ONNX -> OpenVINO
# ============================================================
def convert_torch_to_onnx_to_ov(cfg, model_path: Path, out_dir: Path) -> "ov.Model":
    """单文件转换: ``best_model_foldX.pt`` -> ``best_model_foldX.xml/.bin``."""
    torch_model = LSEModel(
        cfg.model_name, pretrained=False, drop_path_rate=cfg.drop_path_rate,
        num_classes=N_CLASSES, head_dropout=cfg.head_dropout,
        lse_temperature=cfg.lse_temperature,
    )
    torch_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    torch_model.eval()

    dummy_input = torch.randn(64, 1, 256, 256, dtype=torch.float32)
    onnx_path = out_dir / f"{model_path.stem}.onnx"
    torch.onnx.export(
        torch_model, (dummy_input,), str(onnx_path),
        opset_version=11, do_constant_folding=True, dynamo=False,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    ov_model = ov.convert_model(onnx_path)
    onnx_path.unlink()  # 中间产物用完即删, 不留垃圾.
    ov.save_model(ov_model, out_dir / f"{model_path.stem}.xml")
    return ov_model


# ============================================================
#                  2. OpenVINO 异步推理
# ============================================================
def async_infer_with_order(model, lms_batches, num_requests: int = 4) -> np.ndarray:
    """异步并行推理后按原顺序写回, 输出形状 (n_records, N_CLASSES)."""
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
    input_name = model.inputs[0]

    for idxs, lms in zip(idx_batches, lms_batches):
        infer_queue.start_async({input_name: lms}, userdata=idxs)
    infer_queue.wait_all()

    print(f"... Done by {time() - start_time:.2f} sec")
    return logit_arr


# ============================================================
#                  3. 主入口: 转模型 + OV OOF 校验
# ============================================================
def main(skip_oof: bool = False) -> None:
    set_random_seed(RANDOM_SEED)
    # 默认找当前模型最新的 run 目录; 也可通过 HGNET_RUN_DIR 指定具体目录.
    out_dir = resolve_run_dir(CFG.model_name)
    print(f"[run dir] {out_dir}")

    # ---- 1) 转 4 折模型 ----
    for fold_id in range(N_FOLDS):
        convert_torch_to_onnx_to_ov(CFG, out_dir / f"best_model_fold{fold_id}.pt", out_dir)

    if skip_oof:
        print("[export] --skip-oof 已设置, 跳过 OOF AUC 校验.")
        return

    # 重新跑一遍数据 pipeline, 拿到与训练阶段相同的 fold 切分, 以便定位 val_idxs.
    train_labels, train_ss_labels, taxonomy, label2idx, _ = load_label_files()
    classes = taxonomy.primary_label.values.tolist()

    train_ss_labels_merged = split_train_soundscapes(train_ss_labels)
    train_df, labels_arr = build_train_dataframe(train_labels, train_ss_labels_merged, label2idx)

    mlsgkf = MultiLabelStratifiedGroupKFold(n_splits=N_FOLDS, random_state=RANDOM_SEED)
    train_val_splits = list(mlsgkf.split(labels_arr, train_df["audio_id"].values))
    train_df.insert(5, "fold", -1)
    for fold_id, (_, val_idx) in enumerate(train_val_splits):
        train_df.loc[val_idx, "fold"] = fold_id

    # ---- 2) 用 OpenVINO 在 CPU 上重新跑 OOF, 校验数值一致性 ----
    lms_transform = LogMelSpectrogramTransform(
        CFG.mel_spectrogram_params, top_db=CFG.top_db, lms_shape=CFG.lms_shape,
    ).eval().to(device)

    ov_model_list = []
    for fold_id in range(N_FOLDS):
        compiled = ov.compile_model(
            str(out_dir / f"best_model_fold{fold_id}.xml"), "CPU",
            {
                "PERFORMANCE_HINT": "THROUGHPUT",
                "INFERENCE_NUM_THREADS": 4,
                "NUM_STREAMS": 2,
            },
        )
        ov_model_list.append(compiled)
    gc.collect()

    oof_pred_ov = np.zeros((len(train_df), N_CLASSES), dtype="float32")
    oof_pred_ov_rank = np.zeros((len(train_df), N_CLASSES), dtype="float32")

    # OV 推理把 batch 调小, 异步并发更高效.
    CFG.batch_size = 4
    for fold_id, (_, val_idxs) in enumerate(train_val_splits):
        _, val_loader = get_data_loader(train_df, fold_id, classes=classes, cfg=CFG)
        ov_model = ov_model_list[fold_id]

        lms_batches = []
        for batch in tqdm(val_loader, desc=f"OV infer fold{fold_id}"):
            lms = lms_transform(to_device(batch["wave"], device))
            lms_batches.append(lms.cpu().numpy())

        logits = async_infer_with_order(ov_model, lms_batches, num_requests=16)
        oof_pred_ov[val_idxs] = logits
        oof_pred_ov_rank[val_idxs] = rank_normalize(oof_pred_ov[val_idxs])

    print("auc for raw pred (OV) :", roc_auc_score(labels_arr, oof_pred_ov))
    print("auc for rank pred (OV):", roc_auc_score(labels_arr, oof_pred_ov_rank))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-oof", action="store_true",
                        help="只把 .pt 转 OpenVINO, 不跑 OOF AUC 校验.")
    args = parser.parse_args()
    main(skip_oof=args.skip_oof)
