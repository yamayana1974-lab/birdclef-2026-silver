# -*- coding: utf-8 -*-
"""perch_inference.py — Perch 推理 + 12-window cache (Model_5).

对应 cell_07 第 271-568 行 (ONNX/TF backend, ``run_perch``, cache 查找/构建).
和 Model_2 版本最大区别: ONNX 模型自动从 ``INPUT_ROOT.glob('**/perch_v2_no_dft*.onnx')``
找, fallback 到 ``perch_v2*.onnx``; cache 多套候选 (``EXTERNAL_CACHE_DIRS``).
"""
from __future__ import annotations

import concurrent.futures
import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from .config import (
    CFG, FILE_SAMPLES, INPUT_ROOT, MODEL_DIR, N_WINDOWS, SR, WINDOW_SAMPLES, WORK_DIR,
)
from .data import parse_fname


SCORE_KEYS = ("scores", "sc", "logits", "perch_scores", "preds", "arr_0")
EMB_KEYS   = ("embs",  "emb", "embeddings", "features", "perch_embs", "arr_1")


# =============================================================================
# Backend
# =============================================================================
class PerchBackend:

    def __init__(self, onnx_available: bool):
        # 尝试找 ONNX
        onnx_no_dft = next(INPUT_ROOT.glob("**/perch_v2_no_dft*.onnx"), None)
        onnx_any    = next(INPUT_ROOT.glob("**/perch_v2*.onnx"), None)
        self.onnx_path = onnx_no_dft or onnx_any

        self.use_onnx = bool(onnx_available) and self.onnx_path is not None and self.onnx_path.exists()

        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_out_map = None
        self.tf_infer_fn = None

        if self.use_onnx:
            # ONNX 后端优先: 完全不加载 TF SavedModel (没装 TensorFlow 也能跑).
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            self.onnx_session    = ort.InferenceSession(
                str(self.onnx_path), sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.onnx_out_map    = {
                o.name: i for i, o in enumerate(self.onnx_session.get_outputs())
            }
            print(f"[perch] using ONNX backend: {self.onnx_path.name}")
        elif MODEL_DIR.exists():
            # 没有可用 ONNX 时才回退 TF SavedModel; TF 只在此分支 import.
            import tensorflow as tf
            birdclassifier = tf.saved_model.load(str(MODEL_DIR))
            self.tf_infer_fn = birdclassifier.signatures["serving_default"]
            print("[perch] using TF SavedModel backend")
        else:
            raise FileNotFoundError(
                "No usable Perch backend: attach Perch ONNX or TF SavedModel."
            )

    def infer(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_onnx:
            outs = self.onnx_session.run(None, {self.onnx_input_name: x})
            logits = outs[self.onnx_out_map["label"]].astype(np.float32, copy=False)
            emb    = outs[self.onnx_out_map["embedding"]].astype(np.float32, copy=False)
        else:
            import tensorflow as tf
            outputs = self.tf_infer_fn(inputs=tf.convert_to_tensor(x))
            logits  = outputs["label"].numpy().astype(np.float32, copy=False)
            emb     = outputs["embedding"].numpy().astype(np.float32, copy=False)
        return logits, emb


# =============================================================================
# Audio + run_perch (notebook 第 380-441 行)
# =============================================================================
def read_60s(path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    return y


def run_perch(
    paths,
    backend: PerchBackend,
    mapping: Dict,
    n_classes: int,
    batch_files: int = 16,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    paths   = [Path(p) for p in paths]
    n_rows  = len(paths) * N_WINDOWS

    row_ids   = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites     = np.empty(n_rows, dtype=object)
    hours     = np.zeros(n_rows, dtype=np.int16)
    scores    = np.zeros((n_rows, n_classes), dtype=np.float32)
    embs      = np.zeros((n_rows, 1536),      dtype=np.float32)

    wr = 0
    itr = (
        tqdm(range(0, len(paths), batch_files), desc="Perch")
        if verbose else range(0, len(paths), batch_files)
    )

    MAPPED_POS    = mapping["MAPPED_POS"]
    MAPPED_BC_IDX = mapping["MAPPED_BC_IDX"]
    proxy_map     = mapping["proxy_map"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
        next_paths   = paths[0:batch_files]
        future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

        for start in itr:
            batch_paths = next_paths
            batch_n     = len(batch_paths)
            batch_audio = [f.result() for f in future_audio]

            next_start = start + batch_files
            if next_start < len(paths):
                next_paths   = paths[next_start:next_start + batch_files]
                future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

            x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            br = wr
            for bi, path in enumerate(batch_paths):
                y    = batch_audio[bi]
                meta = parse_fname(path.name)
                stem = path.stem
                x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids  [wr:wr + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr:wr + N_WINDOWS] = path.name
                sites    [wr:wr + N_WINDOWS] = meta["site"]
                hours    [wr:wr + N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS

            logits, emb = backend.infer(x)
            scores[br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
            embs  [br:wr]             = emb

            for pos_idx, bc_idxs in proxy_map.items():
                bc_arr = np.array(bc_idxs, dtype=np.int32)
                scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)

            del x, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({
        "row_id":   row_ids,
        "filename": filenames,
        "site":     sites,
        "hour_utc": hours,
    })
    return meta_df, scores, embs


# =============================================================================
# Cache (notebook 第 444-568 行)
# =============================================================================
EXTERNAL_CACHE_DIRS = [
    Path("/kaggle/input/notebooks/vyankteshdwivedi/notebook1b25083f0d"),
    Path("/kaggle/input/datasets/jaejohn/perch-meta"),
]
CACHE_NAME_PAIRS = [
    ("perch_meta.parquet",      "perch_arrays.npz"),
    ("full_perch_meta.parquet", "full_perch_arrays.npz"),
]
CACHE_META_LOCAL = WORK_DIR / "perch_meta.parquet"
CACHE_NPZ_LOCAL  = WORK_DIR / "perch_arrays.npz"


def _find_external_cache():
    roots = [d for d in EXTERNAL_CACHE_DIRS if d.exists()]
    roots.append(INPUT_ROOT)
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        for meta_name, npz_name in CACHE_NAME_PAIRS:
            meta = root / meta_name
            npz  = root / npz_name
            if meta.exists() and npz.exists():
                print(f"[perch] using cache: {meta}\n         {npz}")
                return meta, npz
        for meta_name, npz_name in CACHE_NAME_PAIRS:
            for meta in sorted(root.rglob(meta_name)):
                npz = meta.parent / npz_name
                if npz.exists():
                    print(f"[perch] using cache: {meta}\n         {npz}")
                    return meta, npz
    return None, None


def _pick_array(arr, candidates: Tuple[str, ...], shape_hint_cols: int):
    for k in candidates:
        if k in arr.files:
            v = arr[k]
            if getattr(v, "ndim", 0) == 2 and v.shape[1] == shape_hint_cols:
                return v, k
            print(f"[perch] skip cache key {k!r}: shape={getattr(v, 'shape', None)}, "
                  f"expected second dim={shape_hint_cols}")
    for k in arr.files:
        v = arr[k]
        if getattr(v, "ndim", 0) == 2 and v.shape[1] == shape_hint_cols:
            return v, k
    raise KeyError(f"None of {candidates} found in npz. Available keys: {arr.files}")


def load_or_build_cache(
    backend: PerchBackend,
    mapping: Dict,
    full_files: List[str],
    base_dir: Path,
    n_classes: int,
    primary_labels: List[str],
) -> Tuple[Path, Path]:
    """返回 ``(CACHE_META, CACHE_NPZ)`` 路径."""
    ext_meta, ext_npz = _find_external_cache()
    if ext_meta is not None:
        return ext_meta, ext_npz
    if CACHE_META_LOCAL.exists() and CACHE_NPZ_LOCAL.exists():
        print(f"[perch] using local cache: {WORK_DIR}")
        return CACHE_META_LOCAL, CACHE_NPZ_LOCAL

    print(f"[perch] building cache from {len(full_files)} training files...")
    train_paths = [base_dir / "train_soundscapes" / fn for fn in full_files]
    train_paths = [p for p in train_paths if p.exists()]
    t0 = time.time()
    meta_built, sc_built, emb_built = run_perch(
        train_paths, backend, mapping, n_classes,
        batch_files=CFG["batch_files"], verbose=True,
    )
    print(f"[perch] perch pass done in {time.time() - t0:.1f}s  "
          f"scores={sc_built.shape}  embs={emb_built.shape}")

    meta_built.to_parquet(CACHE_META_LOCAL)
    np.savez(
        CACHE_NPZ_LOCAL,
        scores=sc_built.astype(np.float32),
        embs=emb_built.astype(np.float32),
        primary_labels=np.array(primary_labels),
    )
    print(f"[perch] cache saved to {WORK_DIR}")
    return CACHE_META_LOCAL, CACHE_NPZ_LOCAL


def load_cache_arrays(
    cache_meta: Path,
    cache_npz: Path,
    full_rows: pd.DataFrame,
    Y_SC: np.ndarray,
    n_classes: int,
    primary_labels: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """读 cache + 跟 full_rows 对齐, 返回 ``(meta_tr, sc_tr, emb_tr, Y_FULL_aligned)``."""
    meta_tr = pd.read_parquet(cache_meta)
    arr     = np.load(cache_npz)
    sc_tr_raw,  _ = _pick_array(arr, SCORE_KEYS, n_classes)
    emb_tr_raw, _ = _pick_array(arr, EMB_KEYS,   1536)
    sc_tr  = sc_tr_raw.astype(np.float32)
    emb_tr = emb_tr_raw.astype(np.float32)

    if "primary_labels" in arr.files:
        if arr["primary_labels"].tolist() != primary_labels:
            print("[perch] WARNING: cached primary_labels differ from current")
        else:
            print("[perch] primary_labels schema OK")

    if "row_id" not in meta_tr.columns:
        if "end_sec" in meta_tr.columns:
            end_sec = meta_tr["end_sec"].astype(int)
        elif "window_idx" in meta_tr.columns:
            from .config import WINDOW_SEC
            end_sec = (meta_tr["window_idx"].astype(int) + 1) * WINDOW_SEC
        else:
            from .config import WINDOW_SEC
            assert len(meta_tr) % N_WINDOWS == 0
            end_sec = np.tile(
                np.arange(WINDOW_SEC, WINDOW_SEC * N_WINDOWS + 1, WINDOW_SEC),
                len(meta_tr) // N_WINDOWS,
            )
        meta_tr["row_id"] = (
            meta_tr["filename"].str.replace(".ogg", "", regex=False)
            + "_" + end_sec.astype(str)
        )
    if "end_sec" not in meta_tr.columns:
        meta_tr["end_sec"] = meta_tr["row_id"].str.rsplit("_", n=1).str[-1].astype(int)
    assert len(meta_tr) == sc_tr.shape[0] == emb_tr.shape[0]
    assert meta_tr["row_id"].is_unique

    meta_tr = meta_tr.copy()
    meta_tr["_cache_pos"] = np.arange(len(meta_tr))
    order   = meta_tr.sort_values(["filename", "end_sec"])["_cache_pos"].to_numpy()
    meta_tr = meta_tr.iloc[order].drop(columns=["_cache_pos"]).reset_index(drop=True)
    sc_tr   = sc_tr[order]
    emb_tr  = emb_tr[order]

    row_id_to_index = full_rows.set_index("row_id")["index"]
    missing_rows    = set(meta_tr["row_id"]) - set(row_id_to_index.index)
    if missing_rows:
        raise RuntimeError(f"Cache has {len(missing_rows)} row_ids not in labeled set.")
    Y_FULL_aligned = Y_SC[row_id_to_index.loc[meta_tr["row_id"]].to_numpy()]
    print(f"[perch] sc_tr={sc_tr.shape}  emb_tr={emb_tr.shape}  "
          f"Y_FULL_aligned={Y_FULL_aligned.shape}")
    return meta_tr, sc_tr, emb_tr, Y_FULL_aligned
