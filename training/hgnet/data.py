"""标签加载 / soundscape 切片 / 多标签分层分组 K 折
==================================================================

提供训练前的全部数据准备步骤:

1. ``load_label_files``        : 读 ``train.csv`` / ``train_soundscapes_labels.csv``
                                  / ``taxonomy.csv``, 并把 HH:MM:SS 转成秒.
2. ``split_train_soundscapes`` : 把 soundscape 中标注的连续 5s 区间合并 + 切 wav 落盘.
3. ``build_train_dataframe``   : 拼装 train_audio / soundscape 两个标签源, 生成
                                  ``train_df`` (含 234 类的 multi-hot 列) 与 ``labels_arr``.
4. ``MultiLabelStratifiedGroupKFold`` : 多标签分层 + 分组的 K 折切分器,
                                        以 ``audio_id`` 作 group, 防止训练 / 验证泄漏.
"""

from __future__ import annotations

import ast
import gc
import random
import typing as tp
from time import time

import numpy as np
import pandas as pd
import soundfile
from scipy.sparse import coo_matrix
from tqdm import tqdm

from .config import (
    DATA,
    SAMPLING_RATE,
    TRAIN_AUDIO_WAVS,
    TRAIN_SS,
    TRAIN_SS_SPLIT,
)


# ============================================================
#                  1. 标签 / 类别加载
# ============================================================
def load_label_files() -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tp.Dict[str, int], tp.Dict[int, str]]:
    """加载 train.csv / train_soundscapes_labels.csv / taxonomy.csv.

    Returns:
        train_labels        : 单标签长录音表 (含 secondary_labels).
        train_ss_labels     : soundscape 每 5s 区间的标签表 (含 start_sec / end_sec).
        taxonomy            : 类别表, 决定 234 类的顺序.
        label2idx, idx2label: 类别名 <-> 索引 的双向映射.
    """
    train_labels = pd.read_csv(DATA / "train.csv")

    train_ss_labels = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    train_ss_labels = train_ss_labels.drop_duplicates().reset_index(drop=True)

    # 把 HH:MM:SS 转秒, 方便后面按 wav 帧数切片.
    train_ss_labels["start"] = pd.to_datetime(train_ss_labels["start"], format="%H:%M:%S")
    train_ss_labels["end"] = pd.to_datetime(train_ss_labels["end"], format="%H:%M:%S")
    train_ss_labels["start_sec"] = (
        train_ss_labels["start"].dt.minute * 60 + train_ss_labels["start"].dt.second
    )
    train_ss_labels["end_sec"] = (
        train_ss_labels["end"].dt.minute * 60 + train_ss_labels["end"].dt.second
    )

    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    label2idx = {label: idx for idx, label in enumerate(taxonomy.primary_label.values)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    return train_labels, train_ss_labels, taxonomy, label2idx, idx2label


# ============================================================
#              2. soundscape 5s 区间合并 + 切 wav
# ============================================================
def split_train_soundscapes(
    train_ss_labels: pd.DataFrame,
    sampling_rate: int = SAMPLING_RATE,
    write_files: tp.Optional[bool] = None,
) -> pd.DataFrame:
    """合并 soundscape 同文件 + 同 primary_label 的相邻区间, 切片并落盘.

    输出 DataFrame 列: audio_id / filename / primary_label / labels / file_path.

    Args:
        write_files: 是否真正写 wav 切片到磁盘.
            - True  : 正常合并 + 写盘 (单进程预切片用).
            - False : 只做合并 + 建表, **不写盘** (并行进程复用已切好的切片用,
                      避免多进程抢写同一目录互相覆盖).
            - None  : 读环境变量 ``HGNET_SKIP_SPLIT``. 设为 1/true 时等价 False.

    注意: write_files=False 时, 必须保证 ``TRAIN_SS_SPLIT`` 目录里已经有完整切好的
    wav (例如先用 ``python -m hgnet.data`` 跑一遍预切片). 否则训练时
    dataloader 读不到文件会报错.
    """
    import os
    if write_files is None:
        write_files = os.environ.get("HGNET_SKIP_SPLIT", "0") not in {"1", "true", "True"}

    if write_files:
        TRAIN_SS_SPLIT.mkdir(exist_ok=True, parents=True)
    else:
        print("[split] HGNET_SKIP_SPLIT — 只建表不写盘, 复用已存在的切片目录: "
              f"{TRAIN_SS_SPLIT}")

    train_ss_ai_list: tp.List[str] = []
    train_ss_fn_list: tp.List[str] = []
    train_ss_pl_list: tp.List[str] = []

    # 用第一行作为"正在合并"片段的初始值.
    tmp_in_fn, tmp_pl, tmp_start, tmp_end = train_ss_labels.loc[
        0, ["filename", "primary_label", "start_sec", "end_sec"]
    ].values
    # 只在需要写盘时才读原始 soundscape wav (建表本身用不到波形).
    tmp_wave = None
    if write_files:
        tmp_wave, _ = soundfile.read(TRAIN_SS / tmp_in_fn, dtype="float32")

    def _emit(ai: str, out_fn: str, pl: str, start: int, end: int) -> None:
        if write_files:
            wave_seg = tmp_wave[start * sampling_rate : end * sampling_rate]
            soundfile.write(
                str(TRAIN_SS_SPLIT / out_fn), wave_seg, samplerate=sampling_rate,
                format="wav", subtype="FLOAT",
            )
        train_ss_ai_list.append(ai)
        train_ss_fn_list.append(out_fn)
        train_ss_pl_list.append(pl)

    rows = train_ss_labels.loc[1:, ["filename", "primary_label", "start_sec", "end_sec"]].values
    for in_fn, pl, start, end in tqdm(rows, desc="merge soundscape segments"):
        # 同文件且同 primary_label, 把 end 继续后推一格合并.
        if in_fn == tmp_in_fn and pl == tmp_pl:
            tmp_end = end
            continue

        # 否则把刚才合并的片段落盘.
        ai = tmp_in_fn.split(".")[0]
        out_fn = f"{ai}_{tmp_start}-{tmp_end}.wav"
        _emit(ai, out_fn, tmp_pl, tmp_start, tmp_end)

        # 切换到当前区间作为新的合并起点.
        tmp_pl, tmp_start, tmp_end = pl, start, end
        if in_fn != tmp_in_fn:
            tmp_in_fn = in_fn
            if write_files:
                tmp_wave, _ = soundfile.read(TRAIN_SS / tmp_in_fn, dtype="float32")
    else:
        # 循环正常结束时, 把最后一个挂起的片段也落盘.
        # 原 notebook 这里写成了 f"{ai}_{start}-{end}.wav", 与合并起点 tmp_start/tmp_end
        # 不一致, 是个隐性 bug, 这里修正成 tmp_start/tmp_end.
        ai = tmp_in_fn.split(".")[0]
        out_fn = f"{ai}_{tmp_start}-{tmp_end}.wav"
        _emit(ai, out_fn, tmp_pl, tmp_start, tmp_end)

    train_ss_labels_merged = pd.DataFrame({
        "audio_id": train_ss_ai_list,
        "filename": train_ss_fn_list,
        "primary_label": train_ss_pl_list,
        # soundscape 这里 labels 就当成 primary_label
        # (原始 csv 里 primary_label 已经是 ";" 分隔的组合, 见 train_soundscapes_labels.csv).
        "labels": train_ss_pl_list,
    })
    train_ss_labels_merged["file_path"] = [
        str(TRAIN_SS_SPLIT / fn) for fn in train_ss_labels_merged["filename"].values
    ]
    return train_ss_labels_merged


# ============================================================
#              3. 组装 train_df + multi-hot 标签
# ============================================================
def build_train_dataframe(
    train_labels: pd.DataFrame,
    train_ss_labels_merged: pd.DataFrame,
    label2idx: tp.Dict[str, int],
) -> tp.Tuple[pd.DataFrame, np.ndarray]:
    """合并 train_audio + soundscape, 生成统一 train_df 与 multi-hot 标签矩阵."""
    from .config import USE_OGG_DIRECT
    audio_ext = ".ogg" if USE_OGG_DIRECT else ".wav"

    # ---- 1) primary_label -> 数据集根目录 的映射 ----
    pl2wp: tp.Dict[str, "Path"] = {}
    for wav_path in TRAIN_AUDIO_WAVS:
        if not wav_path.exists():
            continue
        for pl_dir in sorted(wav_path.iterdir()):
            if pl_dir.is_dir():
                pl2wp[pl_dir.name] = wav_path

    # ---- 2) train_audio 的整理 ----
    train_labels_merged = pd.DataFrame()
    # audio_id 取 "XCxxx" 部分: train.csv 的 filename 是 "primary_label/XCxxx.ogg",
    # split(".") 后 explode, 偶数索引就是 "primary_label/XCxxx".
    train_labels_merged["audio_id"] = train_labels["filename"].str.split(".").explode().values[0::2]
    train_labels_merged["filename"] = train_labels["filename"]
    train_labels_merged["primary_label"] = train_labels["primary_label"]
    train_labels_merged["labels"] = [
        ";".join([pl] + ast.literal_eval(sls))
        for pl, sls in train_labels[["primary_label", "secondary_labels"]].values
    ]
    # USE_OGG_DIRECT=True 时直接拼 .ogg, 否则拼 .wav (跟 ttahara 数据集对齐).
    # 注: train.csv 的 filename 形如 "primary_label/XCxxx.ogg",
    # split('.')[0] 拿到 "primary_label/XCxxx" -> 加后缀 -> 跟 pl2wp 拼绝对路径.
    train_labels_merged["file_path"] = [
        str(pl2wp[pl] / f"{fn.split('.')[0]}{audio_ext}")
        for fn, pl in train_labels_merged[["filename", "primary_label"]].values
    ]

    # ---- 3) 拼接两个源 ----
    train_df = pd.concat(
        [train_labels_merged, train_ss_labels_merged],
        axis=0,
        ignore_index=True,
    )

    # ---- 4) multi-hot 标签矩阵 (n_samples, n_classes) ----
    labels_arr = np.zeros((len(train_df), len(label2idx)), dtype=np.float32)
    for idx, labels in enumerate(train_df["labels"].values):
        for l in labels.split(";"):
            labels_arr[idx, label2idx[l]] = 1

    # ---- 5) 把每类的 0/1 列拼到 train_df, 方便后面 train_df[CLASSES] 直接取 ----
    label_df = pd.DataFrame(labels_arr, columns=list(label2idx.keys()))
    train_df = pd.concat([train_df, label_df], axis=1)

    return train_df, labels_arr


# ============================================================
#         4. 多标签 + 分组 的 Stratified K-Fold
# ============================================================
class MultiLabelStratifiedGroupKFold:
    """多标签分层 + 分组的 K 折切分.

    - **分组 (group)**: 用 audio_id 作 group, 保证同一原始音频的所有片段同一折,
      避免训练 / 验证泄漏.
    - **分层 (stratified)**: 通过贪心策略把每个 group 分到 "使各折的类别分布方差最小"
      的那一折, 让 234 类在各折之间尽量均匀.

    参考: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    """

    def __init__(self, n_splits: int, random_state: int):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, label_arr: np.ndarray, gid_arr: np.ndarray):
        """根据 multi-hot label 与 group id 生成 K 折 (train_idx, val_idx).

        Args:
            label_arr: (n_train, n_class) 多热标签矩阵.
            gid_arr  : (n_train,) 每条样本对应的 group id.
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        start_time = time()
        n_train, n_class = label_arr.shape
        gid_unique = sorted(set(gid_arr))
        n_group = len(gid_unique)

        # group_id -> 0..n_group-1 的连续整数 id (aid).
        gid2aid = dict(zip(gid_unique, range(n_group)))
        aid_arr = np.vectorize(lambda x: gid2aid[x])(gid_arr)

        # 整体每类样本数, 用来评估各折占比.
        cnts_by_class = label_arr.sum(axis=0)  # (n_class,)

        # 每个 group 在每类上的样本数: (n_group, n_class).
        col, row = np.array(sorted(enumerate(aid_arr), key=lambda x: x[1])).T
        cnts_by_group = (
            coo_matrix((np.ones(len(label_arr)), (row, col)))
            .dot(coo_matrix(label_arr))
            .toarray()
            .astype(int)
        )
        del col, row

        # 每折当前累计的每类样本数.
        cnts_by_fold = np.zeros((self.n_splits, n_class), int)
        groups_by_fold: tp.List[tp.List[int]] = [[] for _ in range(self.n_splits)]

        # 先打乱 group, 再按 "每类样本数的 std" 从大到小处理 (大方差先放, 优先决定).
        group_and_cnts = list(enumerate(cnts_by_group))
        np.random.shuffle(group_and_cnts)
        print("finished preparation", time() - start_time)

        for aid, cnt_by_g in sorted(group_and_cnts, key=lambda x: -np.std(x[1])):
            best_fold, min_eval = None, None
            for fid in range(self.n_splits):
                # 试着把这个 group 放到第 fid 折, 看类别比例 std 是否最小.
                cnts_by_fold[fid] += cnt_by_g
                fold_eval = (cnts_by_fold / cnts_by_class).std(axis=0).mean()
                cnts_by_fold[fid] -= cnt_by_g
                if min_eval is None or fold_eval < min_eval:
                    min_eval, best_fold = fold_eval, fid
            cnts_by_fold[best_fold] += cnt_by_g
            groups_by_fold[best_fold].append(aid)
        print("finished assignment.", time() - start_time)

        gc.collect()
        idx_arr = np.arange(n_train)
        for fid in range(self.n_splits):
            val_groups = groups_by_fold[fid]
            val_indexs_bool = np.isin(aid_arr, val_groups)
            train_indexs = idx_arr[~val_indexs_bool]
            val_indexs = idx_arr[val_indexs_bool]

            print(
                f"[fold {fid}] "
                f"n_group: (train, val) = ({n_group - len(val_groups)}, {len(val_groups)}) "
                f"n_sample: (train, val) = ({len(train_indexs)}, {len(val_indexs)})"
            )
            yield train_indexs, val_indexs


# ============================================================
#      5. 预切片 CLI: 多终端并行训练前先单独跑一次
# ============================================================
def prepare_split_once(verify_only: bool = False) -> int:
    """单独把 soundscape 切片落盘一次, 供后续多个训练进程共享复用.

    多终端并行训练时的推荐流程:

        # 1) 先单进程切片 (只需一次)
        python -m hgnet.data

        # 2) 之后每个终端训练都设 HGNET_SKIP_SPLIT=1, 直接复用切片, 不再写盘
        HGNET_SKIP_SPLIT=1 HGNET_MODEL_NAME=hgnetv2_b0.ssld_stage2_ft_in1k \
            python -m hgnet.train

    Args:
        verify_only: 只统计已有切片数量, 不重新写盘.

    Returns:
        切片目录下的 wav 数量.
    """
    train_labels, train_ss_labels, taxonomy, label2idx, _ = load_label_files()

    if verify_only:
        n = len(list(TRAIN_SS_SPLIT.glob("*.wav"))) if TRAIN_SS_SPLIT.exists() else 0
        print(f"[verify] {TRAIN_SS_SPLIT} 现有 {n} 个 wav 切片")
        return n

    merged = split_train_soundscapes(train_ss_labels, write_files=True)
    n_expected = len(merged)
    n_actual = len(list(TRAIN_SS_SPLIT.glob("*.wav")))
    print(f"[prepare] 合并出 {n_expected} 个片段, 落盘后目录共 {n_actual} 个 wav")
    if n_actual < n_expected:
        print(f"[prepare] 警告: 落盘数 {n_actual} < 期望 {n_expected}, "
              f"可能有文件名重名覆盖, 请检查.")
    print(f"[prepare] 完成. 之后训练设 HGNET_SKIP_SPLIT=1 即可复用.")
    return n_actual


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="soundscape 预切片 (多终端并行训练前跑一次)")
    p.add_argument("--verify", action="store_true",
                   help="只校验已有切片数量, 不重新写盘")
    args = p.parse_args()
    prepare_split_once(verify_only=args.verify)
