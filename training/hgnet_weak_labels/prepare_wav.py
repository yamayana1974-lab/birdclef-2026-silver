"""ogg -> wav 批量预转换脚本
==================================

把比赛官方 ``train_audio/<primary_label>/<XCxxx>.ogg`` 批量解码成 wav,
等价于复现 ttahara 在 Kaggle 上传的 4 份预转 wav 数据集
(`birdclef2026-train-audio-wav-00..03`).

核心转换函数原样使用 ttahara 在 Kaggle 讨论区给出的版本:

    def convert_ogg_to_wav(ogg_path, output_dir):
        wave, sr = soundfile.read(ogg_path, dtype="float32")
        wav_path = output_dir / f"{ogg_path.stem}.wav"
        soundfile.write(wav_path, wave, samplerate=sr,
                        format="wav", subtype="FLOAT")

外层批处理额外做了几件事 (不改变单文件转换逻辑):

- 断点续跑: 已存在的目标 wav 默认跳过, ``--overwrite`` 强制重转.
- 原子写入: 先写 ``*.wav.tmp`` 再 ``replace``, 中途 Ctrl-C 不会留半截 wav.
- 错误隔离: 单文件抛异常不影响其它文件, 结尾汇总失败列表.
- 可选 ``--split N`` 按 primary_label 字母序均分到 N 份, ``--split 4``
  即可复刻 ttahara 的 ``wav-00..03`` 四份布局.
- 可选 ``--force-sr`` 在源采样率与目标不一致时做高质量 polyphase 重采样.
  默认沿用源 sr (官方 BirdCLEF ogg 都是 32 kHz, 不需要重采样).
- joblib 并行 + tqdm 进度条 (用 ``return_as="generator"`` 让进度条真实推进).

典型用法
--------

1) 最简单, 输出到一个目录 (本地训练最方便):

    cd /path/to/birdclef-2026-repo/training
    python -m hgnet_weak_labels.prepare_wav \\
        --output-root ./data/wav-output --split 1 --n-jobs 16

   产物::

       ./data/wav-output/birdclef2026-train-audio-wav/<primary_label>/XCxxx.wav

   然后在 ``config.py`` 里把 ``TRAIN_AUDIO_WAVS`` 改成::

       TRAIN_AUDIO_WAVS = [Path("./data/wav-output/birdclef2026-train-audio-wav")]

2) 严格复刻 ttahara 的 4 份布局 (路径无需改 config):

    python -m hgnet_weak_labels.prepare_wav \\
        --output-root <INPUT>/datasets/ttahara --split 4 --n-jobs 16

   产物::

       <INPUT>/datasets/ttahara/birdclef2026-train-audio-wav-{00..03}/<pl>/XCxxx.wav

3) Smoke test (只转前 32 个, 检查路径配置是否对):

    python -m hgnet_weak_labels.prepare_wav --limit 32 --n-jobs 4

可选项
------

- ``--input-dir``    : 源 ogg 根目录, 默认 ``config.TRAIN_AUDIO``.
- ``--output-root``  : 输出根目录, 默认 ``<INPUT>/datasets/local``.
- ``--dataset-prefix``: 输出子目录前缀, 默认 ``birdclef2026-train-audio-wav``.
- ``--split N``      : 按类目录字母序均分 N 份. 默认 1.
- ``--force-sr SR``  : 强制重采样到 SR (Hz). 默认 None 沿用源 sr.
- ``--n-jobs J``     : joblib 并发数. 默认 8.
- ``--overwrite``    : 覆盖已存在 wav. 默认跳过.
- ``--limit N``      : 仅转前 N 个 ogg, 用于试跑.
"""

from __future__ import annotations

import argparse
import sys
import typing as tp
from collections import Counter
from math import gcd
from pathlib import Path

import soundfile
from joblib import Parallel, delayed
from tqdm import tqdm

from .config import INPUT, SAMPLING_RATE, TRAIN_AUDIO


# ============================================================
#                       1. 单文件转换
# ============================================================
def _resample(wave, src_sr: int, dst_sr: int):
    """整数比 polyphase 重采样. 只在 --force-sr 与源 sr 不一致时用到, 懒导入 scipy."""
    from scipy.signal import resample_poly

    g = gcd(src_sr, dst_sr)
    return resample_poly(wave, dst_sr // g, src_sr // g).astype("float32", copy=False)


def convert_one(
    ogg_path: Path,
    out_path: Path,
    force_sr: tp.Optional[int] = None,
    overwrite: bool = False,
) -> tp.Tuple[Path, tp.Optional[str]]:
    """ogg -> wav. 返回 (out_path, err_or_None).

    核心三行就是 ttahara 给的官方写法, 其余都是健壮性处理:
    断点续跑 / 原子写入 / 异常捕获 / 可选重采样.
    """
    if out_path.exists() and not overwrite:
        return out_path, None
    try:
        wave, sr = soundfile.read(str(ogg_path), dtype="float32")

        if force_sr is not None and sr != force_sr:
            wave = _resample(wave, sr, force_sr)
            sr = force_sr

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        soundfile.write(
            str(tmp_path), wave, samplerate=sr, format="wav", subtype="FLOAT"
        )
        tmp_path.replace(out_path)
        return out_path, None
    except Exception as e:
        return out_path, f"{type(e).__name__}: {e}"


# ============================================================
#                  2. 文件扫描 / split 分配
# ============================================================
def discover_ogg_files(input_dir: Path) -> tp.List[Path]:
    """递归找出 input_dir 下所有 .ogg 文件 (按相对路径稳定排序)."""
    files = sorted(input_dir.rglob("*.ogg"))
    if not files:
        raise FileNotFoundError(
            f"在 {input_dir} 下没找到任何 .ogg 文件, 请确认 --input-dir 是否正确."
        )
    return files


def _split_by_class(
    ogg_files: tp.List[Path], input_dir: Path, n_split: int
) -> tp.Dict[Path, int]:
    """按一级子目录 (primary_label) 字母序均匀切分到 [0, n_split) 份.

    保证同一 primary_label 的 wav 全部落到同一份, 与 ttahara 的布局一致.
    """
    rels = [p.relative_to(input_dir) for p in ogg_files]
    classes = sorted({r.parts[0] for r in rels if len(r.parts) > 1})
    if not classes:
        raise ValueError(
            f"{input_dir} 下没有 <primary_label>/*.ogg 的二级目录结构, "
            f"无法按类切分; 如果就是想要单目录, 请用 --split 1."
        )
    cls2split = {cls: i * n_split // len(classes) for i, cls in enumerate(classes)}
    return {
        p: cls2split[r.parts[0]]
        for p, r in zip(ogg_files, rels)
        if len(r.parts) > 1
    }


# ============================================================
#                       3. 主入口
# ============================================================
def main(argv: tp.Optional[tp.List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="批量把 train_audio/*.ogg 预转换成 wav (复刻 ttahara 的 4 份数据集).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TRAIN_AUDIO,
        help="ogg 输入根目录. 期望布局: <input>/<primary_label>/<XCxxx>.ogg",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=INPUT / "datasets" / "local",
        help="输出根目录. 最终落到 <output-root>/<dataset-prefix>[-NN]/<pl>/<XCxxx>.wav",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="birdclef2026-train-audio-wav",
        help="输出子目录前缀. 默认与 ttahara 对齐.",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="按 primary_label 字母序均分到几份. 1=单目录; 4=复刻 ttahara 布局.",
    )
    parser.add_argument(
        "--force-sr",
        type=int,
        default=None,
        help=f"强制重采样到该 sr (Hz). 不指定则沿用源 sr (官方 BirdCLEF "
             f"通常已经是 {SAMPLING_RATE} Hz, 不需要重采样).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="joblib 并发进程数. CPU 多 / IO 快可以调到 16~32.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 wav. 默认会跳过.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅转换前 N 个 ogg, 用于 smoke test.",
    )
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    n_split: int = max(1, args.split)
    if not input_dir.exists():
        print(f"[ERR] 输入目录不存在: {input_dir}", file=sys.stderr)
        return 1

    # ---- 1) 扫描 ogg 文件 ----
    print(f"[INFO] 扫描 ogg 文件: {input_dir}")
    ogg_files = discover_ogg_files(input_dir)
    if args.limit:
        ogg_files = ogg_files[: args.limit]
    print(f"[INFO] 待处理 ogg 文件: {len(ogg_files)} 个")

    # ---- 2) 计算每个 ogg 对应的输出路径 ----
    if n_split == 1:
        out_root_for: tp.Dict[Path, Path] = {
            p: args.output_root / args.dataset_prefix for p in ogg_files
        }
    else:
        p2split = _split_by_class(ogg_files, input_dir, n_split)
        out_root_for = {
            p: args.output_root / f"{args.dataset_prefix}-{p2split[p]:02d}"
            for p in ogg_files
        }
        for sp_idx, n in sorted(Counter(p2split.values()).items()):
            print(f"[INFO] split-{sp_idx:02d}: {n} 个 ogg")

    def _task(ogg_path: Path) -> tp.Tuple[Path, Path]:
        rel = ogg_path.relative_to(input_dir)
        out_path = out_root_for[ogg_path] / rel.with_suffix(".wav")
        return ogg_path, out_path

    tasks = [_task(p) for p in ogg_files]

    # ---- 3) 断点续跑: 过滤掉已经存在的 wav ----
    if args.overwrite:
        pending = tasks
    else:
        pending = [(s, t) for s, t in tasks if not t.exists()]
        skipped = len(tasks) - len(pending)
        if skipped:
            print(f"[INFO] 跳过已转换的 wav: {skipped} 个 (加 --overwrite 强制重转)")

    if not pending:
        print("[DONE] 全部已转换完毕, 无事可做.")
        return 0

    print(
        f"[INFO] 即将转换 {len(pending)} 个 ogg | n_jobs={args.n_jobs} "
        f"| force_sr={args.force_sr or 'keep'}"
    )

    # ---- 4) joblib 并行转换 ----
    # return_as="generator" 让结果一边出一边喂 tqdm, 进度条能真实推进 (joblib>=1.3).
    gen = Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(
        delayed(convert_one)(
            src, dst, force_sr=args.force_sr, overwrite=args.overwrite
        )
        for src, dst in pending
    )
    results: tp.List[tp.Tuple[Path, tp.Optional[str]]] = list(
        tqdm(gen, total=len(pending), desc="ogg -> wav", unit="file")
    )

    # ---- 5) 失败汇总 ----
    errs = [(p, e) for p, e in results if e is not None]
    n_ok = len(results) - len(errs)
    print(f"[DONE] 成功 {n_ok} | 失败 {len(errs)}")
    if errs:
        print("[ERR] 失败列表 (最多打印 20 条):")
        for p, e in errs[:20]:
            print(f"  - {p}: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
