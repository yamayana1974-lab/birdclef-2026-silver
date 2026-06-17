# -*- coding: utf-8 -*-
"""build_fake_base.py — 把 unlabeled train_soundscapes 伪装成 test_soundscapes.

Model_5 的 main.py 把 ``test_paths`` 写死成
``sorted((base_dir / "test_soundscapes").glob("*.ogg"))``. 我们不改它的源码,
而是建一个临时目录, 用硬链接把 unlabeled .ogg 复制进 test_soundscapes/,
其它必要文件 (sample_submission.csv / taxonomy.csv / train_soundscapes_labels.csv /
train_soundscapes/) 也都链接过来. 然后设环境变量 ``BIRDCLEF_BASE=fake_base``,
原 pipeline 就以为这就是真比赛目录.

Linux: 用 os.link 硬链接 (不占额外空间).
Windows: 退化成软链接, 需要管理员权限或 Developer Mode; 否则直接 copy (耗空间).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd


# ---- 文件 link / copy 兼容层 ---------------------------------------------------

def _hardlink_or_copy(src: Path, dst: Path) -> None:
    """优先硬链接, 失败则 copy. Windows 上 os.link 多半没权限, 直接 copy."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _link_dir(src: Path, dst: Path) -> None:
    """Linux: 用 symlink 整目录. Windows / 失败 fallback: 单文件硬链接."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    # fallback: 逐文件硬链
    dst.mkdir(parents=True, exist_ok=True)
    for sub in src.iterdir():
        if sub.is_file():
            _hardlink_or_copy(sub, dst / sub.name)


# ---- 主函数 -------------------------------------------------------------------

def list_unlabeled_files(comp_dir: Path) -> List[str]:
    """返回 train_soundscapes/ 中**未出现在 train_soundscapes_labels.csv** 的文件名."""
    labels_csv = comp_dir / "train_soundscapes_labels.csv"
    if not labels_csv.exists():
        # 比赛目录可能只放了 _full / _partial 拆分版
        full = comp_dir / "train_soundscapes_labels_full.csv"
        part = comp_dir / "train_soundscapes_labels_partial.csv"
        labeled: Set[str] = set()
        for p in (full, part):
            if p.exists():
                labeled.update(pd.read_csv(p)["filename"].unique())
    else:
        labeled = set(pd.read_csv(labels_csv)["filename"].unique())

    sc_dir = comp_dir / "train_soundscapes"
    if not sc_dir.exists():
        raise FileNotFoundError(f"未找到 {sc_dir}")

    all_files = sorted(p.name for p in sc_dir.glob("*.ogg"))
    unlabeled = [fn for fn in all_files if fn not in labeled]
    print(f"[fake_base] total soundscapes: {len(all_files)}  "
          f"labeled: {len(labeled)}  unlabeled: {len(unlabeled)}")
    return unlabeled


def build_fake_base(
    comp_dir: Path,
    fake_base: Path,
    unlabeled_subset: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    offset: int = 0,
) -> Path:
    """构造 fake_base 目录.

    Args:
        comp_dir: 真实 birdclef-2026 目录
        fake_base: 输出目录 (会清空重建)
        unlabeled_subset: 只把这些文件当 test (不传 = 用全部 unlabeled)
        max_files: debug 用, 只链接前 N 个文件
        offset: 从第 offset 个 unlabeled 文件开始切片 (用于分批跑)

    Returns:
        fake_base 路径
    """
    comp_dir = Path(comp_dir).resolve()
    fake_base = Path(fake_base).resolve()

    # 1) 清空已有 fake_base
    if fake_base.exists():
        # 不要乱删用户的真实目录, 只在它跟 comp_dir 不一样时清理
        if fake_base.resolve() == comp_dir.resolve():
            raise RuntimeError(
                f"fake_base ({fake_base}) 跟真实 comp_dir 是同一个目录, 拒绝清空."
            )
        shutil.rmtree(fake_base, ignore_errors=True)
    fake_base.mkdir(parents=True, exist_ok=True)

    # 2) 链接元数据文件
    for fn in (
        "sample_submission.csv",
        "taxonomy.csv",
        "train_soundscapes_labels.csv",
        "train_soundscapes_labels_full.csv",
        "train_soundscapes_labels_partial.csv",
        "train.csv",
        "recording_location.txt",
    ):
        src = comp_dir / fn
        if src.exists():
            _hardlink_or_copy(src, fake_base / fn)

    # 3) 链接 train_soundscapes (整目录, ProtoSSM cache 用)
    _link_dir(comp_dir / "train_soundscapes", fake_base / "train_soundscapes")
    # train_audio 不需要 (我们只跑 inference, 不训练)

    # 4) 准备 unlabeled 列表
    if unlabeled_subset is None:
        unlabeled_subset = list_unlabeled_files(comp_dir)
    # 先 offset, 再 max_files
    if offset > 0:
        unlabeled_subset = unlabeled_subset[offset:]
    if max_files is not None:
        unlabeled_subset = unlabeled_subset[:max_files]
    print(f"[fake_base] linking {len(unlabeled_subset)} unlabeled files as test_soundscapes/  "
          f"(offset={offset}, max_files={max_files})")

    # 5) 把 unlabeled 文件链接到 fake_base/test_soundscapes/
    test_dir = fake_base / "test_soundscapes"
    test_dir.mkdir(parents=True, exist_ok=True)
    sc_dir = comp_dir / "train_soundscapes"
    for fn in unlabeled_subset:
        src = sc_dir / fn
        if not src.exists():
            print(f"[fake_base] WARN 缺失文件: {src}")
            continue
        _hardlink_or_copy(src, test_dir / fn)

    n_test = len(list(test_dir.glob("*.ogg")))
    print(f"[fake_base] DONE: {fake_base}  test_soundscapes/={n_test} files")
    return fake_base


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-dir", required=True, type=Path)
    parser.add_argument("--fake-base", required=True, type=Path)
    parser.add_argument("--max-files", type=int, default=None,
                        help="只链接前 N 个 unlabeled 文件 (调试用)")
    parser.add_argument("--offset", type=int, default=0,
                        help="从第 offset 个 unlabeled 文件开始切片 (分批用)")
    args = parser.parse_args()

    build_fake_base(
        comp_dir=args.comp_dir,
        fake_base=args.fake_base,
        max_files=args.max_files,
        offset=args.offset,
    )
