# -*- coding: utf-8 -*-
"""run_batches.py - 把 10592 个 unlabeled 文件分 N 批跑.

每批跑一遍 Model_5, 产出 ``batch_K/`` 子目录, 内含
submission_protossm.csv + submission_sed.csv.

中途某批失败/卡死, 可以 kill 之后用 ``--start-batch K`` 从那一批继续 (前面已跑完的
batch_0~K-1 不会被重跑).

最后用 ``merge_batches.py`` 把所有 batch 合并 + 过滤 + 5 折分组, 产出最终
``pseudo_filtered_grouped.csv``.

用法 (服务器后台运行):

    BIRDCLEF_PERCH_FORCE_CPU=1 \\
    nohup python -u -m pseudo_runner.run_batches \\
        --comp-dir ./data/birdclef-2026 \\
        --output-dir ./data/pseudo_output \\
        --perch-onnx ./models/perch_v2.onnx \\
        --input-root ./models \\
        --sed-dir ./models/distilled_sed_onnx \\
        --num-batches 10 \\
        > pseudo_run.log 2>&1 &

中途断了从 batch 5 继续:

    --start-batch 5
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def _run_subprocess(cmd: List[str], env=None, cwd: Path = None) -> int:
    print(f"\n{'='*70}\n$ {' '.join(cmd)}\n{'='*70}", flush=True)
    sub_env = os.environ.copy()
    sub_env.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        sub_env.update(env)
    proc = subprocess.run(cmd, env=sub_env, cwd=cwd)
    return proc.returncode


def _list_unlabeled_count(comp_dir: Path) -> int:
    """读一遍, 返回 unlabeled 文件数 (用于切批)."""
    from pseudo_runner.build_fake_base import list_unlabeled_files
    return len(list_unlabeled_files(comp_dir))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description=__doc__)
    parser.add_argument("--comp-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="所有 batch 输出的根目录, 会建 batch_0/, batch_1/, ...")
    parser.add_argument("--perch-onnx", required=True, type=Path)
    parser.add_argument("--input-root", default=None, type=Path)
    parser.add_argument("--model-dir", default=None, type=Path)
    parser.add_argument("--sed-dir", required=True, type=Path)
    parser.add_argument("--schemes", nargs="+", default=["m5"])
    parser.add_argument("--num-batches", type=int, default=10,
                        help="把 unlabeled 文件分多少批 (默认 10)")
    parser.add_argument("--start-batch", type=int, default=0,
                        help="从第几批开始 (中断恢复用, 默认 0)")
    parser.add_argument("--end-batch", type=int, default=None,
                        help="跑到第几批 (含, 默认全部)")
    args = parser.parse_args()

    pkg_root = Path(__file__).resolve().parents[1]   # training/ (含 pseudo_runner 包)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 算每批起止
    n_total = _list_unlabeled_count(args.comp_dir)
    batch_size = math.ceil(n_total / args.num_batches)
    print(f"[batches] total_unlabeled={n_total}  num_batches={args.num_batches}  "
          f"batch_size={batch_size}", flush=True)

    end = args.end_batch if args.end_batch is not None else (args.num_batches - 1)

    # 2) 串行跑每批
    for k in range(args.start_batch, end + 1):
        offset = k * batch_size
        if offset >= n_total:
            print(f"[batches] batch {k} offset {offset} >= total {n_total}, skip")
            continue
        max_files = min(batch_size, n_total - offset)

        batch_dir = args.output_dir / f"batch_{k:02d}"

        # ── 跳过已完成 batch ──
        if (batch_dir / ".done").exists():
            print(f"\n[batches] batch_{k:02d} already done (.done exists), skip", flush=True)
            continue

        print(f"\n{'#'*70}\n# Batch {k}/{end}  offset={offset}  files={max_files}  "
              f"-> {batch_dir}\n{'#'*70}", flush=True)

        # 清掉这一批的旧产物 (避免半成品干扰)
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)

        # 调用 run_all.py 跑这批
        cmd = [
            sys.executable, "-u", "-m", "pseudo_runner.run_all",
            "--comp-dir", str(args.comp_dir),
            "--output-dir", str(batch_dir),
            "--perch-onnx", str(args.perch_onnx),
            "--sed-dir", str(args.sed_dir),
            "--schemes", *args.schemes,
            "--max-files", str(max_files),
            "--offset", str(offset),
            "--no-merge",          # 单批不做最终融合, 留给 merge_batches.py
        ]
        if args.input_root is not None:
            cmd += ["--input-root", str(args.input_root)]
        if args.model_dir is not None:
            cmd += ["--model-dir", str(args.model_dir)]

        rc = _run_subprocess(cmd, cwd=pkg_root)
        if rc != 0:
            print(f"\n[batches] BATCH {k} FAILED (rc={rc}). 已完成的批保留.\n"
                  f"  恢复命令:  --start-batch {k}", flush=True)
            sys.exit(rc)

        # 标记完成
        (batch_dir / ".done").touch()
        # 清掉这一批的 fake_base + _work, 节省空间
        for sub in ("fake_base", "_work"):
            p = batch_dir / sub
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

        print(f"\n[batches] batch {k} DONE -> {batch_dir}", flush=True)

    # 3) 全部 batch 跑完, 提示用户做最终合并
    print(f"\n{'='*70}\nALL BATCHES DONE. Now run merge_batches.py:\n"
          f"  python -m pseudo_runner.merge_batches \\\n"
          f"    --output-dir {args.output_dir} \\\n"
          f"    --schemes {' '.join(args.schemes)}\n"
          f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
