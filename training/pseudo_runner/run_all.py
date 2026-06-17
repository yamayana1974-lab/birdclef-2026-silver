# -*- coding: utf-8 -*-
"""run_all.py - 一键打伪标签流水线 (单套 Model_5).

流程:
    1. build_fake_base 构造 fake_base/ (把 unlabeled soundscape 当 test)
    2. 跑一遍 Model_5 (Perch + ProtoSSM + SED), 产出 protossm/sed csv
    3. merge_softlabels 融合 (proto*0.6 + sed*0.4) + 过滤 + 5 折分组

> 历史说明: 早期版本跑 v6/v13/v17 三套 scheme 取平均, 但三套只有 HGNet 分支
> backbone 不同, 而打伪标签阶段 HGNet 分支根本不参与 (只用 protossm + sed,
> 且最终 blend 被 runtime_patches 跳过), 三套输出完全相同. 因此这里简化为单套.

用法 (在 ``training/`` 目录下跑):
    python -m pseudo_runner.run_all \\
        --comp-dir ./data/birdclef-2026 \\
        --output-dir ./data/pseudo_output \\
        --perch-onnx ./models/perch_v2_no_dft.onnx \\
        --sed-dir ./models/distilled_sed_onnx

调试用:
    --max-files 50   只链接 50 个 unlabeled 文件, 验证 pipeline 跑通
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


# 单套方案名 (输出子目录名). 保留列表形式, 方便 merge 复用.
DEFAULT_SCHEMES: List[str] = ["m5"]


def _run_subprocess(cmd: List[str], env: Dict[str, str] = None,
                    cwd: Path = None) -> int:
    print(f"\n{'='*70}\n$ {' '.join(cmd)}\n{'='*70}", flush=True)
    sub_env = os.environ.copy()
    sub_env.setdefault("PYTHONUNBUFFERED", "1")   # 强制子进程 stdout 不缓冲
    if env:
        sub_env.update(env)
    proc = subprocess.run(cmd, env=sub_env, cwd=cwd)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description=__doc__)
    parser.add_argument("--comp-dir", required=True, type=Path,
                        help="birdclef-2026 真实比赛目录")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="所有伪标签 csv 落盘的根目录")
    parser.add_argument("--perch-onnx", required=True, type=Path,
                        help="perch_v2_no_dft.onnx 路径")
    parser.add_argument("--sed-dir", required=True, type=Path,
                        help="包含 sed_fold0..4.onnx 的目录")
    parser.add_argument("--input-root", default=None, type=Path,
                        help="INPUT_ROOT, rglob 入口. 默认 = perch-onnx 的父目录")
    parser.add_argument("--model-dir", default=None, type=Path,
                        help="Perch TF SavedModel 目录 (可选, ONNX 已够用)")
    parser.add_argument("--fake-base", default=None, type=Path,
                        help="fake_base 目录, 默认 output-dir/fake_base")
    parser.add_argument("--schemes", nargs="+", default=DEFAULT_SCHEMES,
                        help="要跑的方案子集 (默认单套 m5)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="调试用: 只链接前 N 个 unlabeled 文件")
    parser.add_argument("--offset", type=int, default=0,
                        help="从第 offset 个 unlabeled 文件开始 (分批用,配合 --max-files)")
    parser.add_argument("--skip-build-base", action="store_true",
                        help="跳过 build_fake_base 阶段 (复用已有的)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="跳过推理, 直接走 merge_softlabels (csv 已经在了)")
    parser.add_argument("--no-merge", action="store_true",
                        help="只跑推理, 不做最后的合并")
    # 训练配置
    parser.add_argument("--primary-min-prob", type=float, default=0.5)
    parser.add_argument("--trim-min-prob", type=float, default=0.1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 路径解析
    pkg_root = Path(__file__).resolve().parents[1]   # training/ (含 pseudo_runner 包)
    repo_root = Path(__file__).resolve().parents[2]  # 仓库根
    scheme_dir = repo_root / "inference" / "scheme_model_5"   # 发布版单套 Model_5
    fake_base = args.fake_base or (args.output_dir / "fake_base")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # === 1) build fake_base ===
    if not args.skip_build_base and not args.skip_inference:
        print(f"\n{'#'*70}\n# Step 1/3 - build fake_base\n{'#'*70}")
        cmd = [
            sys.executable, "-m", "pseudo_runner.build_fake_base",
            "--comp-dir", str(args.comp_dir),
            "--fake-base", str(fake_base),
        ]
        if args.max_files is not None:
            cmd += ["--max-files", str(args.max_files)]
        if args.offset > 0:
            cmd += ["--offset", str(args.offset)]
        rc = _run_subprocess(cmd, cwd=pkg_root)
        if rc != 0:
            print(f"build_fake_base failed (rc={rc}), abort.")
            sys.exit(rc)

    # === 2) 跑方案 (单套, 通过子进程隔离 + 复用发布版 scheme_model_5) ===
    if not args.skip_inference:
        for s in args.schemes:
            scheme_out = args.output_dir / s
            work_dir = args.output_dir / "_work" / s   # 每套独立 cache

            print(f"\n{'#'*70}\n# Step 2/3 - run scheme {s} (Model_5)\n{'#'*70}")
            cmd = [
                sys.executable, "-m", "pseudo_runner.run_one_scheme",
                "--scheme-dir", str(scheme_dir),
                "--comp-dir", str(args.comp_dir),
                "--fake-base", str(fake_base),
                "--output-dir", str(scheme_out),
                "--perch-onnx", str(args.perch_onnx),
                "--sed-dir", str(args.sed_dir),
                "--work-dir", str(work_dir),
            ]
            if args.input_root is not None:
                cmd += ["--input-root", str(args.input_root)]
            if args.model_dir is not None:
                cmd += ["--model-dir", str(args.model_dir)]
            rc = _run_subprocess(cmd, cwd=pkg_root)
            if rc != 0:
                print(f"scheme {s} failed (rc={rc}), abort.")
                sys.exit(rc)

    # === 3) 合并 + 过滤 + 分组 ===
    if not args.no_merge:
        print(f"\n{'#'*70}\n# Step 3/3 - merge & filter\n{'#'*70}")
        cmd = [
            sys.executable, "-m", "pseudo_runner.merge_softlabels",
            "--output-dir", str(args.output_dir),
            "--schemes", *args.schemes,
            "--primary-min-prob", str(args.primary_min_prob),
            "--trim-min-prob", str(args.trim_min_prob),
            "--n-folds", str(args.n_folds),
            "--seed", str(args.seed),
        ]
        rc = _run_subprocess(cmd, cwd=pkg_root)
        if rc != 0:
            print(f"merge_softlabels failed (rc={rc}), abort.")
            sys.exit(rc)

    print(f"\n{'='*70}\nDONE. output: {args.output_dir}/pseudo_filtered_grouped.csv\n{'='*70}")


if __name__ == "__main__":
    main()
