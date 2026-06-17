# -*- coding: utf-8 -*-
"""run_one_scheme.py - 在子进程内跑一遍 Model_5, 产出软概率 csv.

由 ``run_all.py`` 通过 subprocess 调用. 之所以走子进程, 是为了让
``apply_all`` 的 monkey-patch 和 ``BIRDCLEF_*`` 环境变量在干净的解释器里
生效, 不污染父进程.

复用发布版的单套 ``inference/scheme_model_5`` (通过 ``--scheme-dir`` 把它的
父目录加进 sys.path, 然后 ``import scheme_model_5``).

输出:
  ``<output_dir>/submission_protossm.csv``
  ``<output_dir>/submission_sed.csv``

它们都是 [0, 1] 的软概率, 后续 merge_softlabels.py 负责融合.

用法:
    python -m pseudo_runner.run_one_scheme \
        --scheme-dir /path/to/inference/scheme_model_5 \
        --fake-base  /tmp/fake_base \
        --output-dir /path/to/output/m5 \
        --comp-dir   /path/to/birdclef-2026 \
        --perch-onnx /path/to/perch_v2_no_dft.onnx \
        --sed-dir    /path/to/distilled_sed_onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# 全局变量, 在 main 里被 set, 在 _set_env 里读
scheme_dir_global = [None]


def _setup_pythonpath(scheme_dir: Path, pkg_root: Path) -> None:
    """让 ``import scheme_model_5`` 找到发布版那一份源码,
    并让 ``from pseudo_runner import ...`` 能找到本包."""
    # scheme_dir 是 .../inference/scheme_model_5, 把它父目录 (inference/) 加进去
    sys.path.insert(0, str(scheme_dir.parent))
    sys.path.insert(0, str(pkg_root))


def _set_env(comp_dir: Path, fake_base: Path,
             perch_onnx: Path, sed_dir: Path,
             input_root: Path = None,
             model_dir: Path = None,
             work_dir: Path = None) -> None:
    """注入 Model_5 需要的环境变量.

    BIRDCLEF_INPUT_ROOT 用作 rglob 入口: mapping.py 找 labels.csv,
    perch_inference 找 perch_v2*.onnx, sed_pipeline 找 sed_fold0.onnx.
    """
    os.environ["BIRDCLEF_BASE"] = str(fake_base)
    os.environ["BIRDCLEF_MODE"] = "submit"

    # INPUT_ROOT: 给 rglob perch_v2_no_dft*.onnx / sed_fold0.onnx / labels.csv 用
    if input_root is not None:
        os.environ["BIRDCLEF_INPUT_ROOT"] = str(input_root)

    # Perch TF SavedModel (有就用, 没有 fallback ONNX)
    if model_dir is not None:
        os.environ["BIRDCLEF_MODEL_DIR"] = str(model_dir)

    # cache work dir
    if work_dir is not None:
        os.environ["BIRDCLEF_WORK_DIR"] = str(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    # SED ONNX dir (sed_pipeline.find_sed_dir 优先看这个)
    os.environ["BIRDCLEF_SED_DIR"] = str(sed_dir)

    # 打伪标签阶段不启用 HGNet 分支 (只要 protossm + sed)
    os.environ["BIRDCLEF_M5_USE_HGNET"] = "0"


def main() -> None:
    # 强制 stdout 不缓冲, 让 nohup tail -f 能立刻看到日志
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme-dir", required=True, type=Path,
                        help="inference/scheme_model_5 这一级目录")
    parser.add_argument("--comp-dir", required=True, type=Path)
    parser.add_argument("--fake-base", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--perch-onnx", required=True, type=Path)
    parser.add_argument("--sed-dir", required=True, type=Path)
    parser.add_argument("--input-root", default=None, type=Path,
                        help="给 INPUT_ROOT, rglob 入口. 默认 = perch-onnx 的父目录")
    parser.add_argument("--model-dir", default=None, type=Path,
                        help="Perch TF SavedModel 目录 (可选, ONNX 已够用)")
    parser.add_argument("--work-dir", default=None, type=Path,
                        help="Perch cache 目录 (建议每套独立)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scheme_dir_global[0] = args.scheme_dir

    # 默认 input_root: perch-onnx 的父目录 (确保 rglob 能找到 onnx)
    input_root = args.input_root
    if input_root is None:
        input_root = args.perch_onnx.parent

    # 1) 设环境变量
    _set_env(
        comp_dir=args.comp_dir,
        fake_base=args.fake_base,
        perch_onnx=args.perch_onnx,
        sed_dir=args.sed_dir,
        input_root=input_root,
        model_dir=args.model_dir,
        work_dir=args.work_dir,
    )

    # 2) 把 scheme 目录的父目录加到 sys.path 最前
    pkg_root = Path(__file__).resolve().parents[1]   # training/
    _setup_pythonpath(args.scheme_dir, pkg_root)

    # 3) import + 应用 monkey-patch (软概率 + CUDA + skip blend)
    from scheme_model_5 import main as m5main  # noqa: E402
    from pseudo_runner import runtime_patches  # noqa: E402
    runtime_patches.apply_all("scheme_model_5")

    # 4) 切到 output_dir 跑 pipeline (csv 默认写当前目录)
    cwd_old = Path.cwd()
    os.chdir(args.output_dir)
    try:
        m5main.main()
    finally:
        os.chdir(cwd_old)

    # 5) 验证关键产物
    expected = [
        args.output_dir / "submission_protossm.csv",
        args.output_dir / "submission_sed.csv",
    ]
    for p in expected:
        if not p.exists():
            print(f"[run_one] WARN missing {p.name}")
        else:
            print(f"[run_one] OK {p}")


if __name__ == "__main__":
    main()
