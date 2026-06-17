# -*- coding: utf-8 -*-
"""main.py — Model_5 全流程入口 → 产出 ``subm_5.csv``.

用法:

.. code-block:: bash

   # Kaggle (一切 /kaggle/input/... 路径就绪)
   python -m scheme_model_5.main

   # 本地
   set BIRDCLEF_BASE=.\\data\\birdclef-2026
   set BIRDCLEF_INPUT_ROOT=.\\data
   python -m scheme_model_5.main

流程顺序对应 cell_07:

1. ``setup_env.bootstrap``      → wheel install + seed=4 + ONNX 探测
2. ``data.load_data``
3. ``mapping.build_mapping``
4. ``perch_inference``           → ``meta_tr / sc_tr / emb_tr / Y_FULL_aligned``
5. ``protossm_pipeline``         → ``submission_protossm.csv``
6. ``sed_pipeline``              → ``submission_sed.csv``
7. ``blend.blend_proto_sed``     → ``subm_karnakbayev_power_optimization.csv``
8. ``blend.direct_add_safe`` (单模型 weight=1.0) + ``write_final_submission`` → ``subm_5.csv``
"""
from __future__ import annotations

from pathlib import Path

from .blend import blend_proto_sed, direct_add_safe, write_final_submission
from .config import (
    INTERMEDIATE_SUBM, MODE, OUTPUT_SUBM, SOLUT, USE_HGNET, WALL_START, elapsed_min,
)
from .data import load_data
from .mapping import build_mapping
from .perch_inference import (
    PerchBackend, load_cache_arrays, load_or_build_cache,
)
from .protossm_pipeline import run_protossm_pipeline
from .sed_pipeline import run_sed_pipeline
from .setup_env import bootstrap


def main() -> None:
    print("=" * 70)
    print(f"  Model_5  (Perch + ProtoSSM + ResSSM + Distilled-SED)  MODE={MODE}")
    print("=" * 70)
    onnx_ok = bootstrap(seed=4)

    # 1. data
    d = load_data()
    base_dir       = d["BASE"]           # base，数据集的位置
    PRIMARY_LABELS = d["PRIMARY_LABELS"]
    N_CLASSES      = d["N_CLASSES"]

    # 2. mapping (Perch labels + temperatures)
    mp = build_mapping(d["taxonomy"], d["label_to_idx"], PRIMARY_LABELS, N_CLASSES)

    # 3. Perch backend + cache
    backend = PerchBackend(onnx_available=onnx_ok)
    cache_meta, cache_npz = load_or_build_cache(
        backend, mp, d["full_files"], base_dir, N_CLASSES, PRIMARY_LABELS,
    )
    meta_tr, sc_tr, emb_tr, Y_FULL_aligned = load_cache_arrays(
        cache_meta, cache_npz, d["full_rows"], d["Y_SC"],
        N_CLASSES, PRIMARY_LABELS,
    )

    # 4. ProtoSSM pipeline → submission_protossm.csv
    proto_csv = run_protossm_pipeline(
        backend=backend,
        mapping=mp,
        primary_labels=PRIMARY_LABELS,
        n_classes=N_CLASSES,
        base_dir=base_dir,
        sc=d["sc"],
        Y_SC=d["Y_SC"],
        meta_tr=meta_tr, sc_tr=sc_tr, emb_tr=emb_tr,
        Y_FULL_aligned=Y_FULL_aligned,
        output_csv="submission_protossm.csv",
    )

    # 5. SED pipeline → submission_sed.csv
    test_paths = sorted((base_dir / "test_soundscapes").glob("*.ogg"))
    if len(test_paths) == 0:
        from .config import CFG
        n = CFG["dryrun_n_files"] or 20
        test_paths = sorted((base_dir / "train_soundscapes").glob("*.ogg"))[:n]
    sed_csv = run_sed_pipeline(
        test_paths, PRIMARY_LABELS, N_CLASSES,
        output_csv="submission_sed.csv",
    )

    # 6. xSED rank blend + 5 道 gate → subm_karnakbayev_power_optimization.csv
    inter_csv = blend_proto_sed(
        str(proto_csv), str(sed_csv), base_dir,
        output_csv=INTERMEDIATE_SUBM,
    )

    # 7. solut 单模型 weight=1.0 + write_final_submission → subm_5.csv
    files = [m["subm"]   for m in SOLUT["Models"]]
    wts   = [m["weight"] for m in SOLUT["Models"]]
    names = [m["Model"]  for m in SOLUT["Models"]]
    lbs   = [m["LB"]     for m in SOLUT["Models"]]

    final = direct_add_safe(files, wts, names, lbs)
    write_final_submission(final, base_dir, path=OUTPUT_SUBM)

    # 8. (可选) HGNet 4-fold ensemble rank-blend (默认 8% 权重) → 覆盖 subm_5.csv
    if USE_HGNET:
        print("\n" + "=" * 70)
        print("  Stage HGNet — 4-fold ensemble rank-blend")
        print("=" * 70)
        from .hgnet_addon import append_hgnet_to_submission
        append_hgnet_to_submission(
            final_subm_path = OUTPUT_SUBM,
            test_paths      = test_paths,
            primary_labels  = PRIMARY_LABELS,
        )

    print(f"[main] DONE  wall_time={elapsed_min():.1f} min")


if __name__ == "__main__":
    main()
