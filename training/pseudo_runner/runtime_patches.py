# -*- coding: utf-8 -*-
"""runtime_patches.py - monkey-patch scheme_model_5,适配伪标签场景.

调用 ``apply_all(scheme_pkg)`` 后:
  1. ``probes.apply_per_class_thresholds`` 变成 identity (软概率不阈值化).
  2. ``perch_inference.PerchBackend`` 的 ONNX session 加 CUDAExecutionProvider.
  3. ``sed_pipeline.make_sed_session`` 同上.
  4. ``protossm_pipeline / training`` 里 torch tensor 默认走 cuda (如果可用).

这样不需要改 scheme_model_5 源代码, 在子进程启动时 import 这一发就行.
"""
from __future__ import annotations

import os
from importlib import import_module
from typing import Optional


def _patch_thresholds(scheme_pkg: str) -> None:
    """让 apply_per_class_thresholds 变成 identity, 保留软概率."""
    probes = import_module(f"{scheme_pkg}.probes")
    original = probes.apply_per_class_thresholds

    def identity(probs, _thresholds):  # noqa: ARG001
        return probs

    probes.apply_per_class_thresholds = identity
    print(f"[patch] {scheme_pkg}.probes.apply_per_class_thresholds -> identity")

    # protossm_pipeline 里直接 from .probes import apply_per_class_thresholds,
    # 上面的替换不会同步到它的 namespace. 单独再 patch 一份.
    pp = import_module(f"{scheme_pkg}.protossm_pipeline")
    if hasattr(pp, "apply_per_class_thresholds"):
        pp.apply_per_class_thresholds = identity
        print(f"[patch] {scheme_pkg}.protossm_pipeline.apply_per_class_thresholds -> identity")


def _patch_perch_cuda(scheme_pkg: str, intra_op: int = 4) -> None:
    """给 Perch ONNX session 优先用 CUDAExecutionProvider.

    可以通过环境变量 ``BIRDCLEF_PERCH_FORCE_CPU=1`` 强制走 CPU.
    某些 ORT/cuDNN 组合下 Perch ONNX 第一次 GPU 初始化会卡死(autotune 死循环),
    这时让 Perch 回 CPU 即可——反正 cache 命中后 Perch 几乎不参与推理.
    """
    pi = import_module(f"{scheme_pkg}.perch_inference")
    OriginalBackend = pi.PerchBackend

    force_cpu = os.environ.get("BIRDCLEF_PERCH_FORCE_CPU", "0") not in {"0", "false", "False"}

    class CudaPerchBackend(OriginalBackend):
        def __init__(self, onnx_available: bool):
            onnx_no_dft = next(pi.INPUT_ROOT.glob("**/perch_v2_no_dft*.onnx"), None)
            onnx_any = next(pi.INPUT_ROOT.glob("**/perch_v2*.onnx"), None)
            self.onnx_path = onnx_no_dft or onnx_any
            self.use_onnx = bool(onnx_available) and self.onnx_path is not None and self.onnx_path.exists()
            self.onnx_session = None
            self.onnx_input_name = None
            self.onnx_out_map = None
            self.tf_infer_fn = None

            if self.use_onnx:
                # ONNX 优先: 完全不加载 TF SavedModel (没装 TensorFlow 也能跑).
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = intra_op
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                providers = []
                avail = set(ort.get_available_providers())
                if (not force_cpu) and "CUDAExecutionProvider" in avail:
                    providers.append(("CUDAExecutionProvider", {"device_id": 0}))
                providers.append("CPUExecutionProvider")

                self.onnx_session = ort.InferenceSession(
                    str(self.onnx_path), sess_options=so, providers=providers,
                )
                self.onnx_input_name = self.onnx_session.get_inputs()[0].name
                self.onnx_out_map = {
                    o.name: i for i, o in enumerate(self.onnx_session.get_outputs())
                }
                used = self.onnx_session.get_providers()
                print(f"[perch] ONNX backend: {self.onnx_path.name}  providers={used}  "
                      f"(force_cpu={force_cpu})")
            elif pi.MODEL_DIR.exists():
                # 没有可用 ONNX 时才回退 TF SavedModel; TF 只在此分支 import.
                import tensorflow as tf
                birdclassifier = tf.saved_model.load(str(pi.MODEL_DIR))
                self.tf_infer_fn = birdclassifier.signatures["serving_default"]
                print("[perch] TF SavedModel backend")
            else:
                raise FileNotFoundError("No usable Perch backend.")

    pi.PerchBackend = CudaPerchBackend
    print(f"[patch] {scheme_pkg}.perch_inference.PerchBackend -> CUDA-aware  "
          f"(force_cpu={force_cpu})")


def _patch_sed_cuda(scheme_pkg: str, intra_op: int = 4) -> None:
    """SED ONNX session 改用 CUDAExecutionProvider."""
    sp = import_module(f"{scheme_pkg}.sed_pipeline")

    def make_sed_session_cuda(path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        avail = set(ort.get_available_providers())
        providers = []
        if "CUDAExecutionProvider" in avail:
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        providers.append("CPUExecutionProvider")
        sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        print(f"[sed] {path.name} providers={sess.get_providers()}")
        return sess

    sp.make_sed_session = make_sed_session_cuda
    print(f"[patch] {scheme_pkg}.sed_pipeline.make_sed_session -> CUDA-aware")


def _patch_proto_cuda(scheme_pkg: str) -> None:
    """ProtoSSM / ResidualSSM 的大数据量 forward 走 cuda."""
    import torch

    if not torch.cuda.is_available():
        print("[patch] cuda not available; ProtoSSM 留在 cpu")
        return

    training = import_module(f"{scheme_pkg}.training")
    pp = import_module(f"{scheme_pkg}.protossm_pipeline")

    # ── 1) run_tta_proto: ProtoSSM 5-shift TTA forward ─────────────
    def run_tta_proto_cuda(proto_model, emb_files, sc_files, site_t, hour_t,
                            shifts=(0, 1, -1, 2, -2)):
        import numpy as np
        proto_model = proto_model.to("cuda").eval()
        all_preds = []
        emb_t = torch.tensor(emb_files, dtype=torch.float32, device="cuda")
        sc_t = torch.tensor(sc_files, dtype=torch.float32, device="cuda")
        site_t = site_t.to("cuda")
        hour_t = hour_t.to("cuda")

        for shift in shifts:
            e = torch.roll(emb_t, shift, dims=1) if shift else emb_t
            s = torch.roll(sc_t, shift, dims=1) if shift else sc_t
            with torch.no_grad():
                out = proto_model(e, s, site_ids=site_t, hours=hour_t).cpu().numpy()
            if shift:
                out = np.roll(out, -shift, axis=1)
            all_preds.append(out)
        with torch.no_grad():
            out_flip = proto_model(
                emb_t.flip(1), sc_t.flip(1),
                site_ids=site_t, hours=hour_t,
            ).cpu().numpy()
        all_preds.append(out_flip[:, ::-1, :].copy())
        proto_model.cpu()
        del emb_t, sc_t
        torch.cuda.empty_cache()
        return np.mean(all_preds, axis=0)

    training.run_tta_proto = run_tta_proto_cuda
    pp.run_tta_proto = run_tta_proto_cuda
    print(f"[patch] {scheme_pkg}.training.run_tta_proto -> cuda forward")

    # ── 2) ResidualSSM 推理也搬 cuda (monkey-patch torch.tensor) ──
    #     简单粗暴: protossm_pipeline.py 里 res_model(...) 之前会 torch.tensor(..)
    #     创建 cpu tensor. 我们把整个 res_model 在调用前 .to('cuda'),
    #     用 wrap 的方式包一层 forward, 让它接受 cpu tensor 也能跑 cuda.
    import torch.nn as nn

    class _ResModelGpuWrapper(nn.Module):
        """把 inputs 自动搬到 cuda, output 搬回 cpu, 让 protossm_pipeline 不用改."""
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner.to("cuda").eval()

        def forward(self, *args, **kwargs):
            cuda_args = []
            for a in args:
                if torch.is_tensor(a):
                    cuda_args.append(a.to("cuda"))
                else:
                    cuda_args.append(a)
            cuda_kwargs = {}
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    cuda_kwargs[k] = v.to("cuda")
                else:
                    cuda_kwargs[k] = v
            with torch.no_grad():
                out = self.inner(*cuda_args, **cuda_kwargs)
            return out.cpu()

        def eval(self):
            self.inner.eval()
            return self

        def to(self, *args, **kwargs):
            return self

    # patch train_residual_ssm: 训完直接 wrap. 训练阶段还是 cpu (66 文件够快).
    original_train_residual = training.train_residual_ssm

    def train_residual_ssm_wrapped(*args, **kwargs):
        res_model, correction_weight = original_train_residual(*args, **kwargs)
        wrapped = _ResModelGpuWrapper(res_model)
        return wrapped, correction_weight

    training.train_residual_ssm = train_residual_ssm_wrapped
    pp.train_residual_ssm = train_residual_ssm_wrapped
    print(f"[patch] {scheme_pkg}.training.train_residual_ssm -> auto-cuda wrapper")


def _patch_skip_blend(scheme_pkg: str) -> None:
    """跳过 blend / write_final_submission.

    伪标签场景下我们只要 ``submission_protossm.csv`` + ``submission_sed.csv`` 即可.
    后续的 xSED rank blend / sample_submission 对齐 / direct_add_safe 都是
    Kaggle 提交场景的逻辑, 在打伪标签时反而会因为 row_id 不匹配 sample_submission
    (我们用的是 BC2026_Train_* 而不是 BC2026_Test_*) 直接抛 AssertionError.

    把 ``blend.write_final_submission`` 替换成 print + return, 让 main 跑完不报错.
    """
    blend = import_module(f"{scheme_pkg}.blend")

    def write_final_submission_skip(*args, **kwargs):
        print(f"[patch] write_final_submission skipped (pseudo-label mode)")
        return None

    blend.write_final_submission = write_final_submission_skip
    # main.py 是 ``from .blend import write_final_submission`` 引入的, 同步改它
    main_mod = import_module(f"{scheme_pkg}.main")
    if hasattr(main_mod, "write_final_submission"):
        main_mod.write_final_submission = write_final_submission_skip
    print(f"[patch] {scheme_pkg}.blend.write_final_submission -> skip")


def apply_all(scheme_pkg: str = "scheme_model_5") -> None:
    """一次应用所有 patch. 在 main.main() 之前调."""
    _patch_thresholds(scheme_pkg)
    _patch_perch_cuda(scheme_pkg)
    _patch_sed_cuda(scheme_pkg)
    _patch_proto_cuda(scheme_pkg)
    _patch_skip_blend(scheme_pkg)
