# -*- coding: utf-8 -*-
"""
train.py — 训练函数 train_fold
================================

对应 notebook 的 S5 cell.

核心函数 train_fold(fold_k):
  1. 划分 train / val (按 fold)
  2. 构造 ConcatDataset(FocalDS + ScDS) + MixSamp
  3. 创建 BirdSEDModel + Mel + SpecAugment + PerchTeacher
  4. AdamW + Warmup + Cosine LR schedule
  5. 每 epoch:
     - train step: 增强 → 模型 → Loss = cls + α·distill → AMP backward
     - val step:   关 augment, 全 fold val 集预测, 算 5 个指标
     - 早停: 跟踪 best_ns22 + best_macro 两个 checkpoint
  6. 返回 (best_ns22_state, best_macro_state, history)

★ Loss 设计:
    cls_loss     = 0.5 * BCE(clip_logits, label) + 0.5 * BCE(frame_max, label)
    distill_loss = MSE(student_distill_emb, perch_teacher_emb)
    total        = cls_loss + α * distill_loss     (α=1.0)

★ Validation 输出三种 head 模式的 AUC:
    - clip-only:     sigmoid(clip_logits)
    - fmax-only:     sigmoid(frame_max_logits)
    - blend:         0.5*clip + 0.5*fmax  (跟训练 cls_loss 一致, 也跟推理一致)
"""
import time
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from config import (WAVEFORM_CACHE_DIR, SR, VAL_SAMPLES, BATCH, EPOCHS,
                    LR, WD, WARMUP_EPOCHS,
                    USE_PERCH_DISTILL, ALPHA_DISTILL, PERCH_ONNX_PATH,
                    USE_FOCAL, USE_LABELED_SC, USE_FOCAL_SC_MIXUP,
                    SHARES, ACTIVE_SOURCES,
                    USE_PSEUDO, PSEUDO_SC_DIR,
                    NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS, AMP_DTYPE,
                    USE_PERCH_CACHE, EARLY_STOP_PATIENCE,
                    device)
from models import MelSpecTransform, SpecAugment, PerchTeacher, make_model
from dataset import (FocalDS, ScDS, PseudoDS, MixSamp, collate_m, mk_sw,
                     load_sc_waveform_from, extract_chunk_np)
from eval import full_eval, compute_macro_auc


# ── AMP dtype 选择 (bf16 在 Blackwell/H100 上稳, 不需要 GradScaler) ────
_AMP_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_TORCH_AMP_DTYPE = _AMP_MAP.get(AMP_DTYPE, torch.bfloat16)
_USE_GRADSCALER  = (_TORCH_AMP_DTYPE == torch.float16)
print(f"[AMP] dtype={AMP_DTYPE}  use_gradscaler={_USE_GRADSCALER}")


# =============================================================================
# 1. Helper: 加载验证集波形
# =============================================================================

def _load_val_waveforms(val_sc_df):
    """
    把 val fold 的 sc 段一次性加载到内存 (val 集小, 一次加载比每 epoch 重读快).
    返回 list of (1, VAL_SAMPLES) tensors.
    """
    sc_file_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_file_meta.csv")
    sc_file_dict = dict(zip(sc_file_meta["filename"], sc_file_meta["cache_file"]))
    wavs = []
    for _, row in val_sc_df.iterrows():
        cf = sc_file_dict.get(row["filename"])
        if cf is not None:
            w = load_sc_waveform_from(WAVEFORM_CACHE_DIR, cf)
            if w is not None:
                chunk = extract_chunk_np(w, int(row["start_sec"]) * SR, VAL_SAMPLES)
                wavs.append(torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0))
            else:
                wavs.append(torch.zeros(1, VAL_SAMPLES))
        else:
            wavs.append(torch.zeros(1, VAL_SAMPLES))
    return wavs


# =============================================================================
# 2. Helper: 验证推理 (3 种 head 都算 AUC)
# =============================================================================

def _predict_from_waveforms(model, mel_transform, wav_list, batch_size=128):
    """
    Args:
      model:         BirdSEDModel
      mel_transform: MelSpecTransform
      wav_list:      list of (1, VAL_SAMPLES) tensors

    Returns:
      dict {clip, fmax, blend → (N, 234) numpy}
    """
    model.eval()
    preds_clip, preds_fmax, preds_blend = [], [], []
    with torch.no_grad():
        for s in range(0, len(wav_list), batch_size):
            batch = torch.stack(wav_list[s:s + batch_size]).to(device, non_blocking=True)
            mel = mel_transform(batch)
            # 向量化 per-sample 归一化 (跟训练时一致)
            mean = mel.mean(dim=(2, 3), keepdim=True)
            std  = mel.std(dim=(2, 3), keepdim=True) + 1e-6
            mel  = (mel - mean) / std
            with torch.amp.autocast(device_type="cuda", dtype=_TORCH_AMP_DTYPE):
                clip_logits, framewise = model(mel, return_framewise=True)
                frame_max = framewise.max(dim=1).values             # (B, n_classes)
            p_clip  = torch.sigmoid(clip_logits.float()).cpu().numpy()
            p_fmax  = torch.sigmoid(frame_max.float()).cpu().numpy()
            p_blend = 0.5 * p_clip + 0.5 * p_fmax
            preds_clip.append(p_clip)
            preds_fmax.append(p_fmax)
            preds_blend.append(p_blend)
    return {
        "clip":  np.concatenate(preds_clip),
        "fmax":  np.concatenate(preds_fmax),
        "blend": np.concatenate(preds_blend),
    }


# =============================================================================
# 3. Helper: 按 fold_k 构造 train 数据集 (排除 val fold)
# =============================================================================

def build_active_datasets(fold_k, audio_cache_meta, sc_cache_meta, Y_SC,
                           LABEL2IDX, focal_secondary_labels, sc_mixup_sources,
                           pseudo_df=None, primary_labels=None):
    """
    返回 [(name, dataset, size), ...].

    focal:  排除 fold_k 的所有 focal 录音
    sc:     排除 fold_k 的 labeled sc 段
    pseudo: 排除 fold_k 的伪标签段 (软标签, source_weight 由 SOURCE_WEIGHTS["pseudo"] 控制)
    """
    items = []
    if USE_FOCAL:
        fds = FocalDS(
            audio_cache_meta[audio_cache_meta["fold"] != fold_k],
            LABEL2IDX,
            secondary_lookup=focal_secondary_labels,
            sc_mixup_sources=sc_mixup_sources if USE_FOCAL_SC_MIXUP else None,
            fold_k=fold_k,
            aug=True,
        )
        items.append(("focal", fds, len(fds)))
    if USE_LABELED_SC:
        vm           = sc_cache_meta["fold"].values == fold_k
        sc_train_df  = sc_cache_meta[~vm].reset_index(drop=True)
        Y_tr         = Y_SC[~vm]
        sds = ScDS(Y_tr, sc_train_df, aug=True)
        items.append(("sc", sds, len(sds)))
    if USE_PSEUDO and pseudo_df is not None and len(pseudo_df) > 0:
        if primary_labels is None:
            raise ValueError("primary_labels required when USE_PSEUDO=True")
        ps_train = pseudo_df[pseudo_df["fold"] != fold_k].reset_index(drop=True)
        if len(ps_train) > 0:
            pds = PseudoDS(ps_train, classes=primary_labels,
                           sc_dir=PSEUDO_SC_DIR, aug=True)
            items.append(("pseudo", pds, len(pds)))
    return items


# =============================================================================
# 4. 主训练函数 train_fold
# =============================================================================

def train_fold(fold_k, data_bundle):
    """
    训一个 fold.

    Args:
      fold_k:       fold index (0~4)
      data_bundle:  load_data() 返回的 dict

    Returns:
      best_ns22_state:  state_dict (按 non_S22 macro AUC 最高保存的)
      best_macro_state: state_dict (按全集 macro AUC 最高保存的)
      history:          每 epoch 记录的训练 loss / val AUC dict
    """
    audio_cache_meta       = data_bundle["audio_cache_meta"]
    sc_cache_meta          = data_bundle["sc_cache_meta"]
    Y_SC                   = data_bundle["Y_SC"]
    non_s22_mask_sc        = data_bundle["non_s22_mask_sc"]
    LABEL2IDX              = data_bundle["LABEL2IDX"]
    TAXON_MASKS            = data_bundle["TAXON_MASKS"]
    focal_secondary_labels = data_bundle["focal_secondary_labels"]
    sc_mixup_sources       = data_bundle["sc_mixup_sources"]
    pseudo_df              = data_bundle.get("pseudo_df", None)
    PRIMARY_LABELS         = data_bundle["PRIMARY_LABELS"]

    # ── 1. val 数据 ────────────────────────────────────────────────
    vm        = sc_cache_meta["fold"].values == fold_k
    Y_val     = Y_SC[vm]
    ns22_val  = non_s22_mask_sc[vm]
    val_sc_df = sc_cache_meta[vm].reset_index(drop=True)

    # ── 2. train 数据集 (focal + sc + pseudo 多源) ─────────────────
    active = build_active_datasets(
        fold_k, audio_cache_meta, sc_cache_meta, Y_SC, LABEL2IDX,
        focal_secondary_labels, sc_mixup_sources,
        pseudo_df=pseudo_df, primary_labels=PRIMARY_LABELS,
    )
    names, datasets, sizes = zip(*active)
    mds = ConcatDataset(list(datasets))
    nst = max(100, int(sum(sizes) / BATCH))                 # 每 epoch 步数
    print(f"  Streams: {dict(zip(names, sizes))}  steps/ep: {nst}")

    # ── 3. 模型 + 增强 + teacher ───────────────────────────────────
    m             = make_model()
    mel_transform = MelSpecTransform().to(device)
    spec_augment  = SpecAugment().to(device)
    # ★ Perch teacher: 开了 cache 就不需要 ONNX (cache 里都有了)
    perch_teacher = None
    if USE_PERCH_DISTILL and not USE_PERCH_CACHE:
        perch_teacher = PerchTeacher(
            PERCH_ONNX_PATH,
            "cuda" if torch.cuda.is_available() else "cpu",
        )

    # ── 4. Optimizer + LR schedule (warmup + cosine) ───────────────
    opt          = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    scaler       = torch.amp.GradScaler("cuda", enabled=_USE_GRADSCALER)
    warmup_steps = nst * WARMUP_EPOCHS
    total_steps  = nst * EPOCHS

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1 / 25, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=1e-6,
    )
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps],
    )

    # ── 5. 训练 loop ───────────────────────────────────────────────
    history = {
        "ep": [], "train_loss": [], "cls_loss": [], "dist_loss": [],
        "macro": [], "ns22_macro": [],
        "ns22_Aves": [], "ns22_Amphibia": [], "ns22_Insecta": [], "ns22_Mammalia": [],
        "val_preds": [],
    }
    best_ns22,  best_state_ns22  = -1.0, None
    best_macro, best_state_macro = -1.0, None
    no_improve_cnt = 0                            # 连续多少个 epoch ns22 没刷新
    val_wavs = _load_val_waveforms(val_sc_df)

    for ep in range(EPOCHS):
        m.train()
        smp = MixSamp(list(sizes), list(names), SHARES, BATCH, nst, seed=42 + ep)
        tl  = DataLoader(
            mds, batch_sampler=smp, collate_fn=collate_m,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
            prefetch_factor=4 if NUM_WORKERS > 0 else None,
        )
        el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
        t0 = time.time()

        for wav, lb, wt, mk, sr, perch_emb_cache in tl:
            wav = wav.to(device, non_blocking=True)
            lb  = lb.to(device, non_blocking=True)
            wt  = wt.to(device, non_blocking=True)
            mk  = mk.to(device, non_blocking=True)
            perch_emb_cache = perch_emb_cache.to(device, non_blocking=True)
            sw  = mk_sw(sr).to(device, non_blocking=True)

            # ── 5a. Mel + 标准化 + SpecAugment (no_grad, 不算梯度) ─
            with torch.no_grad():
                mel  = mel_transform(wav)
                mean = mel.mean(dim=(2, 3), keepdim=True)
                std  = mel.std(dim=(2, 3), keepdim=True) + 1e-6
                mel  = (mel - mean) / std
                mel  = spec_augment(mel)

            # ── 5b. Forward + Loss (AMP) ────────────────────────────
            with torch.amp.autocast(device_type="cuda", dtype=_TORCH_AMP_DTYPE):
                if USE_PERCH_DISTILL:
                    clip_logits, framewise, distill_emb = m(
                        mel, return_framewise=True, return_distill=True,
                    )
                else:
                    clip_logits, framewise = m(mel, return_framewise=True)

                frame_max_logits = framewise.max(dim=1).values

                # Classification loss = 0.5 clip + 0.5 frame_max
                bce_clip  = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
                bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
                bce       = 0.5 * bce_clip + 0.5 * bce_frame
                ps        = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                cls_loss  = (ps * sw).mean()

                # Distillation loss = MSE(student_emb, perch_teacher_emb)
                if USE_PERCH_DISTILL:
                    if USE_PERCH_CACHE:
                        # ★ 直接用预算好的 emb (大幅加速)
                        perch_emb = perch_emb_cache
                    else:
                        # fallback: 在线跑 ONNX teacher (伪标签段也能拿到真 emb)
                        with torch.no_grad():
                            wav_5s = wav.squeeze(1)
                            N = wav_5s.shape[1]
                            if N > 160000:
                                start = (N - 160000) // 2
                                wav_5s = wav_5s[:, start:start + 160000]
                            elif N < 160000:
                                wav_5s = F.pad(wav_5s, (0, 160000 - N))
                            perch_emb = perch_teacher.embed(wav_5s).to(device)

                    if USE_PERCH_CACHE:
                        # 伪标签段的 perch_emb 没在 cache 里 (是零向量), 必须 mask 掉
                        # 否则 MSE 会把 student_emb 拉向 0, 破坏蒸馏.
                        # 通过 source_tag 构造 (B,) mask: 仅 focal/sc 段进 distill loss.
                        distill_mask = torch.tensor(
                            [s in ("focal", "sc") for s in sr],
                            device=device, dtype=torch.float32,
                        ).view(-1, 1)
                        diff = (distill_emb.float() - perch_emb.float()) ** 2
                        distill_loss = (diff * distill_mask).sum() / (
                            distill_mask.sum() * distill_emb.shape[1] + 1e-8)
                    else:
                        distill_loss = F.mse_loss(distill_emb.float(), perch_emb.float())
                    loss         = cls_loss + ALPHA_DISTILL * distill_loss
                else:
                    distill_loss = torch.tensor(0.0, device=device)
                    loss         = cls_loss

            # ── 5c. Backward (bf16: 直接 backward; fp16: GradScaler) ──
            opt.zero_grad(set_to_none=True)
            if _USE_GRADSCALER:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            sch.step()

            el      += loss.item()
            el_cls  += cls_loss.item()
            el_dist += distill_loss.item()
            nb_count += 1

        # ── 5d. Validation (每 epoch 都跑) ──────────────────────────
        val_preds_dict = _predict_from_waveforms(m, mel_transform, val_wavs)
        val_preds      = val_preds_dict["blend"]                         # 主指标用 blend
        r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
        # 三种 head 的 non_s22 AUC 都算一下 (诊断用)
        for mode in ["clip", "fmax", "blend"]:
            r_mode = full_eval(Y_val, val_preds_dict[mode], ns22_val, TAXON_MASKS)
            r[f"ns22_{mode}"] = r_mode["non_s22_macro"]

        history["ep"].append(ep)
        history["train_loss"].append(round(el / nb_count, 5))
        history["cls_loss"].append(round(el_cls / nb_count, 5))
        history["dist_loss"].append(round(el_dist / nb_count, 5))
        history["macro"].append(r["macro_auc_all"])
        history["ns22_macro"].append(r["non_s22_macro"])
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
            history[f"ns22_{t}"].append(r[f"non_s22_{t}"])
        history["val_preds"].append(val_preds.astype(np.float32))

        # ── 5e. 跟踪 best + 早停 ─────────────────────────────────
        tag_ns22 = ""; tag_macro = ""
        if r["non_s22_macro"] > best_ns22:
            best_ns22       = r["non_s22_macro"]
            best_state_ns22 = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            tag_ns22 = " *ns22"
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1
        if r["macro_auc_all"] > best_macro:
            best_macro       = r["macro_auc_all"]
            best_state_macro = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            tag_macro = " *macro"

        dist_str = f" dist={el_dist/nb_count:.4f}" if USE_PERCH_DISTILL else ""
        print(f"    Ep{ep:02d}: loss={el/nb_count:.4f} cls={el_cls/nb_count:.4f}{dist_str} "
              f"lr={opt.param_groups[0]['lr']:.1e} | "
              f"ns22: {r['ns22_blend']:.4f} | "
              f"Av={r['non_s22_Aves']:.4f} Am={r['non_s22_Amphibia']:.4f} "
              f"In={r['non_s22_Insecta']:.4f} Ma={r['non_s22_Mammalia']:.4f} "
              f"[{time.time()-t0:.0f}s]{tag_ns22}{tag_macro}")

        # ── 5f. 早停判断 ──────────────────────────────────────────
        if EARLY_STOP_PATIENCE > 0 and no_improve_cnt >= EARLY_STOP_PATIENCE:
            print(f"    [EARLY STOP] ns22 no improve for {no_improve_cnt} epochs, "
                  f"stop at Ep{ep:02d}. best_ns22={best_ns22:.4f}")
            break

    # 清理
    del perch_teacher, m, mel_transform, spec_augment
    torch.cuda.empty_cache()
    gc.collect()
    return best_state_ns22, best_state_macro, history
