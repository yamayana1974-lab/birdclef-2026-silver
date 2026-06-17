# -*- coding: utf-8 -*-
"""
eval.py — 评估指标 + OOF 汇总
==============================

对应 notebook 的 S3 部分 (compute_macro_auc / full_eval) + S7 (OOF summary).

提供:
  - compute_macro_auc: 跳过全 0 / 全 1 列, 返回 (mean_auc, n_evaluable)
  - full_eval:        汇总 5 个指标: 全集 / non_S22 / non_S22 × 4 个纲

★ non_S22 是主指标:
  S22 站点有已知标签噪声, 排除它的评估更可靠 (作者经验).
  比赛主排行榜也对应 non_S22 性能.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
    """
    Macro-averaged AUC.
    跳过没正样本或全部正样本的列 (这些类 AUC 没定义).

    Args:
      y_true:     (N, C) 0/1 多标签
      y_pred:     (N, C) 概率
      mask:       (N,) bool, 只评估这些行 (e.g. non_S22_mask)
      class_mask: list/array, 只评估这些列 (e.g. Aves 物种 idx)

    Returns:
      (mean_auc, n_evaluable_classes)
    """
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if class_mask is not None:
        y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]

    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):    # 全 0 或全 1 → 跳
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue

    return (np.mean(aucs) if aucs else float("nan")), len(aucs)


def full_eval(y_true, y_pred, ns22, tm):
    """
    完整评估 (5 个指标):
      macro_auc_all:  全部 sc 段
      non_s22_macro:  排除 S22 后的 (★ 主指标)
      non_s22_Aves:   排除 S22 + 只看 Aves 类
      non_s22_Amphibia / Insecta / Mammalia / Reptilia 同理

    Args:
      y_true: (N, 234) 真值
      y_pred: (N, 234) 概率
      ns22:   (N,) bool, non_S22 mask
      tm:     {纲名 → 列 idx 数组} (TAXON_MASKS)

    Returns:
      dict, 各指标四舍五入到 4 位 + 评估的类数
    """
    r = {}

    a, n = compute_macro_auc(y_true, y_pred)
    r["macro_auc_all"], r["n_all"] = round(a, 4), n

    a, n = compute_macro_auc(y_true, y_pred, mask=ns22)
    r["non_s22_macro"], r["n_ns22"] = round(a, 4), n

    for t, cm in tm.items():
        a, n = compute_macro_auc(y_true, y_pred, mask=ns22, class_mask=cm)
        r[f"non_s22_{t}"] = round(a, 4)

    return r


def print_oof_summary(oof_preds, Y_SC, non_s22_mask_sc, TAXON_MASKS, all_hist,
                      N_FOLDS, sc_cache_meta):
    """
    跑完所有 fold 后打印 OOF 汇总:
      1. 5 个指标的 OOF AUC
      2. 每 epoch 的 pooled non-S22 AUC (看训练曲线)

    Args:
      oof_preds:       (n_sc, 234) OOF 概率 (best-ns22 checkpoint)
      Y_SC:            (n_sc, 234) 真值
      non_s22_mask_sc: (n_sc,) bool
      TAXON_MASKS:     {纲名 → 列 idx}
      all_hist:        {fold_k: history dict}
      N_FOLDS:         5
      sc_cache_meta:   sc 元数据 (用来 reshape fold)
    """
    has = ~np.isnan(oof_preds[:, 0])
    if has.sum() > 0:
        r_all = full_eval(Y_SC[has], oof_preds[has], non_s22_mask_sc[has], TAXON_MASKS)
        print("=" * 60)
        print("OOF RESULTS (best-ns22 checkpoints)")
        print("=" * 60)
        print(f"  macro AUC (all):     {r_all['macro_auc_all']:.4f}")
        print(f"  macro AUC (non-S22): {r_all['non_s22_macro']:.4f}")
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
            print(f"    {t:<12}: {r_all.get(f'non_s22_{t}', float('nan')):.4f}")

    # Per-epoch pooled non-S22 AUC (看每个 epoch 的训练曲线)
    print("\nPer-epoch pooled non-S22 AUC:")
    fold_true, fold_ns22 = {}, {}
    for fk in range(N_FOLDS):
        vm = sc_cache_meta["fold"].values == fk
        fold_true[fk] = Y_SC[vm]
        fold_ns22[fk] = non_s22_mask_sc[vm]

    n_eps = [len(all_hist[k]["val_preds"]) for k in range(N_FOLDS) if k in all_hist]
    max_ep = min(n_eps) if n_eps else 0
    for ep in range(max_ep):
        pp = np.concatenate([all_hist[k]["val_preds"][ep]
                              for k in range(N_FOLDS) if k in all_hist])
        pt = np.concatenate([fold_true[k]
                              for k in range(N_FOLDS) if k in all_hist])
        pm = np.concatenate([fold_ns22[k]
                              for k in range(N_FOLDS) if k in all_hist])
        ns, _ = compute_macro_auc(pt, pp, mask=pm)
        print(f"  Ep{ep:02d}: {ns:.4f}")
