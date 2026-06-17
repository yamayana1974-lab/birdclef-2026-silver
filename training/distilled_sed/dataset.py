# -*- coding: utf-8 -*-
"""
dataset.py — Dataset / Sampler / 增强
=======================================

对应 notebook 的 S4 cell.

主要组件:
  1. load_int16 / load_focal / load_sc_waveform_from  — waveform cache 加载 (LRU)
  2. extract_chunk_np                                  — 切 5s 段 (左 pad)
  3. apply_aug                                         — 波形增强 (gain + noise)
  4. FocalDS                                           — focal 录音 dataset (含 3 种 MixUp)
  5. ScDS                                              — labeled soundscape dataset
  6. MixSamp                                           — 多源 batch sampler (按 SHARES 分配)
  7. collate_m / mk_sw                                 — collate fn + per-sample loss weight

★ MixSamp 是关键: 让每 batch 严格按 SHARES = {focal: 0.9, sc: 0.1} 组成,
  不是随机抽样. 这让 focal/sc 比例在每个 batch 都稳定.

★ FocalDS 里 3 种 MixUp:
  - Focal-Focal: 两个 focal clip 叠加 (MIXUP_HARD=True → label = union)
  - Focal-Soundscape: focal + labeled soundscape 叠加 (模拟域漂移, 关键)
  - 不 MixUp: 仅做 apply_aug
"""
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from config import (WAVEFORM_CACHE_DIR, SR, TRAIN_SAMPLES, NUM_CLASSES,
                    AUG_PROB, AUG_GAIN_DB_RANGE, AUG_NOISE_SNR_DB_RANGE,
                    USE_FOCAL_MIXUP, MIXUP_PROB, MIXUP_ALPHA, MIXUP_HARD,
                    USE_FOCAL_SC_MIXUP, FOCAL_SC_MIXUP_PROB, FOCAL_SC_MIXUP_ALPHA,
                    SOURCE_WEIGHTS, USE_PERCH_CACHE, PERCH_CACHE_DIR,
                    PERCH_EMBED_DIM)


# =============================================================================
# 1. waveform cache 加载 (LRU)
# =============================================================================
# 思路: cache 文件是 int16 .pt, 比 float32 .ogg 小一半.
# 用简单的 dict 当 LRU (满了 pop oldest).

def load_int16(path):
    """加载 int16 waveform tensor → float32 in [-1, 1]."""
    waveform_int16 = torch.load(path, map_location="cpu")
    return waveform_int16.float() / 32767.0


_FC = {}                                            # focal waveform cache (per-worker)
_FC_LIMIT = 200                                     # 每个 worker 上限. num_workers 多时降低防 OOM.
def load_focal(p):
    """加载 focal 波形 (numpy array). 带 LRU 缓存."""
    if p in _FC:
        return _FC[p]
    pp = WAVEFORM_CACHE_DIR / p
    if not pp.exists():
        return None
    a = load_int16(pp).numpy()
    if len(_FC) >= _FC_LIMIT:
        _FC.pop(next(iter(_FC)))                    # pop oldest
    _FC[p] = a
    return a


_SC_CACHE = {}                                      # soundscape cache (per-worker)
_SC_LIMIT = 60                                      # 每个 worker 上限
def load_sc_waveform_from(cache_dir, cache_file):
    """加载 soundscape 波形 (numpy array). 带 LRU 缓存."""
    key = str(cache_dir / cache_file)
    if key in _SC_CACHE:
        return _SC_CACHE[key]
    pp = cache_dir / cache_file
    if not pp.exists():
        return None
    a = load_int16(pp).numpy()
    if len(_SC_CACHE) >= _SC_LIMIT:
        _SC_CACHE.pop(next(iter(_SC_CACHE)))
    _SC_CACHE[key] = a
    return a


# =============================================================================
# 1b. Perch emb cache 加载 (✅ 离线预算好的 1536-d emb)
# =============================================================================
_ZERO_EMB = np.zeros(PERCH_EMBED_DIM, dtype=np.float32)
_PERCH_FC = {}                                      # focal perch emb cache
_PERCH_SC = {}                                      # sc perch emb cache


def load_focal_perch_emb(cache_rel):
    """加载 focal 文件的 perch emb (跟 waveform cache 同 layout). 找不到返回零向量."""
    if not USE_PERCH_CACHE:
        return _ZERO_EMB
    if cache_rel in _PERCH_FC:
        return _PERCH_FC[cache_rel]
    pp = PERCH_CACHE_DIR / cache_rel               # focal/<species>/<file>.pt
    if not pp.exists():
        _PERCH_FC[cache_rel] = _ZERO_EMB
        return _ZERO_EMB
    arr = torch.load(pp, map_location="cpu").float().numpy().astype(np.float32)
    if len(_PERCH_FC) >= 1000:
        _PERCH_FC.pop(next(iter(_PERCH_FC)))
    _PERCH_FC[cache_rel] = arr
    return arr


def load_sc_perch_emb(sc_cache_file, start_sec):
    """加载 sc 段的 perch emb (按 stem + start_sec 命名)."""
    if not USE_PERCH_CACHE:
        return _ZERO_EMB
    stem = Path(sc_cache_file).stem                 # 'BC2026_Train_xxxx_S08_xxx'
    key  = f"{stem}__{int(start_sec)}"
    if key in _PERCH_SC:
        return _PERCH_SC[key]
    pp = PERCH_CACHE_DIR / "sc" / f"{key}.pt"
    if not pp.exists():
        _PERCH_SC[key] = _ZERO_EMB
        return _ZERO_EMB
    arr = torch.load(pp, map_location="cpu").float().numpy().astype(np.float32)
    if len(_PERCH_SC) >= 1000:
        _PERCH_SC.pop(next(iter(_PERCH_SC)))
    _PERCH_SC[key] = arr
    return arr


# =============================================================================
# 2. 切段 + 增强
# =============================================================================

def extract_chunk_np(waveform, start_sample, n_samples):
    """
    切一个 n_samples 长的段, 太短的录音左 pad.

    左 pad 而不是右 pad: 让短 clip 的有声段对齐到 chunk 末尾,
    SpecAugment 等增强不会破坏前面的目标声.
    """
    total = len(waveform)
    if total <= n_samples:
        return np.pad(waveform, (n_samples - total, 0))
    end = start_sample + n_samples
    if end > total:
        start_sample = max(0, total - n_samples)
    return waveform[start_sample:start_sample + n_samples]


def apply_aug(w):
    """
    波形增强:
      - 50% 概率: 增益抖动 ±6 dB
      - 50% 概率: 加白噪声 (SNR 10~30 dB)
    """
    if np.random.random() < AUG_PROB:
        # Gain jitter
        w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
    if np.random.random() < AUG_PROB:
        # Add noise at random SNR
        sp = (w ** 2).mean()                          # signal power
        if sp > 1e-10:
            w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                sp / (10 ** (np.random.uniform(*AUG_NOISE_SNR_DB_RANGE) / 10))
            )
    return w


# =============================================================================
# 3. FocalDS — focal 录音 dataset (含 3 种 MixUp)
# =============================================================================

class FocalDS(Dataset):
    """
    focal recordings dataset. 每个 sample 返回 5 元组:
        (waveform, label, weight, mask, source_tag)

    其中:
      - waveform:    (1, TRAIN_SAMPLES) torch.float32
      - label:       (NUM_CLASSES,) 多标签 0/1
      - weight:      (NUM_CLASSES,) per-class loss weight (这里全 1)
      - mask:        (NUM_CLASSES,) per-class loss mask (这里全 1)
      - source_tag:  str, "focal" 或 "focal_missing"
    """
    def __init__(self, df, l2i, secondary_lookup=None,
                 sc_mixup_sources=None, fold_k=None, aug=False):
        self.df, self.l2i, self.aug = df.reset_index(drop=True), l2i, aug
        self.secondary_lookup = secondary_lookup
        self.sc_mixup_sources = sc_mixup_sources
        self.fold_k           = fold_k

    def __len__(self):
        return len(self.df)

    def _load_chunk(self, r):
        """
        加载一个 focal clip 并切到 TRAIN_SAMPLES.
        训练时随机 crop, 验证时按 start_sec 切.
        """
        w = load_focal(r["cache_file"])
        if w is None:
            return None, None
        if self.aug:
            # 随机起点
            start = (np.random.randint(0, max(1, len(w) - TRAIN_SAMPLES + 1))
                     if len(w) > TRAIN_SAMPLES else 0)
        else:
            start = int(r.get("start_sec", 0)) * SR

        ch = extract_chunk_np(w, start, TRAIN_SAMPLES)

        # 构造多标签 (primary + secondary)
        lb = np.zeros(NUM_CLASSES, dtype=np.float32)
        if str(r["primary_label"]) in self.l2i:
            lb[self.l2i[str(r["primary_label"])]] = 1.0
        if self.secondary_lookup is not None and "original_idx" in self.df.columns:
            for s in self.secondary_lookup.get(int(r["original_idx"]), []):
                if s in self.l2i:
                    lb[self.l2i[s]] = 1.0
        return ch, lb

    def __getitem__(self, i):
        r1 = self.df.iloc[i]
        ch1, lb1 = self._load_chunk(r1)

        # cache 缺失 → 返回全零, source_tag="focal_missing" (loss 权重 0)
        if ch1 is None:
            return (torch.zeros(1, TRAIN_SAMPLES),
                    torch.zeros(NUM_CLASSES),
                    torch.ones(NUM_CLASSES),
                    torch.ones(NUM_CLASSES),
                    "focal_missing",
                    torch.zeros(PERCH_EMBED_DIM))

        emb1 = load_focal_perch_emb(r1["cache_file"])

        # ── MixUp 选项 1: Focal-Focal MixUp ────────────────────────
        # 两个 focal clip 叠加, 模拟多物种共存
        if USE_FOCAL_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
            ch2 = None
            r2  = None
            for _ in range(3):                          # 最多重试 3 次找有效 clip
                j = np.random.randint(len(self.df))
                r2 = self.df.iloc[j]
                ch2, lb2 = self._load_chunk(r2)
                if ch2 is not None:
                    break
            if ch2 is not None:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                if self.aug:
                    ch_mix = apply_aug(ch_mix)
                # Hard union vs soft 加权
                lb = (np.maximum(lb1, lb2) if MIXUP_HARD
                      else (lam * lb1 + (1 - lam) * lb2))
                # ★ perch emb 跟 wav 同分量加权 (蒸馏目标也得跟着 mix 走)
                emb2 = load_focal_perch_emb(r2["cache_file"])
                emb  = (lam * emb1 + (1 - lam) * emb2).astype(np.float32)
                return (torch.from_numpy(ch_mix).unsqueeze(0),
                        torch.from_numpy(lb),
                        torch.ones(NUM_CLASSES),
                        torch.ones(NUM_CLASSES),
                        "focal",
                        torch.from_numpy(emb))

        # ── MixUp 选项 2: Focal-Soundscape MixUp (★ 关键 trick) ──
        if (USE_FOCAL_SC_MIXUP and self.aug and self.sc_mixup_sources
                and np.random.random() < FOCAL_SC_MIXUP_PROB):
            src_idx = np.random.randint(len(self.sc_mixup_sources))
            cache_dir, meta_df_sc, labels = self.sc_mixup_sources[src_idx]
            # 排除当前 fold 的 sc 数据 (防 val 泄漏)
            eligible = (meta_df_sc[meta_df_sc["fold"] != self.fold_k]
                        if self.fold_k is not None else meta_df_sc)
            if len(eligible) > 0:
                sc_row = eligible.iloc[np.random.randint(len(eligible))]
                sc_wav = load_sc_waveform_from(cache_dir, sc_row["cache_file"])
                if sc_wav is not None and len(sc_wav) >= TRAIN_SAMPLES:
                    sc_chunk = extract_chunk_np(
                        sc_wav, int(sc_row["start_sec"]) * SR, TRAIN_SAMPLES,
                    )
                    lam = np.random.beta(FOCAL_SC_MIXUP_ALPHA, FOCAL_SC_MIXUP_ALPHA)
                    ch_mix = (lam * ch1 + (1 - lam) * sc_chunk).astype(np.float32)
                    if self.aug:
                        ch_mix = apply_aug(ch_mix)
                    lb_sc = labels[int(sc_row["label_idx"])].astype(np.float32)
                    lb = (np.maximum(lb1, lb_sc) if MIXUP_HARD
                          else lam * lb1 + (1 - lam) * lb_sc)
                    emb_sc = load_sc_perch_emb(sc_row["cache_file"],
                                                int(sc_row["start_sec"]))
                    emb    = (lam * emb1 + (1 - lam) * emb_sc).astype(np.float32)
                    return (torch.from_numpy(ch_mix).unsqueeze(0),
                            torch.from_numpy(lb),
                            torch.ones(NUM_CLASSES),
                            torch.ones(NUM_CLASSES),
                            "focal",
                            torch.from_numpy(emb))

        # ── 选项 3: 不 MixUp, 只做 apply_aug ────────────────────
        if self.aug:
            ch1 = apply_aug(ch1)
        return (torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(lb1),
                torch.ones(NUM_CLASSES),
                torch.ones(NUM_CLASSES),
                "focal",
                torch.from_numpy(emb1.astype(np.float32)))


# =============================================================================
# 4. ScDS — labeled soundscape dataset
# =============================================================================

class ScDS(Dataset):
    """
    Labeled soundscape windows. 简单加载 + 切段 + 可选 aug.
    每个 sample 返回 (waveform, label, weight, mask, source_tag="sc", perch_emb).
    """
    def __init__(self, Y, sc_df, aug=False):
        self.Y, self.df, self.aug = Y, sc_df.reset_index(drop=True), aug

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav_full = (load_sc_waveform_from(WAVEFORM_CACHE_DIR, row.get("cache_file"))
                    if row.get("cache_file") else None)
        if wav_full is None:
            wav_t = torch.zeros(1, TRAIN_SAMPLES)
        else:
            chunk = extract_chunk_np(wav_full, int(row["start_sec"]) * SR, TRAIN_SAMPLES)
            if self.aug:
                chunk = apply_aug(chunk)
            wav_t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
        emb = load_sc_perch_emb(row.get("cache_file", ""), int(row["start_sec"]))
        return (wav_t,
                torch.from_numpy(self.Y[i].astype(np.float32)),
                torch.ones(NUM_CLASSES),
                torch.ones(NUM_CLASSES),
                "sc",
                torch.from_numpy(emb.astype(np.float32)))


# =============================================================================
# 5. MixSamp — 多源 batch sampler
# =============================================================================

class MixSamp(torch.utils.data.Sampler):
    """
    控制每 batch 的 source 组成比例.
        sizes:  各 source 的 dataset 长度 list, e.g. [35000, 700]
        names:  source 名 list, e.g. ["focal", "sc"]
        shares: {source 名 → 占 batch 的比例}, e.g. {"focal": 0.9, "sc": 0.1}
        bs:     batch_size
        nst:    每 epoch 步数 (= ceil(total / bs))

    每个 batch 严格按 SHARES 分配名额, 不靠随机. e.g. bs=64, shares=0.9/0.1
    → 每 batch 58 个 focal + 6 个 sc.
    """
    def __init__(self, sizes, names, shares, bs, nst, seed=0):
        self.sizes, self.names, self.bs, self.nst = sizes, names, bs, nst
        self.rng = np.random.default_rng(seed)
        # 计算每源占的 batch 名额
        per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
        total = sum(per_src)
        if total != bs:
            # 多/少了几个 → 加/减到最大那个 source 里
            per_src[int(np.argmax(per_src))] += (bs - total)
        self.per_src = per_src
        # 每源在 ConcatDataset 里的起始 idx
        self.offsets = [0]
        for s in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

    def __len__(self):
        return self.nst

    def __iter__(self):
        for _ in range(self.nst):
            batch = []
            for off, size, n in zip(self.offsets, self.sizes, self.per_src):
                if n <= 0 or size <= 0:
                    continue
                # 在该源里随机抽 n 个
                idxs = self.rng.integers(0, size, size=n)
                batch.extend([off + int(i) for i in idxs])
            self.rng.shuffle(batch)
            yield batch


def collate_m(batch):
    """自定义 collate fn (因为 source_tag 是 str 不能 stack)."""
    return (torch.stack([b[0] for b in batch]),     # waveform
            torch.stack([b[1] for b in batch]),     # label
            torch.stack([b[2] for b in batch]),     # weight
            torch.stack([b[3] for b in batch]),     # mask
            [b[4] for b in batch],                  # source_tag list
            torch.stack([b[5] for b in batch]))     # perch_emb (B, 1536)


def mk_sw(sr):
    """根据 source_tag list 构造 per-sample loss 权重 tensor."""
    return torch.tensor([SOURCE_WEIGHTS.get(s, 0.0) for s in sr],
                         dtype=torch.float32)
