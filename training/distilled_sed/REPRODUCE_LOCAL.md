# 本地完全复现训练流程

目标:在你这台机器上跑出 5 个 `sed_fold{0..4}.onnx`(Tucker Distilled SED).

---

## 0. 环境检查

```powershell
python --version              # 建议 3.10+
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi                    # 看显存,12GB+ 较稳
```

显存 < 12 GB 的话,`config.py` 把 `BATCH` 从 64 调到 32(loss 收敛会稍慢但能跑).

---

## 1. 装依赖

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm scikit-learn pandas numpy librosa soundfile tqdm
pip install onnx onnxruntime-gpu tf2onnx tensorflow
```

> `tensorflow` 只在 Step 3 转 Perch 用一次,转完可以卸载省空间.

---

## 2. 生成 waveform cache(★ 必须,1 次)

```powershell
cd BirdCLEF++2026/distilled_sed_model
python build_cache.py
```

预计:
- 35,549 个 focal .pt:30–60 GB,30–45 分钟(取决于 CPU 核数)
- 66 个 sc .pt + 3 个 csv:几分钟

支持断点续跑:已 cache 的文件会跳过.

跑完应该能看到:
```
./data\waveform_cache\
    audio_cache_meta.csv               (~35549 行)
    soundscape_cache_meta.csv          (66 × 12 = 792 行)
    soundscape_file_meta.csv           (66 行)
    focal\<species>\<file>.pt           ← 几万个
    sc\<file>.pt                         ← 66 个
```

---

## 3. 转 Perch SavedModel → ONNX(★ 必须,1 次)

```powershell
python convert_perch_onnx.py
```

产出: `./data\perch_v2.onnx` (~400 MB).

输出末尾应看到:
```
ONNX inputs : [('inputs', [-1, 160000])]
ONNX outputs: [('embedding', [-1, 1536])]
Test forward OK: embedding shape = (1, 1536)
```

---

## 4. 跑训练

### 4.1 先 DEBUG 跑一遍流程(强烈建议)

```powershell
# 编辑 config.py:
#   DEBUG = True       ← 开 DEBUG
#   FOLDS = [0]        ← 只跑 fold 0
#   EPOCHS = 1
python main.py
```

预计 5–10 分钟,目的是验证整个 pipeline(数据 / 模型 / ONNX 导出 / OOF)能跑通.

跑通后会在 `output/` 看到 `fold0_best.pt` + `sed_fold0.onnx`,这时候就放心开正式训练.

### 4.2 正式训练

```powershell
# 编辑 config.py:
#   DEBUG = False
#   FOLDS = [0, 1, 2, 3, 4]
#   EPOCHS = 25
python main.py
```

预计单 fold 1.5–2 小时(RTX 4090 / 3090),5 fold 共 7–10 小时.

显存不够时:
```python
BATCH = 32     # 默认 64 → 32
```

---

## 5. 产物

跑完在 `output/`:
```
fold0_best.pt        fold1_best.pt   ... fold4_best.pt
sed_fold0.onnx       sed_fold1.onnx  ... sed_fold4.onnx   ← 你要的 5 个 ONNX
```

每个 ONNX 大约 25–35 MB,可以拷出去用.

---

## 6. 后续改模型的入口

| 想改什么 | 改哪个文件 | 关键位置 |
|---|---|---|
| Backbone (B0 → B3 / convnext / ...) | `config.py` | `BACKBONE_NAME` |
| 蒸馏开关 / 权重 | `config.py` | `USE_PERCH_DISTILL`, `ALPHA_DISTILL` |
| Mel 参数 | `config.py` | `N_MELS / N_FFT / HOP_LENGTH` |
| Loss / 训练 step | `train.py` | `train_fold` 里 step 5b |
| MixUp 策略 | `dataset.py` | `FocalDS.__getitem__` 三选一分支 |
| Head 结构 (att/cla 改 transformer 等) | `models.py` | `BirdSEDModel.__init__` & `forward` |
| 评估指标 | `eval.py` | `compute_macro_auc` / `full_eval` |

改完 backbone 维度后,`models.py` 会自动通过 dummy forward 拿到新的 `backbone_dim`,
`export_onnx.py` 也会跟着自适应,无需手改其他地方.

---

## 7. 常见问题

**Q: build_cache 中途挂了?**
A: 重跑就行,已 cache 的会跳过.

**Q: torch.load 报安全警告?**
A: 你的 PyTorch 版本对 .pt 默认 weights_only=True. 编辑 `dataset.py`:
```python
def load_int16(path):
    waveform_int16 = torch.load(path, map_location="cpu", weights_only=True)
    return waveform_int16.float() / 32767.0
```

**Q: Perch ONNX 推理慢?**
A: 默认走 `CUDAExecutionProvider`,如果不行 fallback 到 CPU 会拖慢训练.
可以在 `models.py` `PerchTeacher.__init__` 强制用 GPU,失败就报错而不是 fallback.

**Q: 想验证训练是不是收敛?**
A: 看每 epoch 打印的 `ns22: X.XXXX`,作者的 baseline 大约能到 0.85–0.88,
最终 OOF 看 `print_oof_summary` 那段.
