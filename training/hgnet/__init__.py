"""BirdCLEF+2026 HGNetV2-B0 Baseline (训练 + 导出 + 推理) 多文件版.

这是把 ``birdclef-2026-hgnetv2-b0-baseline-training.ipynb`` /
``birdclef-2026-hgnetv2-b0-baseline-inference.ipynb`` 两份 notebook
拆分而成的可复现 Python 包.

模块速览:
    config       - 全局配置 / 路径 / 常量 / CFG / INFER_CFG.
    utils        - set_random_seed / to_device / rank_normalize / sigmoid / device.
    data         - 标签加载 / soundscape 切片 / multi-label stratified group K-Fold.
    dataset      - BirdTrainDataset / BirdValidDataset / get_data_loader.
    transforms   - LogMelSpectrogramTransform / MixUp / dummy_mixup.
    models       - GeMPooling / AttnSEDHead / AttnSEDModel /
                   LSEPooling / LSEHead / LSEModel / CustomBCEWithLogitsLoss.
    prepare_wav  - 把官方 train_audio/*.ogg 批量预转换成 wav (复刻 ttahara 的 4 份数据集).
    train        - train_one_fold + main (训练入口).
    export       - torch -> ONNX -> OpenVINO + OOF CPU 校验.
    infer        - 推理入口, 生成 submission.csv (带 ±2.5s 偏移 TTA).
"""

__all__ = [
    "config",
    "utils",
    "data",
    "dataset",
    "transforms",
    "models",
    "prepare_wav",
    "train",
    "export",
    "infer",
]
