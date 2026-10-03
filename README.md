# Alzheimer's Classification — MRI × PET Multimodal Pipeline

Binary classification of **Alzheimer's Disease (AD) vs. Cognitively Normal (CN)** subjects using 2D brain slices extracted from MRI and PET scans. Six backbone architectures are compared; the final cross-architecture MLP fusion experiments are in `fusion/`.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Dataset](#dataset)
- [Pre-computed Splits](#pre-computed-splits)
- [Installation](#installation)
- [Configuration](#configuration)
- [Reproducing Experiments](#reproducing-experiments)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)

---

## Overview

Each single-backbone model follows the same **3-stage pipeline**:

| Stage | Description |
|-------|-------------|
| **Stage 1** | Train an MRI backbone independently on MRI slices |
| **Stage 2** | Train a PET backbone independently on PET slices |
| **Stage 3** | Freeze both backbones; train a fusion MLP on paired MRI × PET slices |

The `fusion/` package extends this to **all 36 cross-architecture pairs** (30 heterogeneous + 6 homogeneous) using 5-fold CV × 5 random seeds = 25 runs per pair.

All splits are **subject-level** — no subject appears in more than one of train / val / test — to prevent data leakage.

---

## Repository Structure

```
.
├── Subject_splits/                 # Pre-computed subject-level split CSVs
│   ├── mri_backbone_splits.csv     # backbone training (all subjects)
│   ├── pet_backbone_splits.csv
│   ├── mri_fusion_splits.csv       # 150 overlap subjects (MRI ∩ PET)
│   └── pet_fusion_splits.csv
│
├── vgg/                            # VGG19 backbone
├── dino_v2/                        # DINOv2 ViT-B/14
├── swin/                           # Swin Transformer V2-Base
├── vit/                            # ViT-B/16
├── convnext/                       # ConvNeXtV2-Tiny
├── mambaout/                       # MambaOut-Base
│   (each folder: config.py  data.py  model.py  train.py  inference.py  plots.py  main.py)
│
├── fusion/                         # Cross-architecture MLP fusion (all 36 pairs)
│   ├── __init__.py
│   ├── config.py                   # ⚠ Kaggle paths — adapt before running locally
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── baselines.py
│   ├── stats.py
│   ├── plots.py
│   └── run_fusion.py
│
├── PreProcessingSearch/            # Preprocessing hyperparameter search notebooks
├── requirements.txt
└── README.md
```

Each single-backbone folder is **self-contained** — all imports are sibling-relative (`from config import DEVICE`), so `main.py` must be run from inside that folder.

---

## Models

### `vgg/` — VGG19
Pretrained VGG19 (`VGG19_Weights.IMAGENET1K_V1`). Classifier head replaced with a 2-layer dense block. Optimiser: **Adam**, lr = 5×10⁻⁵, 8 epochs, patience 5.

### `dino_v2/` — DINOv2 ViT-B/14
Backbone loaded via `torch.hub` (`facebookresearch/dinov2`, `dinov2_vitb14`). Projection head on top of the [CLS] token (768 → 256). Hyperparameters found via Optuna HPO. Optimiser: **AdamW** with linear warmup + cosine decay, 15 epochs, patience 7.

### `swin/` — Swin Transformer V2-Base
`torchvision.models.swin_v2_b` pretrained at 256×256. Dense block on the 1024-dim pooled feature. Optimiser: **AdamW** with linear warmup + cosine decay, 15 epochs, patience 7.

### `vit/` — ViT-B/16
`timm` ViT-B/16 (ImageNet-1K, 224×224). CLS token (768-dim) → dense block. Optimiser: **AdamW** with linear warmup + cosine decay, lr = 1×10⁻⁴, WD = 0.05, 15 epochs, patience 7.

### `convnext/` — ConvNeXtV2-Tiny
`timm` ConvNeXtV2-Tiny (FCMAE pre-trained, fine-tuned on ImageNet-22K+1K). GAP → dense block. Optimiser: **AdamW**, lr = 5×10⁻⁵, 8 epochs, patience 5.

### `mambaout/` — MambaOut-Base
`timm` MambaOut-Base (ImageNet-1K). Spatial average of feature map → dense block. Optimiser: **AdamW**, lr = 5×10⁻⁵, 8 epochs, patience 5.

### `fusion/` — Cross-Architecture MLP Fusion
Frozen backbone features (from the 6 trained backbones above) are concatenated and passed to a shallow MLP head (256 → 64 → 2, BatchNorm, Dropout). Evaluated over all 36 MRI×PET architecture pairs using 5-fold CV × 5 seeds. Optimiser: **AdamW**, lr = 1×10⁻³, WD = 1×10⁻⁴, 60 epochs, early stopping on val loss (patience 15).

> **Note**: The `fusion/config.py` contains Kaggle-specific paths (`/kaggle/input/...`, `/kaggle/working/...`). Before running locally, edit `SPLIT_DIR`, `MRI_CSV`, `PET_CSV`, `OLD_PFX`, `NEW_PFX`, `CACHE_DIR`, and the `*_DIR` output paths.

---

## Dataset

The experiments use **2D coronal slices** extracted from ADNI (Alzheimer's Disease Neuroimaging Initiative) structural MRI and FDG-PET scans. Each slice is saved as a PNG image.

Expected on-disk layout:

```
<dataset_root>/
├── MRI_Slices/
│   ├── AD/<subject_id>/*.png
│   └── CN/<subject_id>/*.png
└── PET_Slices/
    ├── AD/<subject_id>/*.png
    └── CN/<subject_id>/*.png
```

> **Access**: ADNI data requires registration at [adni.loni.usc.edu](https://adni.loni.usc.edu). This repository does not distribute any imaging data.

---

## Pre-computed Splits

The `Subject_splits/` folder contains four CSVs encoding the subject-level partitions.

| File | Subjects | Used by |
|------|----------|---------|
| `mri_backbone_splits.csv` | ~348 | MRI backbone training (Stage 1) |
| `pet_backbone_splits.csv` | ~500 | PET backbone training (Stage 2) |
| `mri_fusion_splits.csv`   | 150 | MRI side of 5-fold fusion CV |
| `pet_fusion_splits.csv`   | 150 | PET side of 5-fold fusion CV |

**CSV columns**: `subject_id`, `group` (AD/CN), `slice_path`, `split` (train/val/test).

If slice paths in the CSVs differ from your machine, use `OLD_DATA_ROOT` / `NEW_DATA_ROOT` in each `config.py` to remap them.

---

## Installation

```bash
git clone <repo-url>
cd "Transformer-based-Multimodal-Alzheimer-s-Disease-Classification-using-MRI-and-FDG-PET-scans"
python -m venv venv
# Windows: venv\Scripts\activate   |   Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

> **PyTorch + CUDA**: install the CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before `pip install -r requirements.txt`.

> **DINOv2**: `dino_v2/model.py` downloads ~330 MB from GitHub on first run via `torch.hub`.

> **timm models**: `vit/`, `convnext/`, and `mambaout/` download pretrained weights from HuggingFace Hub on first run. An internet connection is required.

---

## Configuration

### Single-backbone models (`vgg/`, `vit/`, `convnext/`, `mambaout/`)

```python
# config.py — edit these before running
OUT_DIR       = "/path/to/output"    # checkpoints + plots
SPLIT_DIR     = "/path/to/Subject_splits"
OLD_DATA_ROOT = ""                   # original path prefix in the CSVs
NEW_DATA_ROOT = ""                   # replacement prefix on this machine
```

### DINOv2 (`dino_v2/`)

```python
OUT_DIR         = "/path/to/output"
SPLIT_DIR       = "/path/to/Subject_splits"
OLD_PATH_PREFIX = ""
NEW_PATH_PREFIX = ""
```

### Swin (`swin/`)

```python
OUT_DIR = "/path/to/output"   # split CSVs are read from a separate SPLIT_DIR variable
```

### Fusion (`fusion/config.py`)

All paths are Kaggle-specific. At minimum replace:
- `SPLIT_DIR` / `MRI_CSV` / `PET_CSV` → point to local `Subject_splits/`
- `OLD_PFX` → original path prefix in the CSVs
- `NEW_PFX` → local dataset root
- `CACHE_DIR`, `CROSS_DIR`, `SAME_DIR`, `SOLO_DIR`, `BASE_DIR`, `STATS_DIR` → local output directories
- Checkpoint constants (`VGG19_MRI_CKPT`, etc.) → paths to your trained `.pth` files

---

## Reproducing Experiments

### Single-backbone models

Run each `main.py` from **inside** its own folder (sibling-relative imports require this):

```bash
cd vgg      && python main.py
cd ../swin  && python main.py
cd ../dino_v2 && python main.py
cd ../vit   && python main.py
cd ../convnext && python main.py
cd ../mambaout && python main.py
```

### Cross-architecture MLP fusion

After editing `fusion/config.py`, run from the **repository root**:

```bash
python -m fusion.run_fusion
```

This runs all 4 phases in sequence:
1. Feature caching (extract and save frozen backbone features)
2. MLP training for all 36 pairs (5-fold CV × 5 seeds)
3. Baseline evaluation (LR stacking, probability averaging, solo LR)
4. Statistical analysis

Set `SKIP_TRAINING = True` in `fusion/config.py` to skip phases 1–3 and rerun only the statistics on existing CSVs.

---

## Evaluation Metrics

| Level | Method |
|-------|--------|
| **Slice-level** | Each 2D slice treated as an independent sample |
| **Subject-level** | Softmax probabilities averaged across all slices; argmax = prediction |

Metrics reported: Accuracy, ROC-AUC, PR-AUC, sensitivity, specificity, F1.

---

## Output Files

Single-backbone runs save to `OUT_DIR`:

| File | Description |
|------|-------------|
| `mri_<model>_best.pth` | Best MRI backbone checkpoint |
| `pet_<model>_best.pth` | Best PET backbone checkpoint |
| `multimodal_best.pth` | Best fusion checkpoint |
| `*_curves.png` | Train/val loss and accuracy curves |
| `*_roc_pr.png` | ROC and PR curves |
| `*_subject_predictions.csv` | Per-subject predictions |

Fusion runs save to the directories specified in `fusion/config.py` (`CROSS_DIR`, `SAME_DIR`, `SOLO_DIR`, `BASE_DIR`, `STATS_DIR`).
