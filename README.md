# Alzheimer's Classification — MRI × PET Multimodal Pipeline

Binary classification of **Alzheimer's Disease (AD) vs. Cognitively Normal (CN)** subjects using 2D brain slices extracted from MRI and PET scans. Five deep learning architectures are compared under a common 3-stage training protocol.

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

Each model follows the same **3-stage pipeline**:

| Stage | Description |
|-------|-------------|
| **Stage 1** | Train an MRI backbone independently on MRI slices |
| **Stage 2** | Train a PET backbone independently on PET slices |
| **Stage 3** | Freeze both backbones; train a fusion MLP on paired MRI × PET slices |

All splits are **subject-level** — no subject appears in more than one of train / val / test — to prevent data leakage. Evaluation is reported at both **slice level** and **subject level** (majority-vote aggregation across all slices belonging to a subject).

---

## Repository Structure

```
Alzheimers Classification/
│
├── splits/                         # Pre-computed subject-level split CSVs
│   ├── mri_backbone_splits.csv
│   ├── pet_backbone_splits.csv
│   ├── mri_fusion_splits.csv
│   └── pet_fusion_splits.csv
│
├── vgg/                            # Standard pretrained VGG19
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── plots.py
│   └── main.py
│
├── vgg_sepconv/                    # VGG19 with depthwise-separable convolutions
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── plots.py
│   └── main.py
│
├── vgg_ghost/                      # VGG19 with Ghost convolutions
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── plots.py
│   └── main.py
│
├── dino_v2/                        # DINOv2 ViT-B/14 (HPO-tuned)
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── plots.py
│   └── main.py
│
├── swin/                           # Swin Transformer V2-Base
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── plots.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

Each model folder is a **self-contained Python package**. All imports are sibling-relative (e.g. `from config import DEVICE`), so each `main.py` must be run from inside its own folder.

---

## Models

### `vgg/` — VGG19 (pretrained)
Standard VGG19 loaded with ImageNet weights (`VGG19_Weights.IMAGENET1K_V1`). The classifier head is replaced with a two-layer dense block. Optimiser: **Adam**, lr = 5e-5, 8 epochs.

### `vgg_sepconv/` — VGG19 + Separable Convolutions
All 3×3 convolutions in VGG19 are replaced with **depthwise-separable** convolutions (depthwise 3×3 + pointwise 1×1) plus BatchNorm. Trained from random initialisation. Same optimiser and schedule as `vgg/`.

### `vgg_ghost/` — VGG19 + Ghost Convolutions
All 3×3 convolutions replaced with **GhostConv** blocks (cheap 1×1 primary + depthwise secondary branch, concatenated). Weights warm-started by copying the centre pixel of each pretrained VGG19 3×3 kernel into the 1×1 primary branch. Same optimiser and schedule as `vgg/`.

### `dino_v2/` — DINOv2 ViT-B/14
Backbone loaded via `torch.hub` (`facebookresearch/dinov2`, `dinov2_vitb14`). A lightweight projection head sits on top of the [CLS] token (dim = 768). Hyperparameters were found via **Optuna HPO** separately for MRI, PET, and the fusion head. Optimiser: **AdamW** with linear warmup + cosine decay, 15 epochs.

### `swin/` — Swin Transformer V2-Base
`torchvision.models.swin_v2_b` pretrained at 256×256. The classification head is replaced with a dense block on top of the 1024-dim pooled feature vector. Optimiser: **AdamW** with linear warmup + cosine decay, 15 epochs.

---

## Dataset

The experiments use **2D axial slices** extracted from ADNI (Alzheimer's Disease Neuroimaging Initiative) structural MRI and PET scans. Each slice is saved as a PNG image.

Expected on-disk layout:

```
<dataset_root>/
├── MRI_Slices/
│   ├── AD/
│   │   ├── <subject_id>/
│   │   │   ├── slice_001.png
│   │   │   └── ...
│   │   └── ...
│   └── CN/
│       ├── <subject_id>/
│       │   └── ...
│       └── ...
└── PET_Slices/
    ├── AD/
    │   └── ...
    └── CN/
        └── ...
```

> **Access**: ADNI data requires registration and approval at [adni.loni.usc.edu](https://adni.loni.usc.edu). This repository does not distribute any imaging data.

---

## Pre-computed Splits

The `splits/` folder contains four CSV files that encode the subject-level train / val / test partition used across all experiments. Using the same splits ensures fair comparison between models.

| File | Used by |
|------|---------|
| `mri_backbone_splits.csv` | MRI backbone training (Stages 1 of all models) |
| `pet_backbone_splits.csv` | PET backbone training (Stages 2 of all models) |
| `mri_fusion_splits.csv`   | MRI side of the fusion dataset (Stage 3) |
| `pet_fusion_splits.csv`   | PET side of the fusion dataset (Stage 3) |

**CSV schema** (minimum required columns):

| Column | Description |
|--------|-------------|
| `subject_id` | Unique subject identifier |
| `filepath` | Absolute path to the slice PNG |
| `label` | Class label (`AD` or `CN`) |
| `split` | Partition assignment (`train`, `val`, or `test`) |

If your slice files live at different absolute paths than what is recorded in the CSVs (e.g. you downloaded them to a different machine), use the `OLD_DATA_ROOT` / `NEW_DATA_ROOT` remapping fields in each model's `config.py` — see [Configuration](#configuration).

The fusion CSVs cover only the **overlap subjects** that have both MRI and PET data available, so the subject counts will be smaller than the backbone CSVs.

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd "Alzheimers Classification"

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **PyTorch with CUDA**: the `requirements.txt` installs the CPU-only wheel by default on some platforms. For GPU support install the appropriate CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

### DINOv2 — internet access required (first run only)

`dino_v2/model.py` loads the backbone via `torch.hub.load("facebookresearch/dinov2", ...)`. On the first run this downloads ~330 MB from the internet and caches it in `~/.cache/torch/hub/`. Subsequent runs use the local cache.

---

## Configuration

Every model has a `config.py` that you **must edit before running**. The path variables are intentionally left empty so no private paths are committed to the repository.

### VGG variants (`vgg/`, `vgg_sepconv/`, `vgg_ghost/`)

```python
# config.py

OUT_DIR   = "/path/to/output"   # where .pth checkpoints and plots are saved
SPLIT_DIR = "/path/to/splits"   # folder containing the four split CSVs

# Only needed if the file paths inside the CSVs differ from your machine:
OLD_DATA_ROOT = ""   # original prefix (leave empty if not needed)
NEW_DATA_ROOT = ""   # replacement prefix (leave empty if not needed)
```

### DINOv2 (`dino_v2/`)

```python
# config.py

BASE      = "/path/to/dataset"              # dataset root
MRI_DIR   = "/path/to/dataset/MRI_Slices"
PET_DIR   = "/path/to/dataset/PET_Slices"
OUT_DIR   = "/path/to/output"
SPLIT_DIR = "/path/to/splits"

# Only needed if CSV paths differ from current machine:
OLD_PATH_PREFIX = ""
NEW_PATH_PREFIX = ""
```

### Swin (`swin/`)

```python
# config.py

BASE    = "/path/to/dataset"
MRI_DIR = "/path/to/dataset/MRI_Slices"
PET_DIR = "/path/to/dataset/PET_Slices"
OUT_DIR = "/path/to/output"   # split CSVs are also read from / written to OUT_DIR
```

> The Swin model reads its split CSVs from `OUT_DIR` (not a separate `SPLIT_DIR`) because it was originally designed to generate and save splits at runtime in the same output directory.

---

## Reproducing Experiments

Each model is run independently by executing its `main.py` **from inside the model's folder**:

```bash
# Example: run the standard VGG19 experiment
cd vgg
python main.py
```

Repeat for each model:

```bash
cd ../vgg_sepconv && python main.py
cd ../vgg_ghost   && python main.py
cd ../dino_v2     && python main.py
cd ../swin        && python main.py
```

> The modules use **sibling-relative imports** (`from config import ...`). Running `python vgg/main.py` from the parent directory will fail with an `ImportError`. Always `cd` into the model folder first.

### Expected runtime (approximate, single A100 GPU)

| Model | Stage 1 (MRI) | Stage 2 (PET) | Stage 3 (Fusion) |
|-------|---------------|---------------|------------------|
| VGG19 variants | ~15 min | ~15 min | ~20 min |
| DINOv2 | ~40 min | ~40 min | ~30 min |
| Swin V2-B | ~35 min | ~35 min | ~30 min |

---

## Evaluation Metrics

Results are reported at two granularities:

| Level | Method |
|-------|--------|
| **Slice-level** | Each 2D slice is treated as an independent sample |
| **Subject-level** | Softmax probabilities are averaged across all slices of a subject; the argmax is the subject prediction |

Metrics reported for each level:

- **Accuracy**
- **ROC-AUC** (one-vs-rest, AD as positive class)
- **PR-AUC** (area under precision-recall curve)

A summary table is printed at the end of each `main.py` run, and per-subject predictions are saved to a CSV in `OUT_DIR`.

---

## Output Files

After a successful run each model saves the following to `OUT_DIR`:

| File | Description |
|------|-------------|
| `mri_<model>_best.pth` | Best MRI backbone checkpoint (by val accuracy) |
| `pet_<model>_best.pth` | Best PET backbone checkpoint |
| `multimodal_<model>_best.pth` | Best fusion model checkpoint |
| `multimodal_<model>_final.pth` | Final fusion model state dict |
| `*_curves.png` | Training / validation loss & accuracy curves |
| `*_roc_pr.png` | ROC and Precision-Recall curves |
| `*_subject_predictions.csv` | Per-subject predicted label, true label, and mean probability |
