"""
config.py — All constants, paths, pair definitions, and global settings.

Paths are Kaggle-specific. Adapt SPLIT_DIR, MRI_CSV, PET_CSV, OLD_PFX,
NEW_PFX, CACHE_DIR, *_DIR, and checkpoint constants for your environment.
"""
import os, gc, warnings, random, itertools, time, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, roc_curve, brier_score_loss, cohen_kappa_score,
)
from statsmodels.stats.multitest import multipletests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import torchvision.models as tvm
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── CONFIG — TRAINING ─────────────────────────────────────────────────────────
SKIP_TRAINING = False   # True = skip Phases 1-3, run only stats on existing CSVs

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS   = 5
AUG_VIEWS = 3
EPOCHS    = 60
BS        = 32
LR        = 1e-3
WD        = 1e-4
PATIENCE  = 15
BASE_SEED = 42
SEEDS     = [42, 0, 1, 2, 3]   # 5 MLP init seeds for stability analysis

torch.manual_seed(BASE_SEED); np.random.seed(BASE_SEED); random.seed(BASE_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLIT_DIR = "/kaggle/input/datasets/adaisdiashdh/new-splits"
MRI_CSV   = f"{SPLIT_DIR}/mri_fusion_splits.csv"
PET_CSV   = f"{SPLIT_DIR}/pet_fusion_splits.csv"
OLD_PFX   = "/hdd3/seecs/abubakar.seecs/adni/datasets"
NEW_PFX   = "/kaggle/input/datasets/jannatttt/new-mri-pet-data"

CACHE_DIR = "/kaggle/working/cache"
CROSS_DIR = "/kaggle/working/kfold_cross"   # cross-modality MLP (36 pairs)
SAME_DIR  = "/kaggle/working/kfold_same"    # same-modality MLP (30 pairs)
SOLO_DIR  = "/kaggle/working/kfold_solo"    # solo backbone LR (12 models)
BASE_DIR  = "/kaggle/working/kfold_base"    # averaging + LR-stacking baselines
STATS_DIR = "/kaggle/working/stats_all"
OUT_DIR   = CROSS_DIR   # legacy alias

for d in (CACHE_DIR, CROSS_DIR, SAME_DIR, SOLO_DIR, BASE_DIR, STATS_DIR):
    os.makedirs(d, exist_ok=True)

LABEL_MAP = {"AD": 1, "CN": 0}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# ── CONFIG — STATS ────────────────────────────────────────────────────────────
N_BOOT   = 2000    # BCa bootstrap (per-model)
N_BOOT_P = 1000    # percentile bootstrap (pairwise)
N_PERM   = 5000    # permutations for group-level tests
N_PERM_P = 500     # permutations for pairwise tests
ALPHA    = 0.05
N_TRAIN  = 120     # approx train subjects per fold
N_TEST   = 30      # approx val subjects per fold

rng = np.random.default_rng(BASE_SEED)

# ── Checkpoint paths ──────────────────────────────────────────────────────────
VGG19_MRI    = "/kaggle/input/datasets/bigbug44/mri-vgg-19baseline/mri_vgg19_best.pth"
VIT_MRI      = "/kaggle/input/datasets/bigbug44/homo-dataset/mri_vit_best.pth"
DINO_MRI     = "/kaggle/input/datasets/bigbug44/homo-dataset/mri_dinov2_best.pth"
SWIN_MRI     = "/kaggle/input/datasets/jannatttt/input-checkpoints/mri_swin_best.pth"
CONVNEXT_MRI = "/kaggle/input/models/jannatttt/convnetx-model/pytorch/default/1/convnextv2_tiny_MRI_best.pth"
MAMBAOUT_MRI = "/kaggle/input/datasets/bigbug44/homo-dataset/mambaout_base_MRI_best.pth"

VGG19_PET    = "/kaggle/input/datasets/bigbug44/homo-dataset/pet_vgg19_best.pth"
VIT_PET      = "/kaggle/input/models/adaisdiashdh/v2-checkpoints/pytorch/default/1/pet_vit_best.pth"
DINO_PET     = "/kaggle/input/datasets/jannatttt/input-checkpoints/pet_dinov2_best.pth"
SWIN_PET     = "/kaggle/input/datasets/bigbug44/homo-dataset/pet_swin_best.pth"
CONVNEXT_PET = "/kaggle/input/datasets/bigbug44/homo-dataset/convnextv2_tiny_PET_best.pth"
MAMBAOUT_PET = "/kaggle/input/models/adaisdiashdh/mamba-chkpt/pytorch/default/1/mambaout_base_PET_best.pth"

# ── Arch metadata ─────────────────────────────────────────────────────────────
ARCHS = ["vgg19", "vitb16", "dino", "swin", "convnext", "mambaout"]

ARCH_IMG = {
    "vgg19": 224, "vitb16": 224, "dino": 224,
    "swin": 256,  "convnext": 224, "mambaout": 224,
}
ARCH_DISPLAY = {
    "vgg19": "VGG19", "vitb16": "ViT-B/16", "dino": "DINOv2",
    "swin": "Swin",   "convnext": "ConvNeXtV2", "mambaout": "MambaOut",
}
MRI_CKPTS = {
    "vgg19": VGG19_MRI, "vitb16": VIT_MRI,      "dino": DINO_MRI,
    "swin":  SWIN_MRI,  "convnext": CONVNEXT_MRI, "mambaout": MAMBAOUT_MRI,
}
PET_CKPTS = {
    "vgg19": VGG19_PET, "vitb16": VIT_PET,      "dino": DINO_PET,
    "swin":  SWIN_PET,  "convnext": CONVNEXT_PET, "mambaout": MAMBAOUT_PET,
}

# ── Fusion pair definitions ───────────────────────────────────────────────────
HETERO_PAIRS = [
    dict(tag=f"{ma}_{pa}", group="hetero",
         label=f"{ARCH_DISPLAY[ma]} × {ARCH_DISPLAY[pa]}",
         b1_arch=ma, b1_mod="mri", b2_arch=pa, b2_mod="pet",
         b1_img=ARCH_IMG[ma], b2_img=ARCH_IMG[pa], out_dir=CROSS_DIR,
         mri_arch=ma, pet_arch=pa)
    for ma in ARCHS for pa in ARCHS if ma != pa
]
HOMO_PAIRS = [
    dict(tag=f"{a}_homo", group="homo",
         label=f"{ARCH_DISPLAY[a]} (Cross-Homo)",
         b1_arch=a, b1_mod="mri", b2_arch=a, b2_mod="pet",
         b1_img=ARCH_IMG[a], b2_img=ARCH_IMG[a], out_dir=CROSS_DIR,
         mri_arch=a, pet_arch=a)
    for a in ARCHS
]
CROSS_MOD_PAIRS = HETERO_PAIRS + HOMO_PAIRS   # 36

SAME_MRI_PAIRS = [
    dict(tag=f"{a}_{b}_mrimri", group="same_mri",
         label=f"{ARCH_DISPLAY[a]}-MRI × {ARCH_DISPLAY[b]}-MRI",
         b1_arch=a, b1_mod="mri", b2_arch=b, b2_mod="mri",
         b1_img=ARCH_IMG[a], b2_img=ARCH_IMG[b], out_dir=SAME_DIR)
    for a, b in itertools.combinations(ARCHS, 2)
]   # 15
SAME_PET_PAIRS = [
    dict(tag=f"{a}_{b}_petpet", group="same_pet",
         label=f"{ARCH_DISPLAY[a]}-PET × {ARCH_DISPLAY[b]}-PET",
         b1_arch=a, b1_mod="pet", b2_arch=b, b2_mod="pet",
         b1_img=ARCH_IMG[a], b2_img=ARCH_IMG[b], out_dir=SAME_DIR)
    for a, b in itertools.combinations(ARCHS, 2)
]   # 15
SAME_MOD_PAIRS = SAME_MRI_PAIRS + SAME_PET_PAIRS   # 30

ALL_FUSION_PAIRS = CROSS_MOD_PAIRS + SAME_MOD_PAIRS   # 66

SOLO_MODELS = [
    dict(tag=f"{arch}_{mod}_solo", group=f"solo_{mod}",
         label=f"{ARCH_DISPLAY[arch]} ({mod.upper()} Solo)",
         arch=arch, mod=mod, img=ARCH_IMG[arch],
         ckpt=MRI_CKPTS[arch] if mod == "mri" else PET_CKPTS[arch])
    for arch in ARCHS for mod in ["mri", "pet"]
]   # 12

ALL_PAIRS = CROSS_MOD_PAIRS   # legacy alias

ALL_LABELS = {p["tag"]: p["label"] for p in ALL_FUSION_PAIRS}
ALL_LABELS.update({s["tag"]: s["label"] for s in SOLO_MODELS})
for p in ALL_FUSION_PAIRS:
    ALL_LABELS[p["tag"] + "_avg"] = p["label"] + " (Avg)"
    ALL_LABELS[p["tag"] + "_lr"]  = p["label"] + " (LR-Stack)"

PAIR_LABELS = {p["tag"]: p["label"] for p in CROSS_MOD_PAIRS}

COLORS = [
    "#648FFF","#785EF0","#DC267F","#FE6100","#FFB000","#009E73",
    "#E69F00","#56B4E9","#CC79A7","#0072B2","#D55E00","#AA4499",
    "#332288","#117733","#44AA99","#88CCEE","#DDCC77","#CC6677",
    "#882255","#6699CC","#661100","#999933","#AA7744","#E8601C",
    "#F4A736","#1965B0","#7BAFDE","#4EB265","#CAE0AB","#F7F056",
    "#DC050C","#72190E","#114477","#4477AA","#77AADD","#AA8844",
]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 11,
})
