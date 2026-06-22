import random
import numpy as np
import torch

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")

# ── Paths ─────────────────────────────────────────────────────────────────────
# BASE    : root of the dataset (parent of MRI_Slices/ and PET_Slices/)
# MRI_DIR : folder containing per-subject MRI slice sub-folders
# PET_DIR : folder containing per-subject PET slice sub-folders
# OUT_DIR : where checkpoints, split CSVs, and plots will be saved
BASE    = ""
MRI_DIR = ""
PET_DIR = ""
OUT_DIR = ""

CFG = dict(
    lr            = 1e-4,
    batch_size    = 32,
    epochs        = 15,
    patience      = 7,
    dropout       = 0.4,
    dense_units   = 256,
    num_classes   = 2,
    img_size      = 256,
    val_frac      = 0.15,
    test_frac     = 0.15,
    warmup_epochs = 3,
    weight_decay  = 0.05,
    swin_feat_dim = 1024,
)

LABEL_MAP = {"AD": 1, "CN": 0}
IDX2LABEL = {0: "CN", 1: "AD"}
