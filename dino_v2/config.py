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
# OUT_DIR : where checkpoints and plots will be saved
# SPLIT_DIR: folder containing the pre-computed split CSVs
BASE      = ""
MRI_DIR   = ""
PET_DIR   = ""
OUT_DIR   = ""
SPLIT_DIR = ""

# Path remapping: if absolute paths baked into the split CSVs differ from the
# current machine, set OLD_PATH_PREFIX (original) and NEW_PATH_PREFIX (current).
# Leave both empty if paths are already correct.
OLD_PATH_PREFIX = ""
NEW_PATH_PREFIX = ""

# ── Shared / architecture settings ────────────────────────────────────────────
CFG_SHARED = dict(
    epochs        = 15,
    patience      = 7,
    num_classes   = 2,
    img_size      = 224,
    dinov2_model  = "dinov2_vitb14",
    vit_feat_dim  = 768,
)

# MRI backbone: HPO Trial #1 (val_acc = 0.7692)
CFG_MRI = dict(
    lr            = 2.2948683681130543e-05,
    weight_decay  = 0.0003511356313970409,
    dropout       = 0.17336180394137352,
    dense_units   = 256,
    batch_size    = 32,
    warmup_epochs = 5,
)

# PET backbone: HPO Trial #14 (val_acc = 0.8753)
CFG_PET = dict(
    lr            = 2.5507545599395824e-05,
    weight_decay  = 0.014105900001541825,
    dropout       = 0.22642705359216325,
    dense_units   = 256,
    batch_size    = 32,
    warmup_epochs = 5,
)

# Fusion MLP: HPO Trial #0 (val_acc = 0.7126, best so far)
# NOTE: Replace these values with the final best trial once HPO completes.
CFG_FUSION = dict(
    lr            = 4.3284502212938785e-05,
    weight_decay  = 0.07114476009343425,
    dropout       = 0.39279757672456206,
    dense_units   = 128,
    batch_size    = 32,
    warmup_epochs = 3,
)

LABEL_MAP = {"AD": 1, "CN": 0}
IDX2LABEL = {0: "CN", 1: "AD"}
