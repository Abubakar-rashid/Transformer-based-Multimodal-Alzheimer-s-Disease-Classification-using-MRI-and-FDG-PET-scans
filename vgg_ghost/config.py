import os
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
# Set OUT_DIR to the folder where checkpoints and plots will be saved.
# Set SPLIT_DIR to the folder containing the pre-computed split CSVs.
OUT_DIR   = ""
SPLIT_DIR = ""

# Path remapping: if slice paths stored in the CSVs were generated on a different
# machine/filesystem, set OLD_DATA_ROOT to the original prefix and NEW_DATA_ROOT
# to the prefix on the current machine. Leave both empty if paths are already correct.
OLD_DATA_ROOT = ""
NEW_DATA_ROOT = ""

CFG = dict(
    lr          = 5e-5,
    batch_size  = 64,
    epochs      = 8,
    patience    = 5,
    dropout     = 0.5,
    dense_units = 128,
    num_classes = 2,
    img_size    = 224,
)

LABEL_MAP = {"AD": 1, "CN": 0}
IDX2LABEL = {0: "CN", 1: "AD"}
