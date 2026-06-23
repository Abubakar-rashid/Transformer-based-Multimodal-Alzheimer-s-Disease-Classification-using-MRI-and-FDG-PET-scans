
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import LambdaLR

# ══════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
OUT_DIR    = ""   # folder where results, checkpoints, and plots will be saved
SPLIT_CSV  = ""   # path to pet_backbone_splits.csv

# Path remapping: if absolute paths baked into the CSV differ from your machine,
# set OLD_PATH_PREFIX to the original prefix and NEW_PATH_PREFIX to the current one.
# Leave both empty if paths are already correct.
OLD_PATH_PREFIX = ""
NEW_PATH_PREFIX = ""

os.makedirs(OUT_DIR, exist_ok=True)

# ── Model configs — identical to MRI search ──────────────────
CFG_SWIN = dict(
    lr=1e-4, batch_size=32, epochs=5, patience=4,
    dropout=0.4, dense_units=256, img_size=256,
    warmup_epochs=2, weight_decay=0.05,
    swin_feat_dim=1024,
)

CFG_DINO = dict(
    lr=2.5e-5, batch_size=32, epochs=5, patience=4,
    dropout=0.22, dense_units=256, img_size=224,
    warmup_epochs=2, weight_decay=0.014,
    dinov2_model="dinov2_vitb14", vit_feat_dim=768,
)

LABEL_MAP  = {"AD": 1, "CN": 0}
IDX2LABEL  = {0: "CN", 1: "AD"}

# ══════════════════════════════════════════════════════════════
# TOP 12 PET RECIPES  (ranked by VGG-19 test_su_roc)
# ══════════════════════════════════════════════════════════════
RECIPES = [
    ("C03_bilateral",                   ["bilateral"]),                               # 0.9917
    ("E08_gaussian_minmax",             ["gaussian_blur", "minmax"]),                 # 0.9903
    ("P07_background_mask",             ["background_mask"]),                         # 0.9903 [PET]
    ("P08_suv_clip_clahe",              ["suv_clip", "clahe"]),                       # 0.9903 [PET]
    ("B04_minmax",                      ["minmax"]),                                  # 0.9890
    ("G02_gamma_dark_zscore",           ["gamma_dark", "zscore"]),                    # 0.9862
    ("H05_gamma_dark_bilateral_zscore", ["gamma_dark", "bilateral", "zscore"]),       # 0.9848
    ("E07_bilateral_minmax",            ["bilateral", "minmax"]),                     # 0.9834
    ("B03_zscore",                      ["zscore"]),                                  # 0.9821
    ("P04_jet_colormap",                ["jet_colormap"]),                            # 0.9807 [PET]
    ("P06_percentile_stretch",          ["percentile_stretch"]),                      # 0.9793 [PET]
    ("A01_raw",                         []),                                          # 0.9779 baseline
]

# ══════════════════════════════════════════════════════════════
# PREPROCESSING OPERATIONS
# ══════════════════════════════════════════════════════════════

# ── MRI-carried-over ops (grayscale float32 [0,1] → float32 [0,1]) ──

def apply_clahe(img):
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8).astype(np.float32) / 255.0

def apply_zscore(img):
    mean, std = img.mean(), img.std()
    if std < 1e-6:
        return img
    return np.clip((img - mean) / std / 6.0 + 0.5, 0, 1)

def apply_minmax(img):
    vmin, vmax = img.min(), img.max()
    if vmax - vmin < 1e-6:
        return img
    return (img - vmin) / (vmax - vmin)

def apply_gamma_dark(img):
    return np.power(img, 1.5)

def apply_gamma_bright(img):
    return np.power(img, 0.7)

def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 1.0)

def apply_median_filter(img):
    img_uint8 = (img * 255).astype(np.uint8)
    return cv2.medianBlur(img_uint8, 5).astype(np.float32) / 255.0

def apply_bilateral(img):
    img_uint8 = (img * 255).astype(np.uint8)
    return cv2.bilateralFilter(img_uint8, 9, 75, 75).astype(np.float32) / 255.0

def apply_edge_enhance(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    return np.clip(img + 0.5 * (img - blurred), 0, 1)

def apply_histogram_eq(img):
    img_uint8 = (img * 255).astype(np.uint8)
    return cv2.equalizeHist(img_uint8).astype(np.float32) / 255.0

# ── PET-specific ops ─────────────────────────────────────────
# NOTE: these ops accept a float32 [0,1] grayscale array and return
#       a float32 [0,1] array.  Colormap ops return an RGB float32
#       array of shape (H, W, 3).

def apply_suv_clip(img):
    """
    SUV window clip to [0, 4] range (≈ 170/255 of full scale).
    Suppresses hot-spot outliers (bladder, kidneys) and preserves cortical
    uptake — the region most relevant for AD diagnosis.
    """
    clip_val = 170.0 / 255.0          # ≈ SUV 4 assuming max SUV ≈ 6
    return np.clip(img, 0, clip_val) / clip_val

def apply_suv_clip_tight(img):
    """Tighter clip to [0, 2.5] (≈ 106/255)."""
    clip_val = 106.0 / 255.0
    return np.clip(img, 0, clip_val) / clip_val

def apply_hot_colormap(img):
    """
    Hot false-colour LUT (black→red→yellow→white).
    Returns an RGB float32 [0,1] array (H, W, 3).
    """
    arr = (img * 255).astype(np.uint8)
    hot = cv2.applyColorMap(arr, cv2.COLORMAP_HOT)   # BGR
    rgb = cv2.cvtColor(hot, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

def apply_jet_colormap(img):
    """
    Jet false-colour LUT (blue→green→red).
    Returns an RGB float32 [0,1] array (H, W, 3).
    """
    arr = (img * 255).astype(np.uint8)
    jet = cv2.applyColorMap(arr, cv2.COLORMAP_JET)   # BGR
    rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

def apply_log_compress(img):
    """
    Log intensity compression: out = log(1 + pixel*255) / log(256).
    Redistributes dynamic range toward darker cortical uptake values.
    """
    arr = img * 255.0
    log_arr = np.log1p(arr) / np.log(256.0)
    return np.clip(log_arr, 0, 1)

def apply_background_mask(img):
    """
    Zero out background voxels below Otsu threshold (PET skull-strip).
    """
    arr_uint8 = (img * 255).astype(np.uint8)
    _, mask = cv2.threshold(arr_uint8, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked = cv2.bitwise_and(arr_uint8, arr_uint8, mask=mask)
    return masked.astype(np.float32) / 255.0

def apply_percentile_stretch(img):
    """
    Robust contrast stretch: rescale [p2, p98] → [0, 1].
    Handles bright PET outliers (bladder, kidneys) without clipping.
    """
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    if p98 - p2 < 1e-6:
        p98 = p2 + 1.0
    return np.clip((img - p2) / (p98 - p2), 0, 1)


OP_MAP = {
    # MRI-carried-over
    "clahe":            apply_clahe,
    "histogram_eq":     apply_histogram_eq,
    "zscore":           apply_zscore,
    "minmax":           apply_minmax,
    "gamma_dark":       apply_gamma_dark,
    "gamma_bright":     apply_gamma_bright,
    "gaussian_blur":    apply_gaussian_blur,
    "median_filter":    apply_median_filter,
    "bilateral":        apply_bilateral,
    "edge_enhance":     apply_edge_enhance,
    # PET-specific
    "suv_clip":           apply_suv_clip,
    "suv_clip_tight":     apply_suv_clip_tight,
    "hot_colormap":       apply_hot_colormap,
    "jet_colormap":       apply_jet_colormap,
    "log_compress":       apply_log_compress,
    "background_mask":    apply_background_mask,
    "percentile_stretch": apply_percentile_stretch,
}

# Colormap ops return (H, W, 3) RGB — track them so the pipeline handles
# them correctly.
COLORMAP_OPS = {"hot_colormap", "jet_colormap"}


def apply_recipe(img_pil, ops_list):
    """
    Apply a preprocessing pipeline to a PIL image.

    Handles both grayscale (H, W) and colormap (H, W, 3) outputs
    transparently.  Returns a PIL.Image (RGB, 0-255).

    Args:
        img_pil  : PIL.Image (any mode)
        ops_list : list of operation name strings

    Returns:
        PIL.Image (RGB)
    """
    if not ops_list:
        return img_pil.convert("RGB")

    img = np.array(img_pil.convert("L")).astype(np.float32) / 255.0
    is_rgb = False

    for op_name in ops_list:
        if op_name not in OP_MAP:
            continue
        result = OP_MAP[op_name](img)
        if op_name in COLORMAP_OPS:
            # result is (H, W, 3) float32 [0,1]
            img = result
            is_rgb = True
        else:
            img = result
            is_rgb = False   # remains grayscale

    if is_rgb:
        # (H, W, 3) float32 → PIL RGB
        arr_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_uint8, mode="RGB")
    else:
        # (H, W) float32 → PIL RGB (stacked grayscale)
        arr_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_uint8, mode="L").convert("RGB")


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
def load_split_csv(csv_path):
    """Load pre-computed subject-level splits."""
    df = pd.read_csv(csv_path)
    assert {"subject_id", "group", "slice_path", "split"}.issubset(df.columns), \
        f"CSV must have columns: subject_id, group, slice_path, split"

    if NEW_PATH_PREFIX and OLD_PATH_PREFIX != NEW_PATH_PREFIX:
        df["slice_path"] = df["slice_path"].str.replace(
            OLD_PATH_PREFIX, NEW_PATH_PREFIX, regex=False)

    print(f"Loaded {len(df)} slices | {df['subject_id'].nunique()} subjects")
    for s in ["train", "val", "test"]:
        n = len(df[df["split"] == s])
        print(f"  {s:>5}: {n:>5} slices")

    return df


def get_transforms(img_size, split="train"):
    """ImageNet-style transforms (applied AFTER preprocessing recipe)."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class SliceDataset(Dataset):
    """PET slice dataset with on-the-fly preprocessing."""

    def __init__(self, df, ops_list, transform=None):
        self.df        = df.reset_index(drop=True)
        self.ops_list  = ops_list
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["slice_path"])         # keep original mode
        img   = apply_recipe(img, self.ops_list)      # always returns RGB PIL
        label = LABEL_MAP[row["group"]]
        if self.transform:
            img = self.transform(img)
        return img, label, row["subject_id"]


def make_loaders(df, ops_list, batch_size, img_size):
    """Create train / val / test DataLoaders."""
    loaders = {}
    for split in ["train", "val", "test"]:
        df_split = df[df["split"] == split]
        dataset  = SliceDataset(df_split, ops_list,
                                get_transforms(img_size, split))
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=2, pin_memory=True)
    return loaders


# ══════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════
class SwinExtractor(nn.Module):
    """Swin Transformer V2-Base.

    Swin outputs features in channels-last format [B, H, W, C], so we pool
    over dims [1, 2] (H, W) — NOT [2, 3] — to obtain [B, C].
    """

    def __init__(self, feat_dim, dense_units, dropout, num_classes):
        super().__init__()
        _base = models.swin_v2_b(weights="DEFAULT")
        self.features = _base.features
        self.norm     = _base.norm
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_out = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = x.mean(dim=[1, 2])   # [B, H, W, C] → [B, C]  ✅
        x = self.proj(x)
        return self.fc_out(x)


class DINOv2Extractor(nn.Module):
    """DINOv2 ViT-B/14 with frozen backbone."""

    def __init__(self, model_name, feat_dim, dense_units, dropout, num_classes):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_out = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)   # [B, feat_dim]
        x = self.proj(x)
        return self.fc_out(x)


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════
def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear LR warmup then constant."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss   = 0.0
    preds_all    = []
    labels_all   = []
    probs_all    = []
    subjects_all = []

    with torch.no_grad():
        for imgs, labels, subjects in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs      = model(imgs)
            loss         = criterion(outputs, labels)

            total_loss  += loss.item() * imgs.size(0)
            probs        = torch.softmax(outputs, dim=1)[:, 1]

            preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())
            subjects_all.extend(subjects)

    preds_all  = np.array(preds_all)
    labels_all = np.array(labels_all)
    probs_all  = np.array(probs_all)

    loss = total_loss / len(labels_all)
    acc  = (preds_all == labels_all).mean()

    # Subject-level aggregation
    df_agg = pd.DataFrame({
        "subject": subjects_all,
        "label":   labels_all,
        "prob":    probs_all,
    }).groupby("subject").agg({"label": "first", "prob": "mean"})

    su_labels = df_agg["label"].values
    su_probs  = df_agg["prob"].values
    su_acc    = ((su_probs >= 0.5).astype(int) == su_labels).mean()
    fpr, tpr, _ = roc_curve(su_labels, su_probs)
    su_roc    = auc(fpr, tpr)

    return loss, acc, su_acc, su_roc


def fit(model, loaders, lr, epochs, patience, device, weight_decay, warmup_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_warmup_scheduler(optimizer, warmup_epochs, epochs)

    best_val_roc     = 0.0
    patience_counter = 0
    best_state       = None

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, loaders["train"], criterion, optimizer, device)
        va_loss, va_acc, va_su_acc, va_roc = evaluate(
            model, loaders["val"], criterion, device)
        scheduler.step()

        if va_roc > best_val_roc:
            best_val_roc     = va_roc
            patience_counter = 0
            best_state       = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            marker = "✅"
        else:
            patience_counter += 1
            marker = ""

        print(f"  ep {ep:02d} | tr_acc {tr_acc:.3f} | va_acc {va_acc:.3f} | "
              f"va_su_acc {va_su_acc:.3f} | va_roc {va_roc:.3f} {marker}")

        if patience_counter >= patience:
            print(f"  early stop @ ep {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ══════════════════════════════════════════════════════════════
# MAIN EXPERIMENT LOOP
# ══════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 80)
    print("PET PREPROCESSING COMPARISON — SWIN-V2 & DINOv2")
    print("=" * 80)
    print(f"Testing {len(RECIPES)} preprocessing recipes on 2 architectures")
    print(f"Total runs: {len(RECIPES) * 2}  (5-epoch fast validation per recipe)\n")

    df = load_split_csv(SPLIT_CSV)
    results = []

    for recipe_idx, (recipe_name, ops_list) in enumerate(RECIPES, 1):
        print(f"\n{'=' * 80}")
        print(f"[{recipe_idx:02d}/{len(RECIPES)}] {recipe_name}")
        print(f"ops: {ops_list if ops_list else '(none)'}")
        print("=" * 80)

        # ──────────────────────────────────────────────────────
        # SWIN TRANSFORMER V2-BASE
        # ──────────────────────────────────────────────────────
        print(f"\n▸ Swin-V2-B (img_size={CFG_SWIN['img_size']})")

        swin_loaders = make_loaders(
            df, ops_list, CFG_SWIN["batch_size"], CFG_SWIN["img_size"])

        swin_model = SwinExtractor(
            feat_dim    = CFG_SWIN["swin_feat_dim"],
            dense_units = CFG_SWIN["dense_units"],
            dropout     = CFG_SWIN["dropout"],
            num_classes = 2,
        ).to(DEVICE)

        swin_save = os.path.join(OUT_DIR, f"swin_{recipe_name}.pth")

        swin_model = fit(
            swin_model, swin_loaders,
            lr            = CFG_SWIN["lr"],
            epochs        = CFG_SWIN["epochs"],
            patience      = CFG_SWIN["patience"],
            device        = DEVICE,
            weight_decay  = CFG_SWIN["weight_decay"],
            warmup_epochs = CFG_SWIN["warmup_epochs"],
        
        )

        _, _, swin_test_acc, swin_test_roc = evaluate(
            swin_model, swin_loaders["test"], nn.CrossEntropyLoss(), DEVICE)

        print(f"  → test_su_acc={swin_test_acc:.4f} | test_su_roc={swin_test_roc:.4f}")

        # ──────────────────────────────────────────────────────
        # DINOv2 ViT-B/14
        # ──────────────────────────────────────────────────────
        print(f"\n▸ DINOv2 ViT-B/14 (img_size={CFG_DINO['img_size']})")

        dino_loaders = make_loaders(
            df, ops_list, CFG_DINO["batch_size"], CFG_DINO["img_size"])

        dino_model = DINOv2Extractor(
            model_name  = CFG_DINO["dinov2_model"],
            feat_dim    = CFG_DINO["vit_feat_dim"],
            dense_units = CFG_DINO["dense_units"],
            dropout     = CFG_DINO["dropout"],
            num_classes = 2,
        ).to(DEVICE)

        dino_save = os.path.join(OUT_DIR, f"dino_{recipe_name}.pth")

        dino_model = fit(
            dino_model, dino_loaders,
            lr            = CFG_DINO["lr"],
            epochs        = CFG_DINO["epochs"],
            patience      = CFG_DINO["patience"],
            device        = DEVICE,
            weight_decay  = CFG_DINO["weight_decay"],
            warmup_epochs = CFG_DINO["warmup_epochs"]
        )

        _, _, dino_test_acc, dino_test_roc = evaluate(
            dino_model, dino_loaders["test"], nn.CrossEntropyLoss(), DEVICE)

        print(f"  → test_su_acc={dino_test_acc:.4f} | test_su_roc={dino_test_roc:.4f}")

        
        results.append({
            "recipe":        recipe_name,
            "ops":           str(ops_list),
            "swin_test_acc": swin_test_acc,
            "swin_test_roc": swin_test_roc,
            "dino_test_acc": dino_test_acc,
            "dino_test_roc": dino_test_roc,
        })

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL RESULTS — PET PREPROCESSING COMPARISON")
    print("=" * 80)
    print(f"{'Recipe':<45} {'Swin AUC':>10} {'DINO AUC':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r['recipe']:<45} {r['swin_test_roc']:>10.4f} {r['dino_test_roc']:>10.4f}")

    # Save CSV
    results_df  = pd.DataFrame(results)
    results_csv = os.path.join(OUT_DIR, "pet_preprocessing_comparison_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")

    # Best per architecture
    best_swin = max(results, key=lambda x: x["swin_test_roc"])
    best_dino = max(results, key=lambda x: x["dino_test_roc"])
    print(f"\nBest for Swin-V2 : {best_swin['recipe']}  (AUC={best_swin['swin_test_roc']:.4f})")
    print(f"Best for DINOv2  : {best_dino['recipe']}  (AUC={best_dino['dino_test_roc']:.4f})")

    # Compare to VGG-19 baseline (raw = 0.9779)
    raw_result = next(r for r in results if r["recipe"] == "A01_raw")
    print(f"\nBaseline (raw) performance:")
    print(f"  VGG-19  : 0.9779  (from PET preprocessing search)")
    print(f"  Swin-V2 : {raw_result['swin_test_roc']:.4f}")
    print(f"  DINOv2  : {raw_result['dino_test_roc']:.4f}")

    # Bar chart
    _plot_results(results, os.path.join(OUT_DIR, "pet_preprocessing_comparison.png"))

    return results_df


def _plot_results(results, save_path):
    """Bar chart comparing Swin and DINOv2 AUC across recipes."""
    recipes       = [r["recipe"] for r in results]
    swin_aucs     = [r["swin_test_roc"] for r in results]
    dino_aucs     = [r["dino_test_roc"] for r in results]
    vgg_baselines = [0.9779] * len(recipes)   # VGG-19 raw baseline

    x     = np.arange(len(recipes))
    width = 0.3

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width, swin_aucs,     width, label="Swin-V2-B",  color="#2196F3")
    ax.bar(x,          dino_aucs,    width, label="DINOv2-B/14", color="#FF9800")
    ax.plot(x, vgg_baselines, "r--", linewidth=1.2, label="VGG-19 raw (0.9779)")

    ax.set_xticks(x - width / 2)
    ax.set_xticklabels(recipes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Subject-level AUC (test)")
    ax.set_title("PET Preprocessing — Swin-V2 vs DINOv2  (5-epoch fast validation)")
    ax.legend()
    ax.set_ylim(0.7, 1.02)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()