import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import LambdaLR

# ── reproducibility ───────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")

# ==============================================================
# CONFIG  ── tuned for Swin Transformer V2-Base
# ==============================================================
BASE    = "/hdd3/seecs/abubakar.seecs/adni/datasets"
MRI_DIR = "/hdd3/seecs/abubakar.seecs/adni/datasets/MRI_Slices/"
PET_DIR = "/hdd3/seecs/abubakar.seecs/adni/datasets/PET_Slices/"
OUT_DIR = "/hdd3/seecs/abubakar.seecs/adni/output"

CFG = dict(
    lr            = 1e-4,    # ↑ from 5e-5  — Swin fine-tuning sweet spot
    batch_size    = 32,      # ↓ from 64    — Swin-B uses more VRAM
    epochs        = 15,      # ↑ from 8     — transformers need more warmup time
    patience      = 7,       # ↑ from 5     — avoid premature stop during warmup
    dropout       = 0.4,     # ↓ from 0.5   — Swin already has internal stochastic depth
    dense_units   = 256,     # ↑ from 128   — matches Swin-B's richer 1024-d feature space
    num_classes   = 2,
    img_size      = 256,     # ↑ from 224   — Swin V2-B native training resolution
    val_frac      = 0.15,
    test_frac     = 0.15,
    warmup_epochs = 3,       # NEW — linear LR warmup, critical for stable Swin fine-tuning
    weight_decay  = 0.05,    # NEW — AdamW weight decay, standard for ViT-family models
    # ── Swin backbone ──────────────────────────────────────────
    swin_feat_dim = 1024,    # Swin-V2-B final feature dimension (do not change)
)

LABEL_MAP = {"AD": 1, "CN": 0}
IDX2LABEL = {0: "CN", 1: "AD"}

# ==============================================================
# 1. SCAN FOLDERS  (unchanged)
# ==============================================================
def scan_modality(root_dir):
    """
    Scans root_dir/AD/<subject>/*.png  and  root_dir/CN/<subject>/*.png
    Returns DataFrame: [subject_id, group, slice_path]
    """
    records = []
    for group in ["AD", "CN"]:
        group_dir = os.path.join(root_dir, group)
        if not os.path.isdir(group_dir):
            print(f"  WARNING: {group_dir} not found")
            continue
        for subject in sorted(os.listdir(group_dir)):
            subj_dir = os.path.join(group_dir, subject)
            if not os.path.isdir(subj_dir):
                continue
            for fname in sorted(os.listdir(subj_dir)):
                if fname.lower().endswith(".png"):
                    records.append({
                        "subject_id": subject,
                        "group":      group,
                        "slice_path": os.path.join(subj_dir, fname),
                    })
    df = pd.DataFrame(records)
    print(f"  {root_dir.split('/')[-1]}: "
        f"{df['subject_id'].nunique()} subjects | "
        f"{len(df)} slices | "
        f"{df['group'].value_counts().to_dict()}")
    return df

# ==============================================================
# 2. SUBJECT PARTITIONING  (unchanged)
# ==============================================================
def partition_subjects_v2(mri_df, pet_df):
    mri_subjs = set(mri_df["subject_id"])
    pet_subjs = set(pet_df["subject_id"])

    overlap  = mri_subjs & pet_subjs
    mri_only = mri_subjs - overlap
    pet_only = pet_subjs - overlap

    print(f"\n=== Subject Partitioning ===")
    print(f"  MRI-only  : {len(mri_only)} subjects  → MRI backbone TRAINING")
    print(f"  PET-only  : {len(pet_only)} subjects  → PET backbone TRAINING")
    print(f"  Overlap   : {len(overlap)}  subjects  → Backbone TESTING + Fusion")

    return mri_only, pet_only, overlap


def subject_split_v2(subjects, group_map, val_frac=0.15, seed=SEED):
    split_map = {}
    by_group  = {}
    for s in subjects:
        g = group_map[s]
        by_group.setdefault(g, []).append(s)

    for group, subjs in by_group.items():
        subjs = sorted(subjs)
        random.Random(seed).shuffle(subjs)
        n     = len(subjs)
        n_val = max(1, int(n * val_frac))
        labels = ["val"] * n_val + ["train"] * (n - n_val)
        split_map.update(zip(subjs, labels))

    counts = pd.Series(split_map).value_counts()
    print(f"    train: {counts.get('train',0)} | val: {counts.get('val',0)} subjects")
    return split_map


def subject_split_fusion(subjects, group_map, val_frac=0.15, test_frac=0.15, seed=SEED):
    split_map = {}
    by_group  = {}
    for s in subjects:
        g = group_map[s]
        by_group.setdefault(g, []).append(s)

    for group, subjs in by_group.items():
        subjs = sorted(subjs)
        random.Random(seed).shuffle(subjs)
        n      = len(subjs)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        labels = (["test"]  * n_test +
                ["val"]   * n_val  +
                ["train"] * (n - n_test - n_val))
        split_map.update(zip(subjs, labels))

    counts = pd.Series(split_map).value_counts()
    print(f"    train: {counts.get('train',0)} | val: {counts.get('val',0)} | test: {counts.get('test',0)} subjects")
    return split_map


def apply_split(df, split_map):
    df = df.copy()
    df["split"] = df["subject_id"].map(split_map)
    df = df[df["split"].notna()].reset_index(drop=True)
    return df


def mark_as_test(df, test_subjects):
    df = df.copy()
    df["split"] = df["subject_id"].apply(lambda x: "test" if x in test_subjects else None)
    df = df[df["split"].notna()].reset_index(drop=True)
    return df

# ==============================================================
# 3. DATASETS & LOADERS
# ==============================================================
def get_transforms(img_size, split="train"):
    """
    Swin V2-B was pretrained at 256×256 with ImageNet-1K stats.
    Training augmentations kept conservative to match medical imaging norms.
    """
    
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),          # NEW — mild rotation for MRI/PET
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
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["slice_path"]).convert("RGB")
        label = LABEL_MAP[row["group"]]
        if self.transform:
            img = self.transform(img)
        return img, label, row["subject_id"]


class MultimodalDataset(Dataset):
    """
    Pairs MRI + PET slices for overlap subjects.
    Cartesian product per subject (all MRI slices × all PET slices).
    """
    def __init__(self, mri_df, pet_df, transform=None):
        self.transform = transform
        pairs  = []
        common = set(mri_df["subject_id"]) & set(pet_df["subject_id"])
        for subj in sorted(common):
            mri_slices = mri_df[mri_df["subject_id"] == subj]["slice_path"].tolist()
            pet_slices = pet_df[pet_df["subject_id"] == subj]["slice_path"].tolist()
            group      = mri_df[mri_df["subject_id"] == subj]["group"].iloc[0]
            for m in mri_slices:
                for p in pet_slices:
                    pairs.append((m, p, group, subj))
        self.pairs = pairs
        print(f"    {len(common)} subjects | {len(pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mri_path, pet_path, group, subj = self.pairs[idx]
        label   = LABEL_MAP[group]
        mri_img = Image.open(mri_path).convert("RGB")
        pet_img = Image.open(pet_path).convert("RGB")
        if self.transform:
            mri_img = self.transform(mri_img)
            pet_img = self.transform(pet_img)
        return mri_img, pet_img, label, subj


def make_single_loaders_backbone(df, batch_size, img_size, name):
    loaders = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        ds  = SliceDataset(sub, transform=get_transforms(img_size, split if split != "test" else "val"))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2, pin_memory=True,
        )
        print(f"    [{name}][{split}] "
            f"{sub['subject_id'].nunique()} subjects | "
            f"{len(sub)} slices")
    return loaders


def make_multimodal_loaders(mri_df, pet_df, batch_size, img_size):
    loaders = {}
    for split in ["train", "val", "test"]:
        m  = mri_df[mri_df["split"] == split]
        p  = pet_df[pet_df["split"] == split]
        print(f"  [{split}] ", end="")
        ds = MultimodalDataset(m, p, transform=get_transforms(img_size, split if split != "test" else "val"))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2, pin_memory=True,
        )
    return loaders

# ==============================================================
# 4. MODEL ARCHITECTURE  ── Swin Transformer V2-Base
# ==============================================================
class SwinExtractor(nn.Module):
    """
    Swin Transformer V2-Base backbone with a 2-layer classification head.

    Architecture:
        SwinV2-B (ImageNet-1K, 256×256)
            ↓  [B, 1024]  CLS-equivalent from adaptive avg pool over patch tokens
        Dropout(dropout)
        Linear(1024 → dense_units)   ← projection head
        ReLU
        [forward_features stops here — returns (B, dense_units)]
        Linear(dense_units → num_classes)   ← classification head

    The split between forward_features / forward is kept identical to the
    original VGG19Extractor so the MultimodalFusionModel works unchanged.
    """
    def __init__(self,
                feat_dim   = 1024,   # Swin-V2-B output channels
                dense_units = 256,
                dropout     = 0.4,
                num_classes = 2):
        super().__init__()

        # ── Load pretrained Swin V2-Base (256×256 variant) ──────
        base = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)

        # Swin's torchvision layout:
        #   base.features   → patch embed + 4 stages of Swin blocks
        #   base.norm       → final LayerNorm
        #   base.permute    → [B, C, H, W] → [B, H, W, C] fix
        #   base.avgpool    → AdaptiveAvgPool2d(1)
        #   base.head       → Linear(1024, 1000)  ← we replace this

        self.features = base.features    # Swin patch embed + transformer stages
        self.norm     = base.norm        # final LayerNorm
        self.permute  = base.permute     # channel-last → channel-first permute
        self.avgpool  = base.avgpool     # AdaptiveAvgPool2d(1)

        self.dropout  = nn.Dropout(dropout)
        self.fc1      = nn.Linear(feat_dim, dense_units)
        self.fc2      = nn.Linear(dense_units, num_classes)

    def forward_features(self, x):
        """Returns (B, dense_units) embedding — used by fusion model."""
        x = self.features(x)   # [B, H', W', C=1024]  (channel-last from Swin)
        x = self.norm(x)       # LayerNorm over last dim
        x = self.permute(x)    # [B, C, H', W']
        x = self.avgpool(x)    # [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, 1024]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))  # [B, dense_units]
        return x

    def forward(self, x):
        return self.fc2(self.forward_features(x))


class MultimodalFusionModel(nn.Module):
    """
    Concatenates MRI and PET Swin embeddings and passes through a fusion MLP.
    Identical interface to the original — only the stream modules change.
    """
    def __init__(self, mri_ext, pet_ext,
                dense_units=256, dropout=0.4, num_classes=2):
        super().__init__()
        self.mri_stream = mri_ext
        self.pet_stream = pet_ext
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dense_units * 2, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units),
            nn.ReLU(inplace=True),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, mri, pet):
        f = torch.cat([
            self.mri_stream.forward_features(mri),
            self.pet_stream.forward_features(pet),
        ], dim=1)
        return self.fusion(f)

# ==============================================================
# 5. LR SCHEDULER  ── linear warmup then cosine decay
# ==============================================================
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to ~0.
    This stabilises Swin's attention layers early in training.
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        # cosine decay phase
        progress = (current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

# ==============================================================
# 6. TRAINING
# ==============================================================
def train_one_epoch(model, loader, optimizer, criterion, device, multimodal=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if multimodal:
            mri, pet, labels, _ = batch
            mri, pet, labels = mri.to(device), pet.to(device), labels.to(device)
            logits = model(mri, pet)
        else:
            imgs, labels, _ = batch
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, multimodal=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs, all_subjects = [], [], [], []
    for batch in loader:
        if multimodal:
            mri, pet, labels, subjects = batch
            mri, pet, labels = mri.to(device), pet.to(device), labels.to(device)
            logits = model(mri, pet)
        else:
            imgs, labels, subjects = batch
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
        loss  = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)[:, 1]
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_subjects.extend(subjects)
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels),
            np.array(all_probs), all_subjects)


def subject_agg(preds, labels, probs, subjects):
    df = pd.DataFrame({"subject": subjects, "label": labels,
                        "pred": preds, "prob": probs})
    rows = []
    for subj, g in df.groupby("subject"):
        mean_prob  = g["prob"].mean()
        true_label = g["label"].iloc[0]
        rows.append({"subject": subj, "true_label": true_label,
                    "pred_label": int(mean_prob >= 0.5), "mean_prob": mean_prob})
    res = pd.DataFrame(rows)
    acc = (res["true_label"] == res["pred_label"]).mean()
    return (res["pred_label"].values, res["true_label"].values,
            res["mean_prob"].values, acc, res)


def print_results(name, sl_preds, sl_labels, sl_probs, subjects):
    # Slice-level
    fpr, tpr, _ = roc_curve(sl_labels, sl_probs)
    sl_roc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(sl_labels, sl_probs)
    sl_pr  = auc(rec, prec)
    sl_acc = (sl_preds == sl_labels).mean()

    print(f"\n  ── SLICE-LEVEL ({len(sl_labels)} slices) ──")
    print(f"  Accuracy : {sl_acc:.4f}  |  ROC-AUC : {sl_roc:.4f}  |  PR-AUC : {sl_pr:.4f}")
    print(classification_report(sl_labels, sl_preds, target_names=["CN","AD"], digits=4))

    # Subject-level
    su_preds, su_labels, su_probs, su_acc, su_df = subject_agg(
        sl_preds, sl_labels, sl_probs, subjects)
    fpr, tpr, _ = roc_curve(su_labels, su_probs)
    su_roc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(su_labels, su_probs)
    su_pr  = auc(rec, prec)

    print(f"  ── SUBJECT-LEVEL ({len(su_labels)} subjects) ──")
    print(f"  Accuracy : {su_acc:.4f}  |  ROC-AUC : {su_roc:.4f}  |  PR-AUC : {su_pr:.4f}")
    print(classification_report(su_labels, su_preds, target_names=["CN","AD"], digits=4))

    su_df["true_name"] = su_df["true_label"].map(IDX2LABEL)
    su_df["pred_name"] = su_df["pred_label"].map(IDX2LABEL)
    csv_path = os.path.join(OUT_DIR, f"{name}_subject_predictions.csv")
    su_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return sl_acc, su_acc, sl_roc, su_roc, sl_pr, su_pr, su_preds, su_labels, su_probs


def fit(model, loaders, save_path, name,
        lr, epochs, patience, device,
        weight_decay=0.05, warmup_epochs=3,
        multimodal=False):
    """
    Training loop with:
    - AdamW (weight_decay) instead of Adam  — better for ViT-family
    - Linear warmup + cosine LR decay       — stabilises Swin attention
    - Early stopping on val accuracy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)

    best_val_acc, no_improve = 0.0, 0
    history = {"train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, multimodal)
        va_loss, va_acc, _, _, _, _ = evaluate(
            model, loaders["val"], criterion, device, multimodal)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        saved = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve   = 0
            torch.save(model.state_dict(), save_path)
            saved = " ✅"
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{name}] Early stop @ epoch {epoch}")
                break

        print(f"  [{name}] Ep {epoch:02d}/{epochs} | "
            f"Train {tr_acc:.4f} ({tr_loss:.4f}) | "
            f"Val {va_acc:.4f} ({va_loss:.4f}) | "
            f"LR {current_lr:.2e}{saved}")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history

# ==============================================================
# 7. PLOTS  (unchanged)
# ==============================================================
def plot_curves(history, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    e = range(1, len(history["train_acc"]) + 1)
    ax1.plot(e, history["train_acc"], label="Train")
    ax1.plot(e, history["val_acc"],   label="Val")
    ax1.set_title(f"{name} — Accuracy"); ax1.legend()
    ax2.plot(e, history["train_loss"], label="Train")
    ax2.plot(e, history["val_loss"],   label="Val")
    ax2.set_title(f"{name} — Loss"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_curves.png"), dpi=150)
    plt.show()


def plot_roc_pr(labels, probs, name):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc      = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(labels, probs)
    pr_auc        = auc(rec, prec)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    ax1.plot([0,1],[0,1],"k--"); ax1.set_title(f"ROC — {name}"); ax1.legend()
    ax2.plot(rec, prec, lw=2, label=f"AUC={pr_auc:.3f}")
    ax2.set_title(f"PR — {name}"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_roc_pr.png"), dpi=150)
    plt.show()

# ==============================================================
# 8. MAIN
# ==============================================================
def main():

    # ── Scan both modalities (only for info/logging) ──────────
    print("\n=== Scanning slices ===")
    mri_df = scan_modality(MRI_DIR)
    pet_df = scan_modality(PET_DIR)

    # ── Load PRECOMPUTED splits ───────────────────────────────
    print("\n=== Loading precomputed splits ===")

    mri_backbone = pd.read_csv(os.path.join(OUT_DIR, "mri_backbone_splits.csv"))
    pet_backbone = pd.read_csv(os.path.join(OUT_DIR, "pet_backbone_splits.csv"))
    mri_fusion   = pd.read_csv(os.path.join(OUT_DIR, "mri_fusion_splits.csv"))
    pet_fusion   = pd.read_csv(os.path.join(OUT_DIR, "pet_fusion_splits.csv"))

    

    # ── Optional: sanity check (VERY IMPORTANT) ───────────────
    def check_leakage(mri_backbone, mri_fusion):
        train_subjs = set(mri_backbone[mri_backbone["split"]=="train"]["subject_id"])
        val_subjs   = set(mri_fusion[mri_fusion["split"]=="val"]["subject_id"])
        test_subjs  = set(mri_fusion[mri_fusion["split"]=="test"]["subject_id"])

        print("\n=== Leakage Check ===")
        print("Train ∩ Val :", len(train_subjs & val_subjs))
        print("Train ∩ Test:", len(train_subjs & test_subjs))

    check_leakage(mri_backbone, mri_fusion)

    # ── Get overlap subjects (for logging only) ───────────────
    overlap = set(mri_fusion["subject_id"].unique())

    # =========================================================
    # STAGE 1 — MRI
    # =========================================================
    print("\n" + "="*60)
    print("STAGE 1 — MRI Swin Transformer V2-Base")
    print("="*60)

    mri_loaders = make_single_loaders_backbone(
        mri_backbone, CFG["batch_size"], CFG["img_size"], "MRI")

    mri_model = SwinExtractor(
        feat_dim    = CFG["swin_feat_dim"],
        dense_units = CFG["dense_units"],
        dropout     = CFG["dropout"],
        num_classes = CFG["num_classes"],
    ).to(DEVICE)

    mri_model, mri_hist = fit(
        mri_model, mri_loaders,
        save_path     = os.path.join(OUT_DIR, "mri_swin_best.pth"),
        name          = "MRI",
        lr            = CFG["lr"],
        epochs        = CFG["epochs"],
        patience      = CFG["patience"],
        device        = DEVICE,
        weight_decay  = CFG["weight_decay"],
        warmup_epochs = CFG["warmup_epochs"],
    )
    plot_curves(mri_hist, "MRI")

    _, _, mri_preds, mri_lbls, mri_probs, mri_subjects = evaluate(
        mri_model, mri_loaders["test"], nn.CrossEntropyLoss(), DEVICE)

    mri_acc, mri_su_acc, mri_sl_roc, mri_su_roc, mri_sl_pr, mri_su_pr, \
        mri_su_preds, mri_su_lbls, mri_su_probs = print_results(
            "MRI_overlap_test", mri_preds, mri_lbls, mri_probs, mri_subjects)

    plot_roc_pr(mri_lbls, mri_probs, "MRI_slice")
    plot_roc_pr(mri_su_lbls, mri_su_probs, "MRI_subject")

    # =========================================================
    # STAGE 2 — PET
    # =========================================================
    print("\n" + "="*60)
    print("STAGE 2 — PET Swin Transformer V2-Base")
    print("="*60)

    pet_loaders = make_single_loaders_backbone(
        pet_backbone, CFG["batch_size"], CFG["img_size"], "PET")

    pet_model = SwinExtractor(
        feat_dim    = CFG["swin_feat_dim"],
        dense_units = CFG["dense_units"],
        dropout     = CFG["dropout"],
        num_classes = CFG["num_classes"],
    ).to(DEVICE)

    pet_model, pet_hist = fit(
        pet_model, pet_loaders,
        save_path     = os.path.join(OUT_DIR, "pet_swin_best.pth"),
        name          = "PET",
        lr            = CFG["lr"],
        epochs        = CFG["epochs"],
        patience      = CFG["patience"],
        device        = DEVICE,
        weight_decay  = CFG["weight_decay"],
        warmup_epochs = CFG["warmup_epochs"],
    )
    plot_curves(pet_hist, "PET")

    _, _, pet_preds, pet_lbls, pet_probs, pet_subjects = evaluate(
        pet_model, pet_loaders["test"], nn.CrossEntropyLoss(), DEVICE)

    pet_acc, pet_su_acc, pet_sl_roc, pet_su_roc, pet_sl_pr, pet_su_pr, \
        pet_su_preds, pet_su_lbls, pet_su_probs = print_results(
            "PET_overlap_test", pet_preds, pet_lbls, pet_probs, pet_subjects)

    plot_roc_pr(pet_lbls, pet_probs, "PET_slice")
    plot_roc_pr(pet_su_lbls, pet_su_probs, "PET_subject")

    # =========================================================
    # STAGE 3 — MULTIMODAL
    # =========================================================
    print("\n" + "="*60)
    print("STAGE 3 — Multimodal Fusion")
    print(f"Using {len(overlap)} overlap subjects")
    print("="*60)

    # Reload best backbone weights
    mri_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "mri_swin_best.pth"), map_location=DEVICE))
    pet_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "pet_swin_best.pth"), map_location=DEVICE))

    # Freeze backbone
    for p in mri_model.features.parameters(): p.requires_grad = False
    for p in mri_model.norm.parameters():     p.requires_grad = False
    for p in pet_model.features.parameters(): p.requires_grad = False
    for p in pet_model.norm.parameters():     p.requires_grad = False

    mm_model = MultimodalFusionModel(
        mri_model, pet_model,
        dense_units = CFG["dense_units"],
        dropout     = CFG["dropout"],
        num_classes = CFG["num_classes"],
    ).to(DEVICE)

    print("\nMultimodal loaders:")
    mm_loaders = make_multimodal_loaders(
        mri_fusion, pet_fusion, CFG["batch_size"], CFG["img_size"])

    mm_model, mm_hist = fit(
        mm_model, mm_loaders,
        save_path     = os.path.join(OUT_DIR, "multimodal_best.pth"),
        name          = "Multimodal",
        lr            = CFG["lr"],
        epochs        = CFG["epochs"],
        patience      = CFG["patience"],
        device        = DEVICE,
        weight_decay  = CFG["weight_decay"],
        warmup_epochs = CFG["warmup_epochs"],
        multimodal    = True,
    )
    plot_curves(mm_hist, "Multimodal")

    _, _, mm_preds, mm_lbls, mm_probs, mm_subjects = evaluate(
        mm_model, mm_loaders["test"], nn.CrossEntropyLoss(),
        DEVICE, multimodal=True)

    mm_acc, mm_su_acc, mm_sl_roc, mm_su_roc, mm_sl_pr, mm_su_pr, \
        mm_su_preds, mm_su_lbls, mm_su_probs = print_results(
            "Multimodal", mm_preds, mm_lbls, mm_probs, mm_subjects)

    plot_roc_pr(mm_lbls, mm_probs, "Multimodal_slice")
    plot_roc_pr(mm_su_lbls, mm_su_probs, "Multimodal_subject")

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print(f"{'Model':<25} {'Slice Acc':>10} {'Subj Acc':>10} {'Subj ROC':>10}")
    print(f"{'-'*65}")
    print(f"{'MRI Swin-V2-B':<25} {mri_acc:>10.4f} {mri_su_acc:>10.4f} {mri_su_roc:>10.4f}")
    print(f"{'PET Swin-V2-B':<25} {pet_acc:>10.4f} {pet_su_acc:>10.4f} {pet_su_roc:>10.4f}")
    print(f"{'Multimodal Fusion':<25} {mm_acc:>10.4f} {mm_su_acc:>10.4f} {mm_su_roc:>10.4f}")

    torch.save(mm_model.state_dict(),
               os.path.join(OUT_DIR, "multimodal_final.pth"))

if __name__ == "__main__":
    main()
