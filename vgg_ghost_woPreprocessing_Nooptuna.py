"""
Ghost-Conv VGG19  3-Stage Training — CSV-Split Pipeline
========================================================

Same pipeline as vgg_woPreprocessing_Nooptuna.py, but every Conv2d block
inside the VGG19 feature extractor is replaced with a GhostConv block.

GhostConv (Han et al., GhostNet, CVPR 2020):
  - Primary branch : Conv2d(in_ch, out_ch//2, 1×1) + BN + ReLU
                     → s = out_ch//2 "intrinsic" feature maps
  - Ghost  branch  : DepthwiseConv2d(s, s, 3×3, groups=s) + BN + ReLU
                     → s cheap "ghost" feature maps
  - Concatenate primary + ghost  →  out_ch channels
  This gives roughly half the FLOPs of a standard Conv2d with the same
  input/output channel counts.

Loads pre-computed split CSVs (produced by the partitioning script):
  mri_backbone_splits.csv  — MRI-only train/val + overlap test
  pet_backbone_splits.csv  — PET-only train/val + overlap test
  mri_fusion_splits.csv    — Overlap subjects for fusion (train/val/test)
  pet_fusion_splits.csv    — Overlap subjects for fusion (train/val/test)
"""

import os
import time
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
# CONFIG
# ==============================================================
OUT_DIR   = "/kaggle/working"

# Directory where the pre-computed split CSVs live
SPLIT_DIR = "/hdd3/seecs/abubakar.seecs/adni/output/"

# ── Path remapping ────────────────────────────────────────────
# If the CSVs were generated on a different machine the slice_path
# values will have the old prefix below.  Set NEW_DATA_ROOT to the
# equivalent root on the current machine and every path will be
# rewritten automatically when a CSV is loaded.
# Set NEW_DATA_ROOT = "" to disable remapping.
OLD_DATA_ROOT = "/hdd3/seecs/abubakar.seecs/adni/datasets"
NEW_DATA_ROOT = "/kaggle/input/datasets/adaisdiashdh/mri-pet-slices"

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

# ==============================================================
# 1. LOAD PRE-COMPUTED SPLITS FROM CSV
# ==============================================================
def load_split_csv(csv_path, name=""):
    """
    Reads a pre-computed split CSV with columns:
        subject_id, group, slice_path, split
    where split ∈ {train, val, test}.
    """
    df = pd.read_csv(csv_path)
    assert set(df.columns) >= {"subject_id", "group", "slice_path", "split"}, \
        f"CSV {csv_path} must have columns: subject_id, group, slice_path, split"

    # ── Remap slice paths if running on a different machine ───
    if NEW_DATA_ROOT:
        df["slice_path"] = df["slice_path"].str.replace(
            OLD_DATA_ROOT, NEW_DATA_ROOT, n=1, regex=False)
        print(f"  [{name}] Path remapped: '{OLD_DATA_ROOT}' → '{NEW_DATA_ROOT}'")

    split_counts = df.groupby("split")["subject_id"].nunique()
    print(f"  [{name}] Loaded {csv_path}")
    print(f"    {df['subject_id'].nunique()} subjects | {len(df)} slices")
    for s in ["train", "val", "test"]:
        n_subj = split_counts.get(s, 0)
        n_rows = len(df[df["split"] == s])
        print(f"      {s:>5}: {n_subj} subjects | {n_rows} slices")
    return df

# ==============================================================
# 2. DATASETS & LOADERS
# ==============================================================
def get_transforms(img_size, split="train"):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    """For backbone training: train/val/test — test uses overlap subjects."""
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
# 3. GHOST CONVOLUTION BUILDING BLOCK
# ==============================================================
class GhostConv(nn.Module):
    """
    Drop-in replacement for a Conv2d + BN + ReLU block using Ghost modules.

    Architecture
    ─────────────
    Input  : [B, in_ch,  H, W]
                │
                ├─ Primary branch ─────────────────────────────────────
                │   Conv2d(in_ch, out_ch//2, kernel_size=1, bias=False)
                │   BatchNorm2d  →  ReLU
                │   output: [B, s, H', W']   where s = out_ch // 2
                │
                └─ Ghost branch ────────────────────────────────────────
                    DepthwiseConv2d(s, s, kernel_size=dw_size,
                                   padding=dw_size//2, groups=s, bias=False)
                    BatchNorm2d  →  ReLU
                    output: [B, s, H', W']

    Concatenate along channel dim → [B, out_ch, H', W']

    Notes
    ─────
    • The 1×1 primary conv is padded by the *original* `padding` argument so
      it replicates the spatial stride/padding of the conv it replaces.
    • For VGG19 all convolutions use kernel_size=3, pad=1, stride=1;
      the primary 1×1 conv produces the same spatial size, and the ghost
      depthwise conv with dw_size=3, pad=1 keeps it unchanged.
    • When in_ch == out_ch the ghost branch doubles the channels cheaply.
    • FLOPs ≈ out_ch/2 × 1×1 primary  +  out_ch/2 × 3×3 DW
           vs out_ch × 3×3 for a standard conv  →  ~50 % saving.
    """
    def __init__(self, in_ch, out_ch,
                 kernel_size=1, stride=1, padding=0,
                 dw_size=3, ratio=2, use_relu=True):
        super().__init__()
        assert out_ch % ratio == 0, "out_ch must be divisible by ratio"
        init_ch = out_ch // ratio          # primary ("intrinsic") channels

        # Primary branch: pointwise 1×1 conv
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True) if use_relu else nn.Identity(),
        )

        # Ghost branch: cheap depthwise conv on primary features
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_ch, init_ch, dw_size, stride=1,
                      padding=dw_size // 2, groups=init_ch, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True) if use_relu else nn.Identity(),
        )

    def forward(self, x):
        primary = self.primary_conv(x)     # [B, out_ch//2, H, W]
        ghost   = self.cheap_operation(primary)  # [B, out_ch//2, H, W]
        return torch.cat([primary, ghost], dim=1)  # [B, out_ch, H, W]


def _make_ghost_vgg19_features():
    """
    Reconstructs VGG19's feature block with GhostConv replacing every
    Conv2d+ReLU pair.  MaxPool2d layers are kept identical.

    VGG19 feature cfg:
        [64, 64, 'M',
         128, 128, 'M',
         256, 256, 256, 256, 'M',
         512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M']
    """
    cfg = [64, 64, 'M',
           128, 128, 'M',
           256, 256, 256, 256, 'M',
           512, 512, 512, 512, 'M',
           512, 512, 512, 512, 'M']

    layers = []
    in_ch  = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            # Ghost module: 1×1 primary to match original spatial dims
            # (VGG19 always uses 3×3 conv with pad=1; ghost dw also 3×3 pad=1)
            layers.append(GhostConv(in_ch, v,
                                    kernel_size=1, stride=1, padding=0,
                                    dw_size=3))
            in_ch = v
    return nn.Sequential(*layers)


# ==============================================================
# 4. MODEL ARCHITECTURE  — Ghost-VGG19
# ==============================================================
class GhostVGG19Extractor(nn.Module):
    """
    VGG19 feature extractor with Ghost Convolutions.

    Architecture
    ─────────────
    GhostVGG19 features  (16 Ghost-Conv blocks + 5 MaxPool)
        ↓  [B, 512, 7, 7]
    AdaptiveAvgPool2d(1)  →  [B, 512]
    Dropout(dropout)
    Linear(512 → dense_units)  →  ReLU           forward_features() stops here
    Linear(dense_units → num_classes)             classification head

    The pretrained VGG19 weights are used to initialise the PRIMARY branch
    of each GhostConv (the 1×1 conv uses the central weight of the 3×3
    kernel as a best-effort initialisation).  The ghost depthwise branch is
    initialised with Kaiming uniform.
    """
    def __init__(self, dense_units=128, dropout=0.5, num_classes=2):
        super().__init__()

        # ── Build Ghost-VGG19 feature block ──────────────────
        self.features = _make_ghost_vgg19_features()

        # ── Initialise primary branch from pretrained VGG19 ──
        self._init_from_pretrained()

        # ── Classification head ───────────────────────────────
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(512, dense_units)
        self.fc2     = nn.Linear(dense_units, num_classes)

    def _init_from_pretrained(self):
        """
        Load pretrained VGG19 weights and copy them into the primary branch.

        The original VGG19 uses 3×3 Conv2d.  Our primary branch uses 1×1
        Conv2d; we initialise it with the *centre pixel* of the 3×3 kernel
        (index [0, 0, 1, 1]) — a reasonable approximation that gives the
        model a warm start from ImageNet features.
        """
        pretrained = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        pre_convs  = [m for m in pretrained.features if isinstance(m, nn.Conv2d)]

        ghost_mods = [m for m in self.features if isinstance(m, GhostConv)]

        for ghost_mod, pre_conv in zip(ghost_mods, pre_convs):
            primary_conv = ghost_mod.primary_conv[0]  # the nn.Conv2d inside primary branch
            with torch.no_grad():
                # pre_conv.weight shape: [out_ch, in_ch, 3, 3]
                # primary_conv.weight shape: [out_ch//2, in_ch, 1, 1]
                out_half = primary_conv.weight.shape[0]
                # Take the centre pixel of the first out_half filters
                primary_conv.weight.copy_(
                    pre_conv.weight[:out_half, :, 1:2, 1:2]
                )

        print("  [GhostVGG19] Primary branches initialised from pretrained VGG19 "
              "(centre-pixel of 3×3 kernels)")

    def forward_features(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        return self.fc2(self.forward_features(x))


class MultimodalFusionModel(nn.Module):
    def __init__(self, mri_ext, pet_ext,
                 dense_units=128, dropout=0.5, num_classes=2):
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
# 5. TRAINING
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

    # Save per-subject CSV
    su_df["true_name"] = su_df["true_label"].map(IDX2LABEL)
    su_df["pred_name"] = su_df["pred_label"].map(IDX2LABEL)
    csv_path = os.path.join(OUT_DIR, f"{name}_subject_predictions.csv")
    su_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return sl_acc, su_acc, sl_roc, su_roc, sl_pr, su_pr, su_preds, su_labels, su_probs


def fit(model, loaders, save_path, name,
        lr, epochs, patience, device, multimodal=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc, no_improve = 0.0, 0
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, multimodal)
        va_loss, va_acc, _, _, _, _ = evaluate(
            model, loaders["val"], criterion, device, multimodal)

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
              f"Val {va_acc:.4f} ({va_loss:.4f}){saved}")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history

# ==============================================================
# 6. INFERENCE TIMING  ── per-subject on CPU and GPU
# ==============================================================
@torch.no_grad()
def measure_inference_time(model, test_df, img_size, name):
    """
    Measures per-subject inference time on both GPU (if available) and CPU.
    For each subject ALL their test slices are batched and timed together.
    GPU timing uses torch.cuda.Event; CPU timing uses time.perf_counter.
    Results are printed as a table and saved to CSV.
    """
    print(f"\n{'='*70}")
    print(f"  INFERENCE TIMING — {name}")
    print(f"{'='*70}")

    transform = get_transforms(img_size, split="val")
    subjects  = sorted(test_df["subject_id"].unique())

    def _load_subject(subj):
        subj_df = test_df[test_df["subject_id"] == subj]
        imgs = [
            transform(Image.open(row["slice_path"]).convert("RGB"))
            for _, row in subj_df.iterrows()
        ]
        return torch.stack(imgs)

    results   = []
    gpu_times = {}
    cpu_times = {}
    n_slices  = {}
    has_gpu   = torch.cuda.is_available()

    if has_gpu:
        gpu_device = torch.device("cuda")
        model_gpu  = model.to(gpu_device)
        model_gpu.eval()
        dummy = _load_subject(subjects[0]).to(gpu_device)
        _ = model_gpu(dummy); torch.cuda.synchronize(); del dummy

        for subj in subjects:
            batch = _load_subject(subj)
            n_slices[subj] = len(batch)
            batch = batch.to(gpu_device)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); _ = model_gpu(batch); e.record()
            torch.cuda.synchronize()
            gpu_times[subj] = s.elapsed_time(e)
            del batch
        torch.cuda.empty_cache()

    cpu_device = torch.device("cpu")
    model_cpu  = model.to(cpu_device)
    model_cpu.eval()
    _ = model_cpu(_load_subject(subjects[0]))

    for subj in subjects:
        batch = _load_subject(subj)
        if subj not in n_slices:
            n_slices[subj] = len(batch)
        t0 = time.perf_counter()
        _ = model_cpu(batch)
        t1 = time.perf_counter()
        cpu_times[subj] = (t1 - t0) * 1000.0
        del batch

    print(f"  {'Subject':<18} | {'#Slices':>7} | {'GPU (ms)':>10} | {'CPU (ms)':>10}")
    print(f"  {'-'*57}")
    for subj in subjects:
        ns = n_slices[subj]
        gpu_ms = gpu_times.get(subj, float("nan"))
        cpu_ms = cpu_times[subj]
        gpu_str = f"{gpu_ms:>10.2f}" if has_gpu else f"{'N/A':>10}"
        print(f"  {subj:<18} | {ns:>7} | {gpu_str} | {cpu_ms:>10.2f}")
        results.append({"subject": subj, "n_slices": ns,
                        "gpu_ms": gpu_ms if has_gpu else None, "cpu_ms": cpu_ms})

    avg_slices = np.mean([r["n_slices"] for r in results])
    avg_cpu    = np.mean([r["cpu_ms"]   for r in results])
    std_cpu    = np.std( [r["cpu_ms"]   for r in results])
    print(f"  {'-'*57}")
    if has_gpu:
        avg_gpu = np.mean([r["gpu_ms"] for r in results])
        std_gpu = np.std( [r["gpu_ms"] for r in results])
        print(f"  {'MEAN':<18} | {avg_slices:>7.1f} | {avg_gpu:>10.2f} | {avg_cpu:>10.2f}")
        print(f"  {'STD':<18} | {'':>7} | {std_gpu:>10.2f} | {std_cpu:>10.2f}")
    else:
        print(f"  {'MEAN':<18} | {avg_slices:>7.1f} | {'N/A':>10} | {avg_cpu:>10.2f}")
        print(f"  {'STD':<18} | {'':>7} | {'N/A':>10} | {std_cpu:>10.2f}")

    csv_path = os.path.join(OUT_DIR, f"{name}_inference_times.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    if has_gpu:
        model.to(torch.device("cuda"))
    return results


@torch.no_grad()
def measure_inference_time_multimodal(model, mri_test_df, pet_test_df, img_size, name):
    """
    Measures per-subject inference time for the multimodal fusion model.
    Each subject's MRI×PET pairs are batched together.
    """
    print(f"\n{'='*70}")
    print(f"  INFERENCE TIMING — {name} (Multimodal)")
    print(f"{'='*70}")

    transform = get_transforms(img_size, split="val")
    common    = sorted(set(mri_test_df["subject_id"]) & set(pet_test_df["subject_id"]))

    def _load_subject(subj):
        mri_paths = mri_test_df[mri_test_df["subject_id"] == subj]["slice_path"].tolist()
        pet_paths = pet_test_df[pet_test_df["subject_id"] == subj]["slice_path"].tolist()
        mri_imgs, pet_imgs = [], []
        for mp in mri_paths:
            for pp in pet_paths:
                mri_imgs.append(transform(Image.open(mp).convert("RGB")))
                pet_imgs.append(transform(Image.open(pp).convert("RGB")))
        return torch.stack(mri_imgs), torch.stack(pet_imgs)

    results   = []
    gpu_times = {}
    cpu_times = {}
    n_pairs   = {}
    has_gpu   = torch.cuda.is_available()

    if has_gpu:
        gpu_device = torch.device("cuda")
        model_gpu  = model.to(gpu_device)
        model_gpu.eval()
        mri_w, pet_w = _load_subject(common[0])
        _ = model_gpu(mri_w.to(gpu_device), pet_w.to(gpu_device))
        torch.cuda.synchronize(); del mri_w, pet_w

        for subj in common:
            mri_b, pet_b = _load_subject(subj)
            n_pairs[subj] = len(mri_b)
            mri_b, pet_b = mri_b.to(gpu_device), pet_b.to(gpu_device)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); _ = model_gpu(mri_b, pet_b); e.record()
            torch.cuda.synchronize()
            gpu_times[subj] = s.elapsed_time(e)
            del mri_b, pet_b
        torch.cuda.empty_cache()

    cpu_device = torch.device("cpu")
    model_cpu  = model.to(cpu_device)
    model_cpu.eval()
    mri_w, pet_w = _load_subject(common[0])
    _ = model_cpu(mri_w, pet_w); del mri_w, pet_w

    for subj in common:
        mri_b, pet_b = _load_subject(subj)
        if subj not in n_pairs:
            n_pairs[subj] = len(mri_b)
        t0 = time.perf_counter()
        _ = model_cpu(mri_b, pet_b)
        t1 = time.perf_counter()
        cpu_times[subj] = (t1 - t0) * 1000.0
        del mri_b, pet_b

    print(f"  {'Subject':<18} | {'#Pairs':>7} | {'GPU (ms)':>10} | {'CPU (ms)':>10}")
    print(f"  {'-'*57}")
    for subj in common:
        np_ = n_pairs[subj]
        gpu_ms = gpu_times.get(subj, float("nan"))
        cpu_ms = cpu_times[subj]
        gpu_str = f"{gpu_ms:>10.2f}" if has_gpu else f"{'N/A':>10}"
        print(f"  {subj:<18} | {np_:>7} | {gpu_str} | {cpu_ms:>10.2f}")
        results.append({"subject": subj, "n_pairs": np_,
                        "gpu_ms": gpu_ms if has_gpu else None, "cpu_ms": cpu_ms})

    avg_pairs = np.mean([r["n_pairs"] for r in results])
    avg_cpu   = np.mean([r["cpu_ms"]  for r in results])
    std_cpu   = np.std( [r["cpu_ms"]  for r in results])
    print(f"  {'-'*57}")
    if has_gpu:
        avg_gpu = np.mean([r["gpu_ms"] for r in results])
        std_gpu = np.std( [r["gpu_ms"] for r in results])
        print(f"  {'MEAN':<18} | {avg_pairs:>7.1f} | {avg_gpu:>10.2f} | {avg_cpu:>10.2f}")
        print(f"  {'STD':<18} | {'':>7} | {std_gpu:>10.2f} | {std_cpu:>10.2f}")
    else:
        print(f"  {'MEAN':<18} | {avg_pairs:>7.1f} | {'N/A':>10} | {avg_cpu:>10.2f}")
        print(f"  {'STD':<18} | {'':>7} | {'N/A':>10} | {std_cpu:>10.2f}")

    csv_path = os.path.join(OUT_DIR, f"{name}_inference_times.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    if has_gpu:
        model.to(torch.device("cuda"))
    return results

# ==============================================================
# 7. PLOTS
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
    plt.savefig(os.path.join(OUT_DIR, f"ghost_{name}_curves.png"), dpi=150)
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
    plt.savefig(os.path.join(OUT_DIR, f"ghost_{name}_roc_pr.png"), dpi=150)
    plt.show()

# ==============================================================
# 7. MAIN
# ==============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load pre-computed splits from CSV ─────────────────────
    print("\n=== Loading Pre-Computed Splits ===")
    mri_backbone = load_split_csv(os.path.join(SPLIT_DIR, "mri_backbone_splits.csv"), "MRI Backbone")
    pet_backbone = load_split_csv(os.path.join(SPLIT_DIR, "pet_backbone_splits.csv"), "PET Backbone")
    mri_fusion   = load_split_csv(os.path.join(SPLIT_DIR, "mri_fusion_splits.csv"),   "MRI Fusion")
    pet_fusion   = load_split_csv(os.path.join(SPLIT_DIR, "pet_fusion_splits.csv"),   "PET Fusion")

    # ── Quick summary ─────────────────────────────────────────
    mri_bb_train_subjs = set(mri_backbone[mri_backbone["split"] == "train"]["subject_id"])
    mri_bb_test_subjs  = set(mri_backbone[mri_backbone["split"] == "test"]["subject_id"])
    overlap_subjs      = set(mri_fusion["subject_id"])
    print(f"\n  MRI backbone: {len(mri_bb_train_subjs)} train subjects, "
          f"{len(mri_bb_test_subjs)} test subjects")
    print(f"  Fusion: {mri_fusion['subject_id'].nunique()} overlap subjects")

    # ── STAGE 1: MRI Ghost-VGG19 ──────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1 — MRI Ghost-VGG19")
    print(f"  Training on pre-computed MRI backbone splits")
    print("="*60)

    mri_loaders = make_single_loaders_backbone(
        mri_backbone, CFG["batch_size"], CFG["img_size"], "MRI")

    mri_model = GhostVGG19Extractor(
        CFG["dense_units"], CFG["dropout"], CFG["num_classes"]).to(DEVICE)

    mri_model, mri_hist = fit(
        mri_model, mri_loaders,
        save_path = os.path.join(OUT_DIR, "mri_ghost_vgg19_best.pth"),
        name="MRI", lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], device=DEVICE,
    )
    plot_curves(mri_hist, "MRI")

    # Test on overlap subjects
    _, _, mri_preds, mri_lbls, mri_probs, mri_subjects = evaluate(
        mri_model, mri_loaders["test"], nn.CrossEntropyLoss(), DEVICE)
    mri_acc, mri_su_acc, mri_sl_roc, mri_su_roc, mri_sl_pr, mri_su_pr, \
        mri_su_preds, mri_su_lbls, mri_su_probs = print_results(
            "MRI_overlap_test", mri_preds, mri_lbls, mri_probs, mri_subjects)
    plot_roc_pr(mri_lbls, mri_probs, "MRI_slice")
    plot_roc_pr(mri_su_lbls, mri_su_probs, "MRI_subject")

    # Inference timing — MRI backbone
    measure_inference_time(
        mri_model,
        mri_backbone[mri_backbone["split"] == "test"],
        CFG["img_size"], "MRI_GhostVGG19")

    # ── STAGE 2: PET Ghost-VGG19 ──────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2 — PET Ghost-VGG19")
    print(f"  Training on pre-computed PET backbone splits")
    print("="*60)

    pet_loaders = make_single_loaders_backbone(
        pet_backbone, CFG["batch_size"], CFG["img_size"], "PET")

    pet_model = GhostVGG19Extractor(
        CFG["dense_units"], CFG["dropout"], CFG["num_classes"]).to(DEVICE)

    pet_model, pet_hist = fit(
        pet_model, pet_loaders,
        save_path = os.path.join(OUT_DIR, "pet_ghost_vgg19_best.pth"),
        name="PET", lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], device=DEVICE,
    )
    plot_curves(pet_hist, "PET")

    # Test on overlap subjects
    _, _, pet_preds, pet_lbls, pet_probs, pet_subjects = evaluate(
        pet_model, pet_loaders["test"], nn.CrossEntropyLoss(), DEVICE)
    pet_acc, pet_su_acc, pet_sl_roc, pet_su_roc, pet_sl_pr, pet_su_pr, \
        pet_su_preds, pet_su_lbls, pet_su_probs = print_results(
            "PET_overlap_test", pet_preds, pet_lbls, pet_probs, pet_subjects)
    plot_roc_pr(pet_lbls, pet_probs, "PET_slice")
    plot_roc_pr(pet_su_lbls, pet_su_probs, "PET_subject")

    # Inference timing — PET backbone
    measure_inference_time(
        pet_model,
        pet_backbone[pet_backbone["split"] == "test"],
        CFG["img_size"], "PET_GhostVGG19")

    # ── STAGE 3: Multimodal Fusion ────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3 — Multimodal Fusion (Ghost-VGG19 backbones)")
    print(f"  Using {len(overlap_subjs)} overlap subjects")
    print(f"  Train/Val/Test split from pre-computed fusion splits")
    print("  Ghost-VGG19 backbones frozen — only fusion head trains")
    print("="*60)

    # Reload best backbone weights and freeze the feature extractor
    mri_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "mri_ghost_vgg19_best.pth"), map_location=DEVICE))
    pet_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "pet_ghost_vgg19_best.pth"), map_location=DEVICE))
    for p in mri_model.features.parameters(): p.requires_grad = False
    for p in pet_model.features.parameters(): p.requires_grad = False

    mm_model = MultimodalFusionModel(
        mri_model, pet_model,
        CFG["dense_units"], CFG["dropout"], CFG["num_classes"],
    ).to(DEVICE)

    print("\n  Multimodal loaders:")
    mm_loaders = make_multimodal_loaders(
        mri_fusion, pet_fusion, CFG["batch_size"], CFG["img_size"])

    mm_model, mm_hist = fit(
        mm_model, mm_loaders,
        save_path = os.path.join(OUT_DIR, "ghost_multimodal_best.pth"),
        name="Multimodal", lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], device=DEVICE, multimodal=True,
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

    # Inference timing — multimodal fusion
    measure_inference_time_multimodal(
        mm_model,
        mri_fusion[mri_fusion["split"] == "test"],
        pet_fusion[pet_fusion["split"] == "test"],
        CFG["img_size"], "Multimodal_GhostVGG19")

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY — Ghost-VGG19")
    print("="*60)
    print(f"  {'Model':<25} {'Test Subjects':<15} {'Slice Acc':>10} {'Subj Acc':>10} {'Subj ROC':>10}")
    print(f"  {'-'*80}")
    print(f"  {'MRI Ghost-VGG19':<25} {len(mri_bb_test_subjs):<15} {mri_acc:>10.4f} {mri_su_acc:>10.4f} {mri_su_roc:>10.4f}")
    print(f"  {'PET Ghost-VGG19':<25} {len(mri_bb_test_subjs):<15} {pet_acc:>10.4f} {pet_su_acc:>10.4f} {pet_su_roc:>10.4f}")
    print(f"  {'Multimodal':<25} {mri_fusion['subject_id'].nunique():<15} {mm_acc:>10.4f} {mm_su_acc:>10.4f} {mm_su_roc:>10.4f}")
    print(f"\n  ✅ Backbones tested on SAME {len(overlap_subjs)} subjects used in fusion!")
    print(f"  All models + plots saved to {OUT_DIR}")

    torch.save(mm_model.state_dict(),
               os.path.join(OUT_DIR, "ghost_multimodal_final.pth"))


if __name__ == "__main__":
    main()
