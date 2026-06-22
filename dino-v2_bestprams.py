import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from collections import defaultdict
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
# CONFIG  ── Best configs from HPO logs
# ==============================================================
BASE    = "/kaggle/input/datasets/adaisdiashdh/mri-pet-slices"
MRI_DIR = "/kaggle/input/datasets/adaisdiashdh/mri-pet-slices/MRI_Slices"
PET_DIR = "/kaggle/input/datasets/adaisdiashdh/mri-pet-slices/PET_Slices"
OUT_DIR = "/kaggle/working/"

# Directory where the pre-computed split CSVs live
SPLIT_DIR = "/kaggle/input/datasets/jannatttt/splits1"

# ── Shared / architecture settings (unchanged) ────────────────
CFG_SHARED = dict(
    epochs        = 15,
    patience      = 7,
    num_classes   = 2,
    img_size      = 224,
    dinov2_model  = "dinov2_vitb14",
    vit_feat_dim  = 768,
)

# ── MRI backbone: HPO Trial #1 (val_acc = 0.7692) ─────────────
CFG_MRI = dict(
    lr            = 2.2948683681130543e-05,
    weight_decay  = 0.0003511356313970409,
    dropout       = 0.17336180394137352,
    dense_units   = 256,
    batch_size    = 32,
    warmup_epochs = 5,
)

# ── PET backbone: HPO Trial #14 (val_acc = 0.8753) ────────────
CFG_PET = dict(
    lr            = 2.5507545599395824e-05,
    weight_decay  = 0.014105900001541825,
    dropout       = 0.22642705359216325,
    dense_units   = 256,
    batch_size    = 32,
    warmup_epochs = 5,
)

# ── Fusion MLP: HPO Trial #0 (val_acc = 0.7126, best so far) ──
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

# ==============================================================
# PATH REMAPPING  ── old server paths → Kaggle paths
# ==============================================================
OLD_PATH_PREFIX = "/hdd3/seecs/abubakar.seecs/adni/datasets"
NEW_PATH_PREFIX = "/kaggle/input/datasets/adaisdiashdh/mri-pet-slices"

def remap_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["slice_path"] = df["slice_path"].str.replace(
        OLD_PATH_PREFIX, NEW_PATH_PREFIX, regex=False
    )
    sample_new = df["slice_path"].iloc[0]
    print(f"    path remapped → {sample_new}")
    return df

# ==============================================================
# 1. LOAD PRE-COMPUTED SPLITS FROM CSV
# ==============================================================
def load_split_csv(csv_path, name=""):
    df = pd.read_csv(csv_path)
    assert set(df.columns) >= {"subject_id", "group", "slice_path", "split"}, \
        f"CSV {csv_path} must have columns: subject_id, group, slice_path, split"

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
# 3. MODEL ARCHITECTURE  ── DINOv2 Backbone
# ==============================================================
class DINOv2Extractor(nn.Module):
    """
    DINOv2 backbone with a projection head and classification layer.
    dense_units is passed in per-modality so MRI and PET can have
    different head sizes if HPO selects them (currently both 256).
    """
    def __init__(self,
                 model_name  = "dinov2_vitb14",
                 feat_dim    = 768,
                 dense_units = 256,
                 dropout     = 0.3,
                 num_classes = 2):
        super().__init__()
        print(f"  Loading DINOv2 backbone: {model_name} (feat_dim={feat_dim})")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
            verbose=False,
        )
        self.proj   = nn.Linear(feat_dim, dense_units)
        self.relu   = nn.ReLU(inplace=True)
        self.drop   = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dense_units, num_classes)

    def forward_features(self, x):
        out     = self.backbone.forward_features(x)
        cls_tok = out["x_norm_clstoken"]          # [B, feat_dim]
        emb     = self.relu(self.proj(cls_tok))   # [B, dense_units]
        emb     = self.drop(emb)
        return emb                                # [B, dense_units]

    def forward(self, x):
        return self.fc_out(self.forward_features(x))


class MultimodalFusionModel(nn.Module):
    """
    Concatenates MRI and PET DINOv2 embeddings and passes through a fusion MLP.
    The fusion head uses CFG_FUSION dense_units and dropout independently of
    the backbone heads — this is important since HPO optimised them separately.

    Input feature dim to fusion = mri_dense_units + pet_dense_units.
    Both are 256 from HPO, so fusion input = 512.
    """
    def __init__(self, mri_ext, pet_ext,
                 fusion_dense_units = 128,   # from CFG_FUSION
                 fusion_dropout     = 0.393, # from CFG_FUSION
                 num_classes        = 2):
        super().__init__()
        self.mri_stream = mri_ext
        self.pet_stream = pet_ext

        # Input dim = mri dense_units + pet dense_units
        mri_out_dim = mri_ext.fc_out.in_features   # 256
        pet_out_dim = pet_ext.fc_out.in_features   # 256
        fusion_in   = mri_out_dim + pet_out_dim    # 512

        self.fusion = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_in, fusion_dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_dense_units, fusion_dense_units),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dense_units, num_classes),
        )

    def forward(self, mri, pet):
        f = torch.cat([
            self.mri_stream.forward_features(mri),
            self.pet_stream.forward_features(pet),
        ], dim=1)
        return self.fusion(f)

# ==============================================================
# 4. LR SCHEDULER  ── linear warmup then cosine decay
# ==============================================================
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        progress = (current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

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
    fpr, tpr, _ = roc_curve(sl_labels, sl_probs)
    sl_roc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(sl_labels, sl_probs)
    sl_pr  = auc(rec, prec)
    sl_acc = (sl_preds == sl_labels).mean()

    print(f"\n  ── SLICE-LEVEL ({len(sl_labels)} slices) ──")
    print(f"  Accuracy : {sl_acc:.4f}  |  ROC-AUC : {sl_roc:.4f}  |  PR-AUC : {sl_pr:.4f}")
    print(classification_report(sl_labels, sl_preds, target_names=["CN","AD"], digits=4))

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
# 6. INFERENCE TIMING
# ==============================================================
@torch.no_grad()
def measure_inference_time(model, test_df, img_size, name, multimodal=False):
    print(f"\n{'='*70}")
    print(f"  INFERENCE TIMING — {name}")
    print(f"{'='*70}")

    transform = get_transforms(img_size, split="val")
    subjects  = sorted(test_df["subject_id"].unique())

    subject_data = {}
    for subj in subjects:
        subj_df = test_df[test_df["subject_id"] == subj]
        imgs = []
        for _, row in subj_df.iterrows():
            img = Image.open(row["slice_path"]).convert("RGB")
            img = transform(img)
            imgs.append(img)
        subject_data[subj] = torch.stack(imgs)

    results = []
    has_gpu = torch.cuda.is_available()

    gpu_times = {}
    if has_gpu:
        gpu_device = torch.device("cuda")
        model_gpu  = model.to(gpu_device)
        model_gpu.eval()
        dummy = subject_data[subjects[0]].to(gpu_device)
        _ = model_gpu(dummy)
        torch.cuda.synchronize()
        for subj in subjects:
            batch = subject_data[subj].to(gpu_device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model_gpu(batch)
            end_event.record()
            torch.cuda.synchronize()
            gpu_times[subj] = start_event.elapsed_time(end_event)

    cpu_device = torch.device("cpu")
    model_cpu  = model.to(cpu_device)
    model_cpu.eval()
    cpu_times = {}
    dummy = subject_data[subjects[0]].to(cpu_device)
    _ = model_cpu(dummy)
    for subj in subjects:
        batch = subject_data[subj].to(cpu_device)
        t0 = time.perf_counter()
        _ = model_cpu(batch)
        t1 = time.perf_counter()
        cpu_times[subj] = (t1 - t0) * 1000.0

    header = f"  {'Subject':<18} | {'#Slices':>7} | {'GPU (ms)':>10} | {'CPU (ms)':>10}"
    print(header)
    print(f"  {'-'*57}")
    for subj in subjects:
        n_slices = len(subject_data[subj])
        gpu_ms   = gpu_times.get(subj, float("nan"))
        cpu_ms   = cpu_times[subj]
        gpu_str  = f"{gpu_ms:>10.2f}" if has_gpu else f"{'N/A':>10}"
        print(f"  {subj:<18} | {n_slices:>7} | {gpu_str} | {cpu_ms:>10.2f}")
        results.append({
            "subject": subj, "n_slices": n_slices,
            "gpu_ms": gpu_ms if has_gpu else None, "cpu_ms": cpu_ms,
        })

    avg_slices = np.mean([r["n_slices"] for r in results])
    avg_cpu    = np.mean([r["cpu_ms"] for r in results])
    std_cpu    = np.std([r["cpu_ms"] for r in results])
    print(f"  {'-'*57}")
    if has_gpu:
        avg_gpu = np.mean([r["gpu_ms"] for r in results])
        std_gpu = np.std([r["gpu_ms"] for r in results])
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
    print(f"\n{'='*70}")
    print(f"  INFERENCE TIMING — {name} (Multimodal)")
    print(f"{'='*70}")

    transform = get_transforms(img_size, split="val")
    common    = sorted(set(mri_test_df["subject_id"]) & set(pet_test_df["subject_id"]))

    subject_data = {}
    for subj in common:
        mri_paths = mri_test_df[mri_test_df["subject_id"] == subj]["slice_path"].tolist()
        pet_paths = pet_test_df[pet_test_df["subject_id"] == subj]["slice_path"].tolist()
        mri_imgs, pet_imgs = [], []
        for mp in mri_paths:
            for pp in pet_paths:
                mri_imgs.append(transform(Image.open(mp).convert("RGB")))
                pet_imgs.append(transform(Image.open(pp).convert("RGB")))
        subject_data[subj] = (torch.stack(mri_imgs), torch.stack(pet_imgs))

    results = []
    has_gpu = torch.cuda.is_available()

    gpu_times = {}
    if has_gpu:
        gpu_device = torch.device("cuda")
        model_gpu  = model.to(gpu_device)
        model_gpu.eval()
        mri_d, pet_d = subject_data[common[0]]
        _ = model_gpu(mri_d.to(gpu_device), pet_d.to(gpu_device))
        torch.cuda.synchronize()
        for subj in common:
            mri_b, pet_b = subject_data[subj]
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model_gpu(mri_b.to(gpu_device), pet_b.to(gpu_device))
            end.record()
            torch.cuda.synchronize()
            gpu_times[subj] = start.elapsed_time(end)

    cpu_device = torch.device("cpu")
    model_cpu  = model.to(cpu_device)
    model_cpu.eval()
    cpu_times  = {}
    mri_d, pet_d = subject_data[common[0]]
    _ = model_cpu(mri_d.to(cpu_device), pet_d.to(cpu_device))
    for subj in common:
        mri_b, pet_b = subject_data[subj]
        t0 = time.perf_counter()
        _ = model_cpu(mri_b.to(cpu_device), pet_b.to(cpu_device))
        cpu_times[subj] = (time.perf_counter() - t0) * 1000.0

    header = f"  {'Subject':<18} | {'#Pairs':>7} | {'GPU (ms)':>10} | {'CPU (ms)':>10}"
    print(header)
    print(f"  {'-'*57}")
    for subj in common:
        n_pairs = len(subject_data[subj][0])
        gpu_ms  = gpu_times.get(subj, float("nan"))
        cpu_ms  = cpu_times[subj]
        gpu_str = f"{gpu_ms:>10.2f}" if has_gpu else f"{'N/A':>10}"
        print(f"  {subj:<18} | {n_pairs:>7} | {gpu_str} | {cpu_ms:>10.2f}")
        results.append({
            "subject": subj, "n_pairs": n_pairs,
            "gpu_ms": gpu_ms if has_gpu else None, "cpu_ms": cpu_ms,
        })

    avg_pairs = np.mean([r["n_pairs"] for r in results])
    avg_cpu   = np.mean([r["cpu_ms"] for r in results])
    std_cpu   = np.std([r["cpu_ms"] for r in results])
    print(f"  {'-'*57}")
    if has_gpu:
        avg_gpu = np.mean([r["gpu_ms"] for r in results])
        std_gpu = np.std([r["gpu_ms"] for r in results])
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
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n=== HPO-tuned Configuration ===")
    print(f"  MRI  → lr={CFG_MRI['lr']:.2e}  wd={CFG_MRI['weight_decay']:.2e}"
          f"  do={CFG_MRI['dropout']:.3f}  du={CFG_MRI['dense_units']}  bs={CFG_MRI['batch_size']}  wu={CFG_MRI['warmup_epochs']}")
    print(f"  PET  → lr={CFG_PET['lr']:.2e}  wd={CFG_PET['weight_decay']:.2e}"
          f"  do={CFG_PET['dropout']:.3f}  du={CFG_PET['dense_units']}  bs={CFG_PET['batch_size']}  wu={CFG_PET['warmup_epochs']}")
    print(f"  FUSE → lr={CFG_FUSION['lr']:.2e}  wd={CFG_FUSION['weight_decay']:.2e}"
          f"  do={CFG_FUSION['dropout']:.3f}  du={CFG_FUSION['dense_units']}  bs={CFG_FUSION['batch_size']}  wu={CFG_FUSION['warmup_epochs']}")
    print(f"  NOTE: Fusion config is from best trial found so far (HPO incomplete).")
    print(f"        Replace CFG_FUSION values if HPO yields a better trial.\n")

    # ── Load pre-computed splits from CSV ─────────────────────
    print("\n=== Loading Pre-Computed Splits ===")
    mri_backbone = remap_paths(load_split_csv(os.path.join(SPLIT_DIR, "mri_backbone_splits.csv"), "MRI Backbone"))
    pet_backbone = remap_paths(load_split_csv(os.path.join(SPLIT_DIR, "pet_backbone_splits.csv"), "PET Backbone"))
    mri_fusion   = remap_paths(load_split_csv(os.path.join(SPLIT_DIR, "mri_fusion_splits.csv"),   "MRI Fusion"))
    pet_fusion   = remap_paths(load_split_csv(os.path.join(SPLIT_DIR, "pet_fusion_splits.csv"),   "PET Fusion"))

    mri_bb_train_subjs = set(mri_backbone[mri_backbone["split"] == "train"]["subject_id"])
    mri_bb_test_subjs  = set(mri_backbone[mri_backbone["split"] == "test"]["subject_id"])
    print(f"\n  MRI backbone: {len(mri_bb_train_subjs)} train subjects, "
          f"{len(mri_bb_test_subjs)} test subjects")
    print(f"  Fusion: {mri_fusion['subject_id'].nunique()} overlap subjects")

    # ── STAGE 1: MRI DINOv2 (HPO best config) ─────────────────
    print("\n" + "="*60)
    print("STAGE 1 — MRI DINOv2 ViT-B/14  [HPO Trial #1, val_acc=0.7692]")
    print(f"  lr={CFG_MRI['lr']:.2e}  wd={CFG_MRI['weight_decay']:.2e}"
          f"  dropout={CFG_MRI['dropout']:.3f}  dense={CFG_MRI['dense_units']}"
          f"  bs={CFG_MRI['batch_size']}  warmup={CFG_MRI['warmup_epochs']}")
    print("="*60)

    mri_loaders = make_single_loaders_backbone(
        mri_backbone, CFG_MRI["batch_size"], CFG_SHARED["img_size"], "MRI")

    mri_model = DINOv2Extractor(
        model_name  = CFG_SHARED["dinov2_model"],
        feat_dim    = CFG_SHARED["vit_feat_dim"],
        dense_units = CFG_MRI["dense_units"],
        dropout     = CFG_MRI["dropout"],
        num_classes = CFG_SHARED["num_classes"],
    ).to(DEVICE)

    mri_model, mri_hist = fit(
        mri_model, mri_loaders,
        save_path     = os.path.join(OUT_DIR, "mri_dinov2_best.pth"),
        name          = "MRI",
        lr            = CFG_MRI["lr"],
        epochs        = CFG_SHARED["epochs"],
        patience      = CFG_SHARED["patience"],
        device        = DEVICE,
        weight_decay  = CFG_MRI["weight_decay"],
        warmup_epochs = CFG_MRI["warmup_epochs"],
    )
    plot_curves(mri_hist, "MRI")

    _, _, mri_preds, mri_lbls, mri_probs, mri_subjects = evaluate(
        mri_model, mri_loaders["test"], nn.CrossEntropyLoss(), DEVICE)
    mri_acc, mri_su_acc, mri_sl_roc, mri_su_roc, mri_sl_pr, mri_su_pr, \
        mri_su_preds, mri_su_lbls, mri_su_probs = print_results(
            "MRI_overlap_test", mri_preds, mri_lbls, mri_probs, mri_subjects)
    plot_roc_pr(mri_lbls, mri_probs, "MRI_slice")
    plot_roc_pr(mri_su_lbls, mri_su_probs, "MRI_subject")

    mri_test_df = mri_backbone[mri_backbone["split"] == "test"]
    measure_inference_time(mri_model, mri_test_df, CFG_SHARED["img_size"], "MRI_DINOv2")

    # ── STAGE 2: PET DINOv2 (HPO best config) ─────────────────
    print("\n" + "="*60)
    print("STAGE 2 — PET DINOv2 ViT-B/14  [HPO Trial #14, val_acc=0.8753]")
    print(f"  lr={CFG_PET['lr']:.2e}  wd={CFG_PET['weight_decay']:.2e}"
          f"  dropout={CFG_PET['dropout']:.3f}  dense={CFG_PET['dense_units']}"
          f"  bs={CFG_PET['batch_size']}  warmup={CFG_PET['warmup_epochs']}")
    print("="*60)

    pet_loaders = make_single_loaders_backbone(
        pet_backbone, CFG_PET["batch_size"], CFG_SHARED["img_size"], "PET")

    pet_model = DINOv2Extractor(
        model_name  = CFG_SHARED["dinov2_model"],
        feat_dim    = CFG_SHARED["vit_feat_dim"],
        dense_units = CFG_PET["dense_units"],
        dropout     = CFG_PET["dropout"],
        num_classes = CFG_SHARED["num_classes"],
    ).to(DEVICE)

    pet_model, pet_hist = fit(
        pet_model, pet_loaders,
        save_path     = os.path.join(OUT_DIR, "pet_dinov2_best.pth"),
        name          = "PET",
        lr            = CFG_PET["lr"],
        epochs        = CFG_SHARED["epochs"],
        patience      = CFG_SHARED["patience"],
        device        = DEVICE,
        weight_decay  = CFG_PET["weight_decay"],
        warmup_epochs = CFG_PET["warmup_epochs"],
    )
    plot_curves(pet_hist, "PET")

    _, _, pet_preds, pet_lbls, pet_probs, pet_subjects = evaluate(
        pet_model, pet_loaders["test"], nn.CrossEntropyLoss(), DEVICE)
    pet_acc, pet_su_acc, pet_sl_roc, pet_su_roc, pet_sl_pr, pet_su_pr, \
        pet_su_preds, pet_su_lbls, pet_su_probs = print_results(
            "PET_overlap_test", pet_preds, pet_lbls, pet_probs, pet_subjects)
    plot_roc_pr(pet_lbls, pet_probs, "PET_slice")
    plot_roc_pr(pet_su_lbls, pet_su_probs, "PET_subject")

    pet_test_df = pet_backbone[pet_backbone["split"] == "test"]
    measure_inference_time(pet_model, pet_test_df, CFG_SHARED["img_size"], "PET_DINOv2")

    # ── STAGE 3: Multimodal Fusion (HPO best config) ──────────
    print("\n" + "="*60)
    print("STAGE 3 — Multimodal Fusion  [HPO Trial #0 (best so far), val_acc=0.7126]")
    print(f"  lr={CFG_FUSION['lr']:.2e}  wd={CFG_FUSION['weight_decay']:.2e}"
          f"  dropout={CFG_FUSION['dropout']:.3f}  dense={CFG_FUSION['dense_units']}"
          f"  bs={CFG_FUSION['batch_size']}  warmup={CFG_FUSION['warmup_epochs']}")
    print(f"  DINOv2 backbone frozen — only projection heads + fusion MLP train")
    print("="*60)

    # Reload best backbone weights
    mri_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "mri_dinov2_best.pth"), map_location=DEVICE))
    pet_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "pet_dinov2_best.pth"), map_location=DEVICE))

    # Freeze DINOv2 backbone — keep proj / fc_out trainable
    for p in mri_model.backbone.parameters():  p.requires_grad = False
    for p in pet_model.backbone.parameters():  p.requires_grad = False

    mm_model = MultimodalFusionModel(
        mri_model, pet_model,
        fusion_dense_units = CFG_FUSION["dense_units"],
        fusion_dropout     = CFG_FUSION["dropout"],
        num_classes        = CFG_SHARED["num_classes"],
    ).to(DEVICE)

    print("\n  Multimodal loaders:")
    mm_loaders = make_multimodal_loaders(
        mri_fusion, pet_fusion, CFG_FUSION["batch_size"], CFG_SHARED["img_size"])

    mm_model, mm_hist = fit(
        mm_model, mm_loaders,
        save_path     = os.path.join(OUT_DIR, "multimodal_dinov2_best.pth"),
        name          = "Multimodal",
        lr            = CFG_FUSION["lr"],
        epochs        = CFG_SHARED["epochs"],
        patience      = CFG_SHARED["patience"],
        device        = DEVICE,
        weight_decay  = CFG_FUSION["weight_decay"],
        warmup_epochs = CFG_FUSION["warmup_epochs"],
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

    mri_fusion_test = mri_fusion[mri_fusion["split"] == "test"]
    pet_fusion_test = pet_fusion[pet_fusion["split"] == "test"]
    measure_inference_time_multimodal(
        mm_model, mri_fusion_test, pet_fusion_test,
        CFG_SHARED["img_size"], "Multimodal_DINOv2_Fusion")

    # ── Final summary ─────────────────────────────────────────
    n_bb_test  = mri_backbone[mri_backbone["split"] == "test"]["subject_id"].nunique()
    n_fus_test = mri_fusion[mri_fusion["split"] == "test"]["subject_id"].nunique()

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"  {'Model':<25} {'Test Subjects':<15} {'Slice Acc':>10} {'Subj Acc':>10} {'Subj ROC':>10}")
    print(f"  {'-'*75}")
    print(f"  {'MRI DINOv2':<25} {f'{n_bb_test} overlap':<15} {mri_acc:>10.4f} {mri_su_acc:>10.4f} {mri_su_roc:>10.4f}")
    print(f"  {'PET DINOv2':<25} {f'{n_bb_test} overlap':<15} {pet_acc:>10.4f} {pet_su_acc:>10.4f} {pet_su_roc:>10.4f}")
    print(f"  {'Multimodal DINOv2 Fusion':<25} {f'{n_fus_test} overlap':<15} {mm_acc:>10.4f} {mm_su_acc:>10.4f} {mm_su_roc:>10.4f}")
    print(f"\n   All models use HPO-tuned hyperparameters")
    print(f"   Inference times measured per-subject on CPU & GPU")
    print(f"    Fusion HPO was incomplete — update CFG_FUSION when full HPO finishes")
    print(f"  All models + plots saved to {OUT_DIR}")

    torch.save(mm_model.state_dict(),
               os.path.join(OUT_DIR, "multimodal_dinov2_final.pth"))


if __name__ == "__main__":
    main()