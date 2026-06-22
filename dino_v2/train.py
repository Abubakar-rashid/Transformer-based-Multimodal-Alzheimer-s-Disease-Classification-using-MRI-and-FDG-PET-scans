import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve

from config import DEVICE, OUT_DIR, IDX2LABEL


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup for warmup_epochs, then cosine annealing to ~0."""
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        progress = (current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


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
    print(classification_report(sl_labels, sl_preds, target_names=["CN", "AD"], digits=4))

    su_preds, su_labels, su_probs, su_acc, su_df = subject_agg(
        sl_preds, sl_labels, sl_probs, subjects)
    fpr, tpr, _ = roc_curve(su_labels, su_probs)
    su_roc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(su_labels, su_probs)
    su_pr  = auc(rec, prec)

    print(f"  ── SUBJECT-LEVEL ({len(su_labels)} subjects) ──")
    print(f"  Accuracy : {su_acc:.4f}  |  ROC-AUC : {su_roc:.4f}  |  PR-AUC : {su_pr:.4f}")
    print(classification_report(su_labels, su_preds, target_names=["CN", "AD"], digits=4))

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
