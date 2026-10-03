import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from config import OUT_DIR


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
    ax1.plot([0, 1], [0, 1], "k--"); ax1.set_title(f"ROC — {name}"); ax1.legend()
    ax2.plot(rec, prec, lw=2, label=f"AUC={pr_auc:.3f}")
    ax2.set_title(f"PR — {name}"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_roc_pr.png"), dpi=150)
    plt.show()
