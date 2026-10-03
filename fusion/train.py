"""
train.py — MLP fusion head training and evaluation.

run_pair() runs the full 5-seeds × 5-folds training loop for one fusion pair
and saves per-fold metrics, per-subject predictions, and seed summaries.
"""
import gc, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score,
)

from .config import (
    DEVICE, EPOCHS, BS, LR, WD, PATIENCE,
    BASE_SEED, SEEDS, K_FOLDS,
)
from .data import build_tensors


def make_head(in_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2),
    )


def train_fold(model, X_tr, y_tr, X_va, y_va):
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.CrossEntropyLoss()
    dl_tr = DataLoader(TensorDataset(X_tr.to(DEVICE), y_tr.to(DEVICE)), BS, shuffle=True)
    dl_va = DataLoader(TensorDataset(X_va.to(DEVICE), y_va.to(DEVICE)), BS)
    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([crit(model(xb), yb).item() for xb, yb in dl_va])
        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def _cuda_sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def _time_mlp_inference(model, x_sample, n_warmup=20, n_runs=100):
    """Return (mean_ms, std_ms) for MLP forward-pass on a single sample."""
    model.eval(); x = x_sample.to(DEVICE)
    with torch.no_grad():
        for _ in range(n_warmup): model(x)
        _cuda_sync()
        times = []
        for _ in range(n_runs):
            _cuda_sync()
            t0 = time.perf_counter()
            model(x)
            _cuda_sync()
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


@torch.no_grad()
def eval_model(model, X, y):
    model.eval()
    dl = DataLoader(TensorDataset(X.to(DEVICE), y.to(DEVICE)), BS)
    probs, preds, labels = [], [], []
    for xb, yb in dl:
        p = F.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
        probs.extend(p); preds.extend((p >= 0.5).astype(int)); labels.extend(yb.cpu().numpy())
    y_np = np.array(labels); yp = np.array(preds); p_np = np.array(probs)
    tn, fp, fn, tp = confusion_matrix(y_np, yp).ravel()
    return {
        "auc":     roc_auc_score(y_np, p_np),
        "pr_auc":  average_precision_score(y_np, p_np),
        "acc":     float((y_np == yp).mean()),
        "sens":    tp / (tp + fn) if (tp + fn) else 0.,
        "spec":    tn / (tn + fp) if (tn + fp) else 0.,
        "f1":      f1_score(y_np, yp, zero_division=0),
        "bal_acc": balanced_accuracy_score(y_np, yp),
        "probs":   p_np, "labels": y_np,
    }


def run_pair(pair, all_caches, global_subjects):
    """
    Train MLP fusion for one pair over SEEDS × K_FOLDS.
    KFold splits are fixed (BASE_SEED); only MLP init varies per seed.
    Saves:
      {out_dir}/{tag}_kfold_results.csv   — 25 rows (fold × seed)
      {out_dir}/{tag}_predictions.csv     — subject-level mean_prob
      {out_dir}/{tag}_seed_means.csv      — per-seed 5-fold mean
    Returns (all_kfold_rows, seed42_fold_results).
    """
    tag     = pair["tag"]; group = pair["group"]; lbl = pair["label"]
    out_dir = pair["out_dir"]

    b1_aug = all_caches[(pair["b1_arch"], pair["b1_mod"], "aug")]
    b2_aug = all_caches[(pair["b2_arch"], pair["b2_mod"], "aug")]
    b1_det = all_caches[(pair["b1_arch"], pair["b1_mod"], "det")]
    b2_det = all_caches[(pair["b2_arch"], pair["b2_mod"], "det")]

    pair_subjs  = sorted(set(b1_aug) & set(b2_aug) & set(b1_det) & set(b2_det) & set(global_subjects))
    pair_labels = np.array([b1_aug[s][1] for s in pair_subjs])
    ad_n = int(pair_labels.sum()); cn_n = len(pair_subjs) - ad_n
    s0 = next(iter(b1_aug.values()))[0][0]; s1 = next(iter(b2_aug.values()))[0][0]
    in_dim = len(s0) + len(s1)
    print(f"\n{'='*65}\n  [{group}]  {lbl}")
    print(f"  N={len(pair_subjs)}  AD={ad_n}  CN={cn_n}  fusion_in={in_dim}")

    skf         = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=BASE_SEED)
    fixed_folds = list(skf.split(pair_subjs, pair_labels))

    all_kfold_rows = []
    seed42_folds   = []
    subj_probs     = {s: [] for s in pair_subjs}
    subj_label     = {}
    subj_fold_asgn = {}
    mlp_ms, mlp_std = float("nan"), float("nan")

    import random
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        for fold, (tr_idx, va_idx) in enumerate(fixed_folds):
            tr_subjs = [pair_subjs[i] for i in tr_idx]
            va_subjs = [pair_subjs[i] for i in va_idx]
            X_tr, y_tr = build_tensors(b1_aug, b2_aug, tr_subjs, use_aug=True)
            X_va, y_va = build_tensors(b1_det, b2_det, va_subjs, use_aug=False)
            model = make_head(in_dim).to(DEVICE)
            model = train_fold(model, X_tr, y_tr, X_va, y_va)
            if seed == BASE_SEED and fold == 0:
                mlp_ms, mlp_std = _time_mlp_inference(model, X_va[:1].float())
            m = eval_model(model, X_va, y_va)
            del model; gc.collect(); torch.cuda.empty_cache()
            all_kfold_rows.append({"tag": tag, "fold": fold+1, "seed": seed,
                **{k: round(v, 4) for k, v in m.items() if not isinstance(v, np.ndarray)}})
            if seed == BASE_SEED:
                seed42_folds.append(m)
            for i, s in enumerate(va_subjs):
                subj_probs[s].append(float(m["probs"][i]))
                subj_label[s]     = int(m["labels"][i])
                subj_fold_asgn[s] = fold + 1

    pred_rows = []
    for s in pair_subjs:
        if not subj_probs[s]: continue
        p = float(np.mean(subj_probs[s]))
        pred_rows.append({"subject_id": s, "true_label": subj_label[s],
                          "mean_prob": round(p, 4),
                          "prob_std":  round(float(np.std(subj_probs[s])), 4),
                          "pred_label": int(p >= 0.5),
                          "fold": subj_fold_asgn[s], "group": group})

    seed_means = []
    for seed in SEEDS:
        rows_s = [r for r in all_kfold_rows if r["seed"] == seed]
        sm = {"seed": seed}
        for met in ["auc", "acc", "f1", "pr_auc", "bal_acc"]:
            vals = [r[met] for r in rows_s]
            sm[f"{met}_mean"] = round(float(np.mean(vals)), 4)
            sm[f"{met}_std"]  = round(float(np.std(vals)),  4)
        seed_means.append(sm)

    sm_aucs = [s["auc_mean"] for s in seed_means]
    print(f"  AUC: {np.mean(sm_aucs):.4f} ± {np.std(sm_aucs):.4f}  (mean±SD across {len(SEEDS)} seeds)")

    pd.DataFrame(all_kfold_rows).to_csv(f"{out_dir}/{tag}_kfold_results.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{out_dir}/{tag}_predictions.csv",         index=False)
    pd.DataFrame(seed_means).to_csv(f"{out_dir}/{tag}_seed_means.csv",         index=False)
    pd.DataFrame([{"tag": tag, "mlp_ms_per_sample": round(mlp_ms, 4),
                   "mlp_std_ms": round(mlp_std, 4),
                   "in_dim": in_dim}]).to_csv(f"{out_dir}/{tag}_timing.csv", index=False)

    return all_kfold_rows, seed42_folds
