"""
baselines.py — Baseline classifiers for comparison with MLP fusion.

- eval_solo_lr:           LogReg on single-backbone features (12 models)
- eval_lr_stacking:       LogReg on concatenated backbone features (66 pairs)
- compute_averaging_baseline: probability averaging of two solo models (66 pairs)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score,
)

from .config import K_FOLDS, BASE_SEED, SOLO_DIR, BASE_DIR


def _save_fold_summary(tag, fold_results, out_path):
    mets = [k for k in fold_results[0] if not isinstance(fold_results[0][k], np.ndarray)]
    row = {"tag": tag}
    for m in mets:
        vs = [r[m] for r in fold_results]
        row[f"{m}_mean"] = round(float(np.mean(vs)), 4)
        row[f"{m}_std"]  = round(float(np.std(vs)),  4)
    pd.DataFrame([row]).to_csv(out_path, index=False)


def _lr_metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {"auc": roc_auc_score(y_true, probs),
            "pr_auc": average_precision_score(y_true, probs),
            "acc": float((y_true == preds).mean()),
            "sens": tp / (tp + fn + 1e-15), "spec": tn / (tn + fp + 1e-15),
            "f1": f1_score(y_true, preds, zero_division=0),
            "bal_acc": balanced_accuracy_score(y_true, preds),
            "probs": probs, "labels": y_true}


def eval_solo_lr(solo, all_caches, global_subjects):
    """Evaluate one backbone (arch+modality) with LogReg, 5-fold CV."""
    tag = solo["tag"]; group = solo["group"]; lbl = solo["label"]
    det = all_caches[(solo["arch"], solo["mod"], "det")]
    pair_subjs  = sorted(set(det) & set(global_subjects))
    pair_labels = np.array([det[s][1] for s in pair_subjs])
    ad_n = int(pair_labels.sum()); cn_n = len(pair_subjs) - ad_n
    print(f"\n  [SOLO]  {lbl}  N={len(pair_subjs)}  AD={ad_n}  CN={cn_n}")

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=BASE_SEED)
    csv_rows = []; fold_pred_rows = []; subj_probs = {s: [] for s in pair_subjs}
    subj_label = {}; subj_fold = {}; fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(pair_subjs, pair_labels)):
        tr_subjs = [pair_subjs[i] for i in tr_idx]
        va_subjs = [pair_subjs[i] for i in va_idx]
        X_tr = np.array([det[s][0][0] for s in tr_subjs])
        y_tr = np.array([det[s][1]    for s in tr_subjs])
        X_va = np.array([det[s][0][0] for s in va_subjs])
        y_va = np.array([det[s][1]    for s in va_subjs])
        clf  = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)[:, 1]
        m = _lr_metrics(y_va, probs)
        fold_results.append(m)
        csv_rows.append({"tag": tag, "fold": fold+1,
            **{k: round(v, 4) for k, v in m.items() if not isinstance(v, np.ndarray)}})
        for i, s in enumerate(va_subjs):
            fold_pred_rows.append({"subject_id": s, "fold": fold+1,
                                   "prob": round(float(probs[i]), 4),
                                   "pred_label": int(probs[i] >= 0.5),
                                   "true_label": int(y_va[i])})
            subj_probs[s].append(float(probs[i]))
            subj_label[s] = int(y_va[i]); subj_fold[s] = fold + 1

    pred_rows = [{"subject_id": s, "true_label": subj_label[s],
                  "mean_prob": round(float(np.mean(subj_probs[s])), 4),
                  "prob_std":  round(float(np.std(subj_probs[s])),  4),
                  "pred_label": int(np.mean(subj_probs[s]) >= 0.5),
                  "fold": subj_fold[s], "group": group}
                 for s in pair_subjs if subj_probs[s]]

    for met in ["auc", "acc", "f1"]:
        vs = [r[met] for r in fold_results]
        print(f"    {met.upper()}: {np.mean(vs):.4f} ± {np.std(vs):.4f}")

    pd.DataFrame(csv_rows).to_csv(f"{SOLO_DIR}/{tag}_kfold_results.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{SOLO_DIR}/{tag}_predictions.csv",   index=False)
    pd.DataFrame(fold_pred_rows).to_csv(f"{SOLO_DIR}/{tag}_fold_preds.csv", index=False)
    _save_fold_summary(tag, fold_results, f"{SOLO_DIR}/{tag}_summary.csv")
    return fold_results, fold_pred_rows


def eval_lr_stacking(pair, all_caches, global_subjects):
    """LR stacking: concatenate both branch features, fit LogReg, 5-fold CV."""
    tag = pair["tag"] + "_lr"; group = "lr_stack"
    b1_det = all_caches[(pair["b1_arch"], pair["b1_mod"], "det")]
    b2_det = all_caches[(pair["b2_arch"], pair["b2_mod"], "det")]
    pair_subjs  = sorted(set(b1_det) & set(b2_det) & set(global_subjects))
    pair_labels = np.array([b1_det[s][1] for s in pair_subjs])

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=BASE_SEED)
    csv_rows = []; subj_probs = {s: [] for s in pair_subjs}
    subj_label = {}; subj_fold = {}; fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(pair_subjs, pair_labels)):
        tr_s = [pair_subjs[i] for i in tr_idx]
        va_s = [pair_subjs[i] for i in va_idx]
        X_tr = np.array([np.concatenate([b1_det[s][0][0], b2_det[s][0][0]]) for s in tr_s])
        y_tr = np.array([b1_det[s][1] for s in tr_s])
        X_va = np.array([np.concatenate([b1_det[s][0][0], b2_det[s][0][0]]) for s in va_s])
        y_va = np.array([b1_det[s][1] for s in va_s])
        clf  = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)[:, 1]
        m = _lr_metrics(y_va, probs)
        fold_results.append(m)
        csv_rows.append({"tag": tag, "fold": fold+1,
            **{k: round(v, 4) for k, v in m.items() if not isinstance(v, np.ndarray)}})
        for i, s in enumerate(va_s):
            subj_probs[s].append(float(probs[i]))
            subj_label[s] = int(y_va[i]); subj_fold[s] = fold + 1

    pred_rows = [{"subject_id": s, "true_label": subj_label[s],
                  "mean_prob": round(float(np.mean(subj_probs[s])), 4),
                  "prob_std":  round(float(np.std(subj_probs[s])),  4),
                  "pred_label": int(np.mean(subj_probs[s]) >= 0.5),
                  "fold": subj_fold[s], "group": group}
                 for s in pair_subjs if subj_probs[s]]

    pd.DataFrame(csv_rows).to_csv(f"{BASE_DIR}/{tag}_kfold_results.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{BASE_DIR}/{tag}_predictions.csv",   index=False)
    _save_fold_summary(tag, fold_results, f"{BASE_DIR}/{tag}_summary.csv")
    return fold_results


def compute_averaging_baseline(pair, solo_fold_preds):
    """Probability averaging ensemble: mean(prob_b1, prob_b2) per subject per fold."""
    tag    = pair["tag"] + "_avg"; group = "avg"
    b1_tag = f"{pair['b1_arch']}_{pair['b1_mod']}_solo"
    b2_tag = f"{pair['b2_arch']}_{pair['b2_mod']}_solo"
    if b1_tag not in solo_fold_preds or b2_tag not in solo_fold_preds:
        print(f"  [avg] Skipping {tag} — solo preds missing"); return []

    df1 = solo_fold_preds[b1_tag]; df2 = solo_fold_preds[b2_tag]
    mg  = df1.merge(df2, on=["subject_id", "fold"], suffixes=("_b1", "_b2"))
    if mg.empty: return []
    mg["avg_prob"] = (mg["prob_b1"] + mg["prob_b2"]) / 2.0

    csv_rows = []; subj_probs = {}; subj_label = {}; subj_fold = {}
    fold_results = []
    for fold, gdf in mg.groupby("fold"):
        y_va  = gdf["true_label_b1"].values
        probs = gdf["avg_prob"].values
        m = _lr_metrics(y_va, probs)
        fold_results.append(m)
        csv_rows.append({"tag": tag, "fold": fold,
            **{k: round(v, 4) for k, v in m.items() if not isinstance(v, np.ndarray)}})
        for _, row in gdf.iterrows():
            s = row["subject_id"]
            subj_probs.setdefault(s, []).append(float(row["avg_prob"]))
            subj_label[s] = int(row["true_label_b1"]); subj_fold[s] = int(fold)

    pred_rows = [{"subject_id": s, "true_label": subj_label[s],
                  "mean_prob": round(float(np.mean(subj_probs[s])), 4),
                  "prob_std":  round(float(np.std(subj_probs[s])),  4),
                  "pred_label": int(np.mean(subj_probs[s]) >= 0.5),
                  "fold": subj_fold[s], "group": group}
                 for s in subj_probs]

    pd.DataFrame(csv_rows).to_csv(f"{BASE_DIR}/{tag}_kfold_results.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{BASE_DIR}/{tag}_predictions.csv",   index=False)
    _save_fold_summary(tag, fold_results, f"{BASE_DIR}/{tag}_summary.csv")
    return fold_results
