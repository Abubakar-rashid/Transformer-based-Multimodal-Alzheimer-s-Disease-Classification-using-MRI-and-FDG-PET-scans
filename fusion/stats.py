"""
stats.py — Statistical tests for comparing fusion models.

Primitives:  delong_test, mcnemar_test, bootstrap_bca, bootstrap_ci_pair,
             wilson_ci, clopper_pearson, spiegelhalter_z, brier_decomposition,
             ece, nadeau_bengio, cochran_q, friedman_nemenyi, yule_q,
             tost_test, permutation_group, hanley_mcneil, run_lmm

Analysis:    stats_per_model, stats_pairwise, stats_omnibus,
             stats_group, stats_diversity_gain, stats_ablation

Loaders:     load_stats_inputs, build_aligned
"""
import os, itertools
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    brier_score_loss, cohen_kappa_score,
)
from statsmodels.stats.multitest import multipletests

from .config import (
    STATS_DIR, CROSS_DIR, SOLO_DIR, BASE_DIR,
    N_BOOT, N_BOOT_P, N_PERM, N_PERM_P, ALPHA,
    N_TRAIN, N_TEST, rng,
)

# ── Statistical primitives ────────────────────────────────────────────────────

def _placement_values(y_true, y_pred):
    pos = y_pred[y_true == 1]; neg = y_pred[y_true == 0]
    V10 = np.array([np.mean(p > neg) + 0.5*np.mean(p == neg) for p in pos])
    V01 = np.array([np.mean(pos > q) + 0.5*np.mean(pos == q) for q in neg])
    return V10, V01


def delong_test(y_true, p1, p2):
    n1 = int(y_true.sum()); n0 = len(y_true) - n1
    V10_1, V01_1 = _placement_values(y_true, p1)
    V10_2, V01_2 = _placement_values(y_true, p2)
    s10 = np.cov(V10_1, V10_2, ddof=1) / n1
    s01 = np.cov(V01_1, V01_2, ddof=1) / n0
    cov = s10 + s01
    var_diff = cov[0,0] + cov[1,1] - 2*cov[0,1]
    z = (V10_1.mean() - V10_2.mean()) / (np.sqrt(var_diff) + 1e-15)
    return float(z), float(2*scipy.stats.norm.sf(abs(z))), float(V10_1.mean()), float(V10_2.mean())


def mcnemar_test(y_true, pred1, pred2):
    b = int(np.sum((pred1 == y_true) & (pred2 != y_true)))
    c = int(np.sum((pred1 != y_true) & (pred2 == y_true)))
    n = b + c
    if n == 0: return 1., 1., b, c
    k = min(b, c)
    p_e = float(min(1., 2*scipy.stats.binom.cdf(k, n, 0.5)))
    p_m = float(max(0., p_e - scipy.stats.binom.pmf(k, n, 0.5)))
    return p_e, p_m, b, c


def bootstrap_ci_pair(y_true, p1, p2, fn, n_boot=N_BOOT_P):
    n = len(y_true); obs = fn(y_true, p1) - fn(y_true, p2); diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n); yt = y_true[idx]
        if len(np.unique(yt)) < 2: continue
        try: diffs.append(fn(yt, p1[idx]) - fn(yt, p2[idx]))
        except Exception: pass
    d = np.array(diffs)
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def bootstrap_bca(y_true, y_pred, fn, n_boot=N_BOOT):
    n = len(y_true); obs = fn(y_true, y_pred); boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n); yt = y_true[idx]
        if len(np.unique(yt)) < 2: continue
        try: boot.append(fn(yt, y_pred[idx]))
        except Exception: pass
    boot = np.array(boot)
    if len(boot) < 10:
        return float(obs), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    z0 = scipy.stats.norm.ppf(np.clip(np.mean(boot < obs), 1e-6, 1-1e-6))
    jack = []
    for i in range(n):
        idx_j = np.concatenate([np.arange(i), np.arange(i+1, n)]); yt_j = y_true[idx_j]
        try: jack.append(fn(yt_j, y_pred[idx_j]) if len(np.unique(yt_j)) >= 2 else obs)
        except Exception: jack.append(obs)
    jack = np.array(jack); jm = np.mean(jack)
    a = np.sum((jm - jack)**3) / (6*(np.sum((jm - jack)**2)**1.5) + 1e-15)
    def _adj(za): return scipy.stats.norm.cdf(z0 + (z0+za) / (1 - a*(z0+za) + 1e-15))
    lo = float(np.percentile(boot, 100*np.clip(_adj(scipy.stats.norm.ppf(ALPHA/2)), 0.001, 0.999)))
    hi = float(np.percentile(boot, 100*np.clip(_adj(scipy.stats.norm.ppf(1-ALPHA/2)), 0.001, 0.999)))
    return float(obs), lo, hi


def wilson_ci(k, n):
    p = k/n; z = scipy.stats.norm.ppf(1 - ALPHA/2); d = 1 + z**2/n
    return (float((p + z**2/(2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / d),
            float((p + z**2/(2*n) + z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / d))


def clopper_pearson(k, n):
    lo = float(scipy.stats.beta.ppf(ALPHA/2, k, n-k+1)) if k > 0 else 0.
    hi = float(scipy.stats.beta.ppf(1-ALPHA/2, k+1, n-k)) if k < n else 1.
    return lo, hi


def spiegelhalter_z(y_true, y_prob):
    p = y_prob.astype(float); y = y_true.astype(float)
    num = np.sum((y - p) * (1 - 2*p))
    den = np.sqrt(np.sum((1 - 2*p)**2 * p * (1-p)) + 1e-15)
    z = num / den
    return float(z), float(2*scipy.stats.norm.sf(abs(z)))


def brier_decomposition(y_true, y_prob, n_bins=10):
    brier = float(brier_score_loss(y_true, y_prob)); base = y_true.mean()
    bins = np.linspace(0, 1, n_bins+1); bi = np.digitize(y_prob, bins[1:-1])
    rel = res = 0.; n = len(y_true)
    for k in range(n_bins):
        mask = bi == k
        if not mask.any(): continue
        nk = mask.sum(); ok = y_prob[mask].mean(); yk = y_true[mask].mean()
        rel += nk*(ok - yk)**2; res += nk*(yk - base)**2
    return brier, float(rel/n), float(res/n), float(base*(1-base))


def ece(y_true, y_prob, n_bins=5):
    quantiles = np.quantile(y_prob, np.linspace(0, 1, n_bins+1))
    quantiles = np.unique(quantiles)
    v = 0.; n = len(y_true)
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi)
        if not mask.any(): continue
        v += mask.sum() / n * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(v)


def nadeau_bengio(fold_v1, fold_v2):
    d = np.asarray(fold_v1) - np.asarray(fold_v2)
    k = len(d); mu = d.mean(); s2 = d.var(ddof=1)
    cv = (1/k + N_TEST/N_TRAIN) * s2
    if cv <= 0: return np.nan, 1.
    t = mu / np.sqrt(cv)
    return float(t), float(2*scipy.stats.t.sf(abs(t), df=k-1))


def cochran_q(correct_mat):
    k = correct_mat.shape[1]
    L = correct_mat.sum(0); R = correct_mat.sum(1)
    num = k*(k-1)*np.sum(L**2) - (k-1)*np.sum(L)**2
    den = k*np.sum(R) - np.sum(R**2)
    if den == 0: return 0., 1.
    Q = float(num/den)
    return Q, float(scipy.stats.chi2.sf(Q, df=k-1))


def friedman_nemenyi(auc_mat):
    n_models, n_folds = auc_mat.shape
    chi2, p = scipy.stats.friedmanchisquare(*[auc_mat[i] for i in range(n_models)])
    ranks = np.zeros_like(auc_mat, float)
    for f in range(n_folds): ranks[:, f] = scipy.stats.rankdata(-auc_mat[:, f])
    avg = ranks.mean(1); se = np.sqrt(n_models*(n_models+1) / (6*n_folds))
    n_comp = n_models*(n_models-1)/2; nem = np.ones((n_models, n_models))
    for i, j in itertools.combinations(range(n_models), 2):
        p_raw = 2*scipy.stats.norm.sf(abs(avg[i] - avg[j]) / se)
        v = min(1., p_raw*n_comp); nem[i, j] = nem[j, i] = float(v)
    return float(chi2), float(p), avg, nem


def yule_q(pred1, pred2, y_true):
    c1 = (pred1 == y_true); c2 = (pred2 == y_true)
    a, b, c, d = (np.sum(c1 & c2), np.sum(c1 & ~c2),
                  np.sum(~c1 & c2), np.sum(~c1 & ~c2))
    return float((a*d - b*c) / (a*d + b*c + 1e-15))


def tost_test(v1, v2, delta=0.05):
    a = np.asarray(v1, float); b = np.asarray(v2, float); mu = a.mean() - b.mean()
    _, p1 = scipy.stats.ttest_ind(a - delta, b, equal_var=False)
    _, p2 = scipy.stats.ttest_ind(a + delta, b, equal_var=False)
    # ponytail: Welch TOST; switch to paired if groups ever become equal length
    return max(float(p1)/2, float(p2)/2), float(mu)


def permutation_group(a, b, n_perm=N_PERM):
    va = np.asarray(a, float); vb = np.asarray(b, float)
    obs = va.mean() - vb.mean(); all_ = np.concatenate([va, vb]); na = len(va)
    cnt = sum(abs(rng.permutation(all_)[:na].mean() - rng.permutation(all_)[na:].mean()) >= abs(obs)
              for _ in range(n_perm))
    return float(obs), float(cnt/n_perm)


def hanley_mcneil(auc, n_pos, n_neg):
    q1 = auc / (2 - auc); q2 = 2*auc**2 / (1 + auc)
    var = (auc*(1-auc) + (n_pos-1)*(q1 - auc**2) + (n_neg-1)*(q2 - auc**2)) / (n_pos*n_neg)
    z_a = scipy.stats.norm.ppf(1 - ALPHA/2)
    return float(scipy.stats.norm.cdf((auc - 0.5) / (np.sqrt(var) + 1e-15) - z_a))


def run_lmm(long_df):
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
        res = MixedLM.from_formula("correct ~ is_hetero", data=long_df,
                                   groups=long_df["subject_id"]).fit(reml=True, method="lbfgs")
        return {"coef": round(float(res.params.get("is_hetero", np.nan)), 4),
                "pval": round(float(res.pvalues.get("is_hetero", np.nan)), 4),
                "note": "Linear mixed model; subject=random intercept"}
    except Exception as e:
        return {"coef": np.nan, "pval": np.nan, "note": f"LMM failed: {e}"}


def _apply_mc(df, col):
    pv = df[col].values
    _, bh, _, _   = multipletests(pv, method="fdr_bh")
    _, holm, _, _ = multipletests(pv, method="holm")
    df = df.copy()
    df[f"{col}_bh"] = bh.round(4); df[f"{col}_holm"] = holm.round(4)
    df["sig_bh"] = bh < ALPHA; df["sig_holm"] = holm < ALPHA
    return df


def _fold_cal_std(pred_df):
    briers = []; eces = []
    for _, grp in pred_df.groupby("fold"):
        yt = grp["true_label"].values.astype(int)
        yp = grp["mean_prob"].values.astype(float)
        if len(np.unique(yt)) < 2: continue
        briers.append(brier_score_loss(yt, yp))
        eces.append(ece(yt, yp))
    if len(briers) < 2: return float("nan"), float("nan")
    return float(np.std(briers)), float(np.std(eces))


# ── Load saved CSV outputs ────────────────────────────────────────────────────

def load_stats_inputs():
    """
    Read model_info.csv (all model types) + per-tag prediction/result CSVs.
    Falls back to pair_info.csv + CROSS_DIR if model_info.csv is absent.
    Returns (tags, predictions, folds, pair_info).
    """
    mi_path = "/kaggle/working/model_info.csv"
    if os.path.exists(mi_path):
        mi = pd.read_csv(mi_path).set_index("tag")
    else:
        pi = pd.read_csv("/kaggle/working/pair_info.csv")
        pi["out_dir"] = CROSS_DIR
        mi = pi.set_index("tag")
    tags = mi.index.tolist(); preds = {}; folds = {}; missing = []
    for tag in tags:
        od = mi.loc[tag, "out_dir"]
        pf = f"{od}/{tag}_predictions.csv"
        rf = f"{od}/{tag}_kfold_results.csv"
        if os.path.exists(pf) and os.path.exists(rf):
            preds[tag] = pd.read_csv(pf)
            rf_df = pd.read_csv(rf)
            if "seed" in rf_df.columns:
                fold_aucs = rf_df.groupby("fold")["auc"].mean().values
            else:
                fold_aucs = rf_df["auc"].values
            folds[tag] = {"auc": fold_aucs}
        else:
            missing.append(tag)
    if missing:
        print(f"  WARNING: {len(missing)} tags missing CSVs — skipping.")
        tags = [t for t in tags if t not in missing]
    return tags, preds, folds, mi


def build_aligned(tags, preds):
    """Stack per-subject predictions into (n_models, n_subjects) arrays."""
    all_subjs = sorted(set.intersection(*[set(preds[t]["subject_id"]) for t in tags]))
    si = {s: i for i, s in enumerate(all_subjs)}
    n = len(all_subjs); m = len(tags)
    y_true = np.full(n, -1, int)
    probs   = np.full((m, n), np.nan)
    pred_arr = np.full((m, n), -1, int)
    for mi, tag in enumerate(tags):
        for _, row in preds[tag].iterrows():
            if row["subject_id"] not in si: continue
            idx = si[row["subject_id"]]
            y_true[idx]    = int(row["true_label"])
            probs[mi, idx]   = float(row["mean_prob"])
            pred_arr[mi, idx] = int(row["pred_label"])
    valid = y_true >= 0
    return y_true[valid], probs[:, valid], pred_arr[:, valid]


# ── Stats analysis sections ───────────────────────────────────────────────────

def stats_per_model(tags, y_true, probs, pred_arr, pair_info, predictions=None):
    print("  [stats 1/6] Per-model CIs and calibration...")
    n = len(y_true); n_pos = int(y_true.sum()); n_neg = n - n_pos
    rows_ci = []; rows_cal = []
    def _auc(yt, yp): return roc_auc_score(yt, yp)
    def _pra(yt, yp): return average_precision_score(yt, yp)
    def _f1(yt, yp):  return f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
    for mi, tag in enumerate(tags):
        p = probs[mi]; yp = pred_arr[mi]
        lbl = pair_info.loc[tag, "label"]; grp = pair_info.loc[tag, "group"]
        acc = float((y_true == yp).mean()); k_cor = int((y_true == yp).sum())
        av, al, ah = bootstrap_bca(y_true, p, _auc)
        pv, pl, ph = bootstrap_bca(y_true, p, _pra)
        fv, fl, fh = bootstrap_bca(y_true, yp.astype(float), _f1)
        wl, wh = wilson_ci(k_cor, n); cl, ch = clopper_pearson(k_cor, n)
        rows_ci.append({"tag": tag, "label": lbl, "group": grp,
                        "auc": round(av, 4), "auc_bca_lo": round(al, 4), "auc_bca_hi": round(ah, 4),
                        "prauc": round(pv, 4), "prauc_bca_lo": round(pl, 4), "prauc_bca_hi": round(ph, 4),
                        "f1": round(fv, 4), "f1_bca_lo": round(fl, 4), "f1_bca_hi": round(fh, 4),
                        "acc": round(acc, 4), "acc_wilson_lo": round(wl, 4), "acc_wilson_hi": round(wh, 4),
                        "acc_cp_lo": round(cl, 4), "acc_cp_hi": round(ch, 4)})
        br, rel, res, unc = brier_decomposition(y_true, p)
        ec = ece(y_true, p); sz, zp = spiegelhalter_z(y_true, p)
        pw = hanley_mcneil(av, n_pos, n_neg)
        prev = n_pos/n; brier_noskill = round(prev*(1-prev), 4)
        br_std, ec_std = float("nan"), float("nan")
        if predictions is not None and tag in predictions and "fold" in predictions[tag].columns:
            br_std, ec_std = _fold_cal_std(predictions[tag])
        rows_cal.append({"tag": tag, "label": lbl, "group": grp,
                         "brier": round(br, 4),
                         "brier_std": round(br_std, 4) if not np.isnan(br_std) else float("nan"),
                         "brier_noskill": brier_noskill, "brier_skill": round(brier_noskill - br, 4),
                         "reliability": round(rel, 4), "resolution": round(res, 4),
                         "uncertainty": round(unc, 4),
                         "ece": round(ec, 4),
                         "ece_std": round(ec_std, 4) if not np.isnan(ec_std) else float("nan"),
                         "spiegelhalter_z": round(sz, 4), "spiegelhalter_p": round(zp, 4),
                         "calibrated": zp > ALPHA, "n_pos": n_pos, "n_neg": n_neg,
                         "power": round(pw, 4)})
    pd.DataFrame(rows_ci).to_csv(f"{STATS_DIR}/per_model_ci.csv", index=False)
    pd.DataFrame(rows_cal).to_csv(f"{STATS_DIR}/per_model_calibration.csv", index=False)
    return pd.DataFrame(rows_ci), pd.DataFrame(rows_cal)


def stats_pairwise(tags, y_true, probs, pred_arr, folds, pair_info):
    print("  [stats 2/6] Pairwise tests...")
    pairs = list(itertools.combinations(range(len(tags)), 2))
    dl = []; mc = []; da = []; pa = []; nb = []; db = []; div = []
    def _auc(yt, yp): return roc_auc_score(yt, yp)
    def _acc(yt, yp): return float((yt == (yp >= 0.5).astype(int)).mean())
    def _br(yt, yp):  return float(brier_score_loss(yt, yp))
    for ci, (i, j) in enumerate(pairs):
        if ci % 150 == 0: print(f"    {ci}/{len(pairs)}...")
        t1 = tags[i]; t2 = tags[j]
        p1 = probs[i]; p2 = probs[j]
        d1 = pred_arr[i]; d2 = pred_arr[j]
        l1 = pair_info.loc[t1, "label"]; l2 = pair_info.loc[t2, "label"]
        z, pv, a1, a2 = delong_test(y_true, p1, p2)
        dl.append({"model_a": t1, "label_a": l1, "model_b": t2, "label_b": l2,
                   "delta_auc": round(a1 - a2, 4), "z": round(z, 4), "p_raw": round(pv, 4)})
        pe, pm, b, c = mcnemar_test(y_true, d1, d2)
        mc.append({"model_a": t1, "label_a": l1, "model_b": t2, "label_b": l2,
                   "b": b, "c": c, "p_exact": round(pe, 4), "p_midp": round(pm, 4)})
        obs, lo, hi = bootstrap_ci_pair(y_true, p1, p2, _auc)
        da.append({"model_a": t1, "model_b": t2, "delta_auc": round(obs, 4),
                   "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "sig_95": (lo > 0 or hi < 0)})
        od, pp = permutation_group(p1, p2, n_perm=N_PERM_P)
        pa.append({"model_a": t1, "model_b": t2, "delta_auc": round(od, 4), "p_perm": round(pp, 4)})
        t_nb, p_nb = nadeau_bengio(folds[t1]["auc"].tolist(), folds[t2]["auc"].tolist())
        nb.append({"model_a": t1, "label_a": l1, "model_b": t2, "label_b": l2,
                   "t": round(float(t_nb), 4) if not np.isnan(t_nb) else np.nan,
                   "p_raw": round(p_nb, 4)})
        oa, la_, ha_ = bootstrap_ci_pair(y_true, p1, p2, _acc)
        ob, lb_, hb_ = bootstrap_ci_pair(y_true, p1, p2, _br)
        db.append({"model_a": t1, "model_b": t2,
                   "delta_acc": round(oa, 4), "acc_ci_lo": round(la_, 4), "acc_ci_hi": round(ha_, 4),
                   "delta_brier": round(ob, 4), "brier_ci_lo": round(lb_, 4), "brier_ci_hi": round(hb_, 4)})
        kap = float(cohen_kappa_score(d1, d2)); yq = yule_q(d1, d2, y_true)
        dis = float(np.mean(d1 != d2)); df_ = float(np.mean((d1 != y_true) & (d2 != y_true)))
        div.append({"model_a": t1, "label_a": l1, "model_b": t2, "label_b": l2,
                    "kappa": round(kap, 4), "yule_q": round(yq, 4),
                    "disagreement": round(dis, 4), "double_fault": round(df_, 4)})
    df_dl = _apply_mc(pd.DataFrame(dl), "p_raw")
    df_mc = _apply_mc(pd.DataFrame(mc), "p_exact"); df_mc = _apply_mc(df_mc, "p_midp")
    df_nb = _apply_mc(pd.DataFrame(nb), "p_raw")
    df_dl.to_csv(f"{STATS_DIR}/pairwise_delong.csv", index=False)
    df_mc.to_csv(f"{STATS_DIR}/pairwise_mcnemar.csv", index=False)
    pd.DataFrame(da).to_csv(f"{STATS_DIR}/pairwise_delta_auc_boot.csv", index=False)
    pd.DataFrame(pa).to_csv(f"{STATS_DIR}/pairwise_perm_delta_auc.csv", index=False)
    df_nb.to_csv(f"{STATS_DIR}/pairwise_nb_ttest.csv", index=False)
    pd.DataFrame(db).to_csv(f"{STATS_DIR}/pairwise_delta_acc_brier.csv", index=False)
    pd.DataFrame(div).to_csv(f"{STATS_DIR}/pairwise_diversity.csv", index=False)
    print(f"    DeLong sig (BH): {df_dl['sig_bh'].sum()}/{len(df_dl)}  "
          f"McNemar sig (BH): {df_mc['sig_bh'].sum()}/{len(df_mc)}")
    return df_dl, df_mc, pd.DataFrame(div), df_nb


def stats_omnibus(tags, y_true, pred_arr, folds, pair_info):
    print("  [stats 3/6] Omnibus tests...")
    correct = np.stack([(pred_arr[i] == y_true).astype(int) for i in range(len(tags))], axis=1)
    Q, qp = cochran_q(correct)
    auc_mat = np.array([np.asarray(folds[t]["auc"]) for t in tags])
    chi2, fp, avg_ranks, nem_p = friedman_nemenyi(auc_mat)
    print(f"    Cochran Q={Q:.3f} p={qp:.4f}  |  Friedman χ²={chi2:.3f} p={fp:.4f}")
    pd.DataFrame([{"test": "Cochran's Q", "statistic": round(Q, 4), "df": len(tags)-1, "p_value": round(qp, 4)},
                  {"test": "Friedman", "statistic": round(chi2, 4), "df": len(tags)-1, "p_value": round(fp, 4)}
                  ]).to_csv(f"{STATS_DIR}/omnibus_tests.csv", index=False)
    labels = [pair_info.loc[t, "label"] for t in tags]
    nem_rows = [{"model_a": tags[i], "label_a": labels[i], "model_b": tags[j], "label_b": labels[j],
                 "rank_a": round(float(avg_ranks[i]), 3), "rank_b": round(float(avg_ranks[j]), 3),
                 "p_nemenyi": round(float(nem_p[i, j]), 4), "sig_005": float(nem_p[i, j]) < ALPHA}
                for i, j in itertools.combinations(range(len(tags)), 2)]
    pd.DataFrame(nem_rows).to_csv(f"{STATS_DIR}/nemenyi_pvalues.csv", index=False)
    return avg_ranks, nem_p


def stats_group(tags, y_true, probs, pred_arr, folds, pair_info, predictions):
    print("  [stats 4/6] Group comparison (6 groups)...")
    ALL_GROUPS = ["hetero", "homo", "same_mri", "same_pet", "solo_mri", "solo_pet"]
    group_aucs      = {g: [] for g in ALL_GROUPS}
    group_fold_aucs = {g: [] for g in ALL_GROUPS}
    for tag in tags:
        g = pair_info.loc[tag, "group"]
        if g not in group_aucs: continue
        group_aucs[g].append(float(folds[tag]["auc"].mean()))
        group_fold_aucs[g].extend(folds[tag]["auc"].tolist())

    nonempty = [(g, v) for g, v in group_aucs.items() if len(v) >= 2]
    rows = []

    if len(nonempty) >= 2:
        H, kp = scipy.stats.kruskal(*[v for _, v in nonempty])
        rows.append({"test": "Kruskal-Wallis (6 groups)", "statistic": round(float(H), 4),
                     "p_value": round(float(kp), 4),
                     "note": f"Groups: {[g for g, _ in nonempty]}"})

    mw_rows = []
    for (g1, v1), (g2, v2) in itertools.combinations(nonempty, 2):
        u, up = scipy.stats.mannwhitneyu(v1, v2, alternative="two-sided")
        mw_rows.append({"group_a": g1, "group_b": g2, "n_a": len(v1), "n_b": len(v2),
                        "U": round(float(u), 4), "p_raw": round(float(up), 4)})
    if mw_rows:
        df_mw = pd.DataFrame(mw_rows)
        _, bh, _, _ = multipletests(df_mw["p_raw"].values, method="fdr_bh")
        df_mw["p_bh"] = bh.round(4); df_mw["sig_bh"] = bh < ALPHA
        df_mw.to_csv(f"{STATS_DIR}/group_pairwise_mw.csv", index=False)
        rows.append({"test": "Pairwise Mann-Whitney (all group pairs)",
                     "statistic": float(len(df_mw)), "p_value": float(df_mw["p_bh"].min()),
                     "note": f"Saved group_pairwise_mw.csv; min BH p={df_mw['p_bh'].min():.4f}"})

    h_fold = group_fold_aucs["hetero"]; o_fold = group_fold_aucs["homo"]
    if h_fold and o_fold:
        tp, tmu = tost_test(h_fold, o_fold)
        rows.append({"test": "TOST hetero vs homo (delta=0.05)",
                     "statistic": round(tmu, 4), "p_value": round(tp, 4),
                     "note": "p<0.05 → equivalence rejected"})

    if group_aucs["hetero"] and group_aucs["homo"]:
        od, pp = permutation_group(group_aucs["hetero"], group_aucs["homo"], n_perm=N_PERM)
        rows.append({"test": "Permutation hetero vs homo",
                     "statistic": round(od, 4), "p_value": round(pp, 4),
                     "note": f"μ_hetero-μ_homo={od:.4f}, {N_PERM} permutations"})

    has_mri_arch = "mri_arch" in pair_info.columns
    has_pet_arch = "pet_arch" in pair_info.columns
    homo_auc_by_arch = {}
    if has_mri_arch:
        for tag in tags:
            if pair_info.loc[tag, "group"] == "homo":
                ma = pair_info.loc[tag, "mri_arch"]
                homo_auc_by_arch[ma] = float(folds[tag]["auc"].mean())
    paired_diffs = []
    if has_mri_arch and has_pet_arch:
        for tag in tags:
            if pair_info.loc[tag, "group"] != "hetero": continue
            ma = pair_info.loc[tag, "mri_arch"]; pa = pair_info.loc[tag, "pet_arch"]
            if ma in homo_auc_by_arch and pa in homo_auc_by_arch:
                paired_diffs.append(
                    float(folds[tag]["auc"].mean()) - (homo_auc_by_arch[ma] + homo_auc_by_arch[pa]) / 2)
    if len(paired_diffs) >= 5:
        ws, wp = scipy.stats.wilcoxon(paired_diffs, alternative="two-sided")
        rows.append({"test": "Wilcoxon signed-rank (hetero vs paired homo)",
                     "statistic": round(float(ws), 4), "p_value": round(float(wp), 4),
                     "note": f"d_i=AUC_hetero(a,b)-0.5*(homo_a+homo_b), n={len(paired_diffs)}"})

    if group_aucs["hetero"] and group_aucs["homo"]:
        gm = np.mean(group_aucs["homo"]); h_arr = np.array(group_aucs["hetero"])
        npos = int(np.sum(h_arr > gm)); n = len(h_arr)
        bp = float(2 * min(scipy.stats.binom.cdf(npos, n, 0.5),
                           scipy.stats.binom.sf(npos - 1, n, 0.5)))
        rows.append({"test": "Sign test (hetero > grand homo mean)",
                     "statistic": float(npos), "p_value": round(bp, 4),
                     "note": f"{npos}/{n} hetero > grand homo mean ({gm:.4f})"})

    lmm_rows = []
    for tag in tags:
        is_hetero = 1 if pair_info.loc[tag, "group"] == "hetero" else 0
        df_ = predictions[tag][["subject_id", "pred_label", "true_label"]].copy()
        df_["correct"] = (df_["pred_label"] == df_["true_label"]).astype(int)
        df_["is_hetero"] = is_hetero
        lmm_rows.append(df_)
    lr = run_lmm(pd.concat(lmm_rows, ignore_index=True))
    rows.append({"test": "Linear mixed model (GLMM approx.)",
                 "statistic": lr["coef"], "p_value": lr["pval"], "note": lr["note"]})

    df_grp = pd.DataFrame(rows)
    df_grp.to_csv(f"{STATS_DIR}/group_comparison.csv", index=False)
    for _, row in df_grp.iterrows():
        print(f"    {row['test']:<60} p={row['p_value']:.4f}")
    return df_grp, paired_diffs


def stats_diversity_gain(tags, probs, y_true, folds, pair_info):
    print("  [stats 5/6] Diversity vs AUC gain...")
    homo_auc = {pair_info.loc[t, "mri_arch"]: float(folds[t]["auc"].mean())
                for t in tags if pair_info.loc[t, "group"] == "homo"}
    homo_mi  = {pair_info.loc[t, "mri_arch"]: i for i, t in enumerate(tags)
                if pair_info.loc[t, "group"] == "homo"}
    rows = []
    for i, tag in enumerate(tags):
        if pair_info.loc[tag, "group"] != "hetero": continue
        ma = pair_info.loc[tag, "mri_arch"]; pa = pair_info.loc[tag, "pet_arch"]
        if ma not in homo_auc or pa not in homo_auc: continue
        gain = float(folds[tag]["auc"].mean()) - max(homo_auc[ma], homo_auc[pa])
        mii = homo_mi.get(ma); pii = homo_mi.get(pa)
        if mii is not None and pii is not None:
            dis = float(np.mean(np.abs(probs[mii] - probs[pii])))
            df_ = float(np.mean(((probs[mii] >= 0.5).astype(int) != y_true) &
                                ((probs[pii] >= 0.5).astype(int) != y_true)))
        else: dis = df_ = np.nan
        rows.append({"tag": tag, "label": pair_info.loc[tag, "label"],
                     "auc_gain": round(gain, 4),
                     "disagreement": round(dis, 4) if not np.isnan(dis) else np.nan,
                     "double_fault": round(df_, 4) if not np.isnan(df_) else np.nan})
    df_dg = pd.DataFrame(rows).dropna(subset=["disagreement"])
    if len(df_dg) >= 3:
        r1, p1 = scipy.stats.spearmanr(df_dg["disagreement"], df_dg["auc_gain"])
        r2, p2 = scipy.stats.spearmanr(df_dg["double_fault"], df_dg["auc_gain"])
        df_dg["spearman_dis_r"] = round(float(r1), 4); df_dg["spearman_dis_p"] = round(float(p1), 4)
        df_dg["spearman_df_r"]  = round(float(r2), 4); df_dg["spearman_df_p"]  = round(float(p2), 4)
        print(f"    Disagreement vs AUC gain ρ={r1:.3f} p={p1:.4f}")
    df_dg.to_csv(f"{STATS_DIR}/diversity_vs_gain.csv", index=False)
    return df_dg


def stats_ablation(all_tags, predictions, pair_info):
    """DeLong + McNemar: MLP vs avg baseline and MLP vs LR stacking, per fusion pair."""
    from sklearn.metrics import roc_auc_score
    print("  [ablation] MLP vs Avg vs LR stacking (DeLong + McNemar per pair)...")
    rows = []
    fusion_tags = [t for t in all_tags
                   if pair_info.loc[t, "group"] not in ("avg", "lr_stack")
                   and "solo" not in str(pair_info.loc[t, "group"])]
    for tag in fusion_tags:
        if tag not in predictions: continue
        base_df = predictions[tag]
        subjs   = sorted(base_df["subject_id"])
        si      = {s: i for i, s in enumerate(subjs)}
        y     = np.array([base_df.set_index("subject_id").loc[s, "true_label"] for s in subjs])
        p_mlp = np.array([base_df.set_index("subject_id").loc[s, "mean_prob"]  for s in subjs])

        for suffix, kind in [("_avg", "avg"), ("_lr", "lr_stack")]:
            comp_tag = tag + suffix
            if comp_tag not in predictions: continue
            comp_df = predictions[comp_tag]
            common  = sorted(set(subjs) & set(comp_df["subject_id"]))
            if len(common) < 10: continue
            si2 = {s: i for i, s in enumerate(common)}
            yt  = np.array([y[si[s]]    for s in common])
            pm  = np.array([p_mlp[si[s]] for s in common])
            comp_df2 = comp_df.set_index("subject_id")
            pc  = np.array([float(comp_df2.loc[s, "mean_prob"]) for s in common])
            d1  = (pm >= 0.5).astype(int); d2 = (pc >= 0.5).astype(int)
            try:
                auc_mlp  = roc_auc_score(yt, pm); auc_comp = roc_auc_score(yt, pc)
                z, p_dl, _, _ = delong_test(yt, pm, pc)
                p_mc, p_mcp, b, c = mcnemar_test(yt, d1, d2)
            except Exception: continue
            rows.append({"mlp_tag": tag, "comp_tag": comp_tag, "baseline": kind,
                         "mlp_label": pair_info.loc[tag, "label"],
                         "auc_mlp": round(auc_mlp, 4), "auc_comp": round(auc_comp, 4),
                         "delta_auc": round(auc_mlp - auc_comp, 4),
                         "delong_z": round(z, 4), "delong_p": round(p_dl, 4),
                         "mcnemar_p_exact": round(p_mc, 4), "mcnemar_p_midp": round(p_mcp, 4),
                         "b": b, "c": c})
    if rows:
        df = pd.DataFrame(rows)
        _, bh, _, _   = multipletests(df["delong_p"].values, method="fdr_bh")
        _, holm, _, _ = multipletests(df["delong_p"].values, method="holm")
        df["delong_p_bh"] = bh.round(4); df["delong_p_holm"] = holm.round(4)
        df.to_csv(f"{STATS_DIR}/ablation_delong_mcnemar.csv", index=False)
        print(f"    Saved ablation_delong_mcnemar.csv  ({len(df)} rows)")
    return rows
