"""
plots.py — All visualization functions.

Training-phase:  plot_roc_folds, plot_confusion, plot_fold_auc,
                 plot_roc_grid_by_mri, plot_auc_by_mri, plot_auc_heatmap_6x6,
                 plot_hetero_vs_homo, plot_summary_all

Post-training:   plot_seed_stability, plot_ablation_comparison,
                 plot_solo_vs_fusion, plot_group_summary

Stats:           plot_calibration_grid, plot_pairwise_hm, plot_cd_diagram,
                 plot_group_violin, plot_diversity_scatter, plot_power_bars
"""
import os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import scipy.stats

from .config import (
    ARCHS, ARCH_DISPLAY, COLORS, PAIR_LABELS, ALL_LABELS, ALL_PAIRS,
    HETERO_PAIRS, HOMO_PAIRS,
    K_FOLDS, OUT_DIR, STATS_DIR, ALPHA, rng,
)


# ── Training-phase plots ──────────────────────────────────────────────────────

def plot_roc_folds(fold_results, tag, color, out_dir=None):
    if out_dir is None: out_dir = OUT_DIR
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#ccc", lw=1)
    fpr_grid = np.linspace(0, 1, 300); tprs = []
    for r in fold_results:
        fp, tp, _ = roc_curve(r["labels"], r["probs"])
        ax.plot(fp, tp, color=color, alpha=0.35, lw=1.2)
        tprs.append(np.interp(fpr_grid, fp, tp))
    mt = np.mean(tprs, axis=0); st = np.std(tprs, axis=0)
    ma = np.mean([r["auc"] for r in fold_results]); sa = np.std([r["auc"] for r in fold_results])
    ax.plot(fpr_grid, mt, color=color, lw=2.5, label=f"Mean AUC = {ma:.3f} ± {sa:.3f}")
    ax.fill_between(fpr_grid, mt - st, mt + st, color=color, alpha=0.15)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(PAIR_LABELS[tag], fontsize=11)
    ax.legend(fontsize=10, loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/roc_{tag}.png", bbox_inches="tight"); plt.close()


def plot_confusion(fold_results, tag, out_dir=None):
    from sklearn.metrics import confusion_matrix
    if out_dir is None: out_dir = OUT_DIR
    agg = sum(confusion_matrix(r["labels"], (r["probs"] >= 0.5).astype(int))
              for r in fold_results)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(agg, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["CN", "AD"], yticklabels=["CN", "AD"], annot_kws={"size": 14})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(PAIR_LABELS[tag], fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/cm_{tag}.png", bbox_inches="tight"); plt.close()


def plot_fold_auc(fold_results, tag, color, out_dir=None):
    if out_dir is None: out_dir = OUT_DIR
    aucs = [r["auc"] for r in fold_results]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(range(1, K_FOLDS+1), aucs, color=color, alpha=0.75, edgecolor="white")
    ax.axhline(np.mean(aucs), color="#333", lw=1.5, ls="--", label=f"Mean={np.mean(aucs):.3f}")
    for i, v in enumerate(aucs):
        ax.text(i+1, v+0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(1, K_FOLDS+1))
    ax.set_xticklabels([f"Fold {k}" for k in range(1, K_FOLDS+1)])
    ax.set_ylabel("AUC"); ax.set_ylim(max(0.5, min(aucs)-0.06), 1.0)
    ax.set_title(PAIR_LABELS[tag], fontsize=11); ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fold_auc_{tag}.png", bbox_inches="tight"); plt.close()


def plot_roc_grid_by_mri(all_results, mri_arch):
    tags = [f"{mri_arch}_{pa}" for pa in ARCHS if pa != mri_arch and f"{mri_arch}_{pa}" in all_results]
    if not tags: return
    fig, axes = plt.subplots(1, len(tags), figsize=(len(tags)*5, 4.5), squeeze=False)
    fpr_grid = np.linspace(0, 1, 300)
    for idx, (tag, ax) in enumerate(zip(tags, axes[0])):
        color = COLORS[idx % len(COLORS)]
        ax.plot([0, 1], [0, 1], "--", color="#ccc", lw=1); tprs = []
        for r in all_results[tag]:
            fp, tp, _ = roc_curve(r["labels"], r["probs"])
            ax.plot(fp, tp, color=color, alpha=0.3, lw=1)
            tprs.append(np.interp(fpr_grid, fp, tp))
        mt = np.mean(tprs, axis=0)
        ax.plot(fpr_grid, mt, color=color, lw=2,
                label=f"AUC={np.mean([r['auc'] for r in all_results[tag]]):.3f}")
        ax.fill_between(fpr_grid, mt-np.std(tprs, axis=0), mt+np.std(tprs, axis=0),
                        color=color, alpha=0.12)
        ax.set_title(PAIR_LABELS[tag], fontsize=10); ax.legend(fontsize=9, loc="lower right")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.suptitle(f"{ARCH_DISPLAY[mri_arch]} MRI — Hetero PET Partners", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/roc_grid_{mri_arch}.png", bbox_inches="tight"); plt.close()


def plot_auc_by_mri(all_results, mri_arch):
    tags = [f"{mri_arch}_{pa}" for pa in ARCHS if pa != mri_arch and f"{mri_arch}_{pa}" in all_results]
    if not tags: return
    pet_labels = [f"{ARCH_DISPLAY[t[len(mri_arch)+1:]]} PET" for t in tags]
    means = [np.mean([r["auc"] for r in all_results[t]]) for t in tags]
    stds  = [np.std( [r["auc"] for r in all_results[t]]) for t in tags]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(tags)), means, yerr=stds, capsize=5, color=COLORS[:len(tags)],
           alpha=0.85, edgecolor="white", error_kw={"lw": 1.5, "ecolor": "#333"})
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m+s+0.006, f"{m:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(pet_labels, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel("AUC (mean ± std, 5-fold CV)")
    ax.set_title(f"{ARCH_DISPLAY[mri_arch]} MRI — AUC by PET Backbone")
    ax.set_ylim(max(0.5, min(m-s for m, s in zip(means, stds))-0.06), 1.05)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/summary_auc_{mri_arch}.png", bbox_inches="tight"); plt.close()


def plot_auc_heatmap_6x6(all_results):
    mat = np.full((len(ARCHS), len(ARCHS)), np.nan)
    for i, ma in enumerate(ARCHS):
        for j, pa in enumerate(ARCHS):
            tag = f"{ma}_homo" if ma == pa else f"{ma}_{pa}"
            if tag in all_results:
                mat[i, j] = np.mean([r["auc"] for r in all_results[tag]])
    labels = [ARCH_DISPLAY[a] for a in ARCHS]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(np.where(np.isnan(mat), 0., mat), annot=False, cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=0.70, vmax=1.0, linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.75, "label": "Mean AUC (5-fold)"})
    for i in range(len(ARCHS)):
        for j in range(len(ARCHS)):
            if np.isnan(mat[i, j]):
                ax.text(j+0.5, i+0.5, "N/A", ha="center", va="center", fontsize=11, color="#bbb")
            else:
                ax.text(j+0.5, i+0.5, f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=11, color="white" if mat[i, j] > 0.88 else "#111",
                        fontweight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="#222", lw=2.5))
    ax.set_xlabel("PET Backbone", fontsize=12, labelpad=8)
    ax.set_ylabel("MRI Backbone", fontsize=12, labelpad=8)
    ax.set_title("Mean AUC (5-fold CV)\nOff-diagonal=Hetero | Diagonal(bold)=Homo",
                 fontsize=12, pad=12)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(labels, rotation=0, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/heatmap_auc_6x6.png", bbox_inches="tight")
    plt.close(); print("  Saved: heatmap_auc_6x6.png")


def plot_hetero_vs_homo(all_results):
    h = [np.mean([r["auc"] for r in all_results[p["tag"]]]) for p in HETERO_PAIRS if p["tag"] in all_results]
    o = [np.mean([r["auc"] for r in all_results[p["tag"]]]) for p in HOMO_PAIRS   if p["tag"] in all_results]
    if not h or not o: return
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (vals, col) in enumerate(zip([h, o], ["#4C8EDA", "#E8622A"])):
        vp = ax.violinplot([vals], positions=[i], showmeans=True)
        for body in vp["bodies"]: body.set_facecolor(col); body.set_alpha(0.5)
        for part in ("cbars","cmins","cmaxes","cmeans"): vp[part].set_color("#222"); vp[part].set_lw(1.8)
        jit = rng.uniform(-0.07, 0.07, len(vals))
        ax.scatter(np.full(len(vals), i)+jit, vals, s=50, color=col, edgecolors="white", lw=0.7, zorder=5)
        ax.text(i, max(vals)+0.012, f"μ={np.mean(vals):.3f}", ha="center", fontsize=11,
                fontweight="bold", color=col)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Hetero\n(n={len(h)})", f"Homo\n(n={len(o)})"], fontsize=12)
    ax.set_ylabel("Mean AUC (5-fold CV)", fontsize=12)
    ax.set_title("Heterogeneous vs Homogeneous Fusion", fontsize=13)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/hetero_vs_homo.png", bbox_inches="tight")
    plt.close(); print("  Saved: hetero_vs_homo.png")


def plot_summary_all(all_results):
    tags   = [p["tag"] for p in ALL_PAIRS if p["tag"] in all_results]
    labels = [PAIR_LABELS[t] for t in tags]
    metrics = ["auc","acc","sens","spec","f1"]
    bar_colors = ["#4C8EDA","#E8622A","#2CA02C","#9467BD","#8C564B"]
    fig, ax = plt.subplots(figsize=(24, 5))
    means = [np.mean([r["auc"] for r in all_results[t]]) for t in tags]
    stds  = [np.std( [r["auc"] for r in all_results[t]]) for t in tags]
    ax.bar(range(len(tags)), means, yerr=stds, capsize=4,
           color=[COLORS[i % len(COLORS)] for i in range(len(tags))],
           alpha=0.85, edgecolor="white", error_kw={"lw": 1.2, "ecolor": "#333"})
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m+s+0.005, f"{m:.3f}", ha="center", fontsize=7.5, fontweight="bold", rotation=90)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUC"); ax.set_title("All 36 Fusion Pairs — AUC Summary")
    ax.set_ylim(0.65, 1.1); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/summary_auc_all.png", bbox_inches="tight"); plt.close()
    x = np.arange(len(tags)); w = 0.13; n_m = len(metrics)
    fig, ax = plt.subplots(figsize=(26, 5.5))
    for mi, metric in enumerate(metrics):
        mm = [np.mean([r[metric] for r in all_results[t]]) for t in tags]
        ms = [np.std( [r[metric] for r in all_results[t]]) for t in tags]
        ax.bar(x+(mi-n_m/2+0.5)*w, mm, yerr=ms, width=w, capsize=2,
               label=metric.upper(), color=bar_colors[mi], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score"); ax.set_ylim(0.5, 1.1); ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True); ax.legend(fontsize=9, ncol=n_m)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/summary_metrics_all.png", bbox_inches="tight"); plt.close()
    mn = ["auc","acc","sens","spec","f1","pr_auc","bal_acc"]
    tbl = np.array([[np.mean([r[m] for r in all_results[t]]) for m in mn] for t in tags])
    fig, ax = plt.subplots(figsize=(14, 18))
    sns.heatmap(tbl, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=[m.upper() for m in mn], yticklabels=labels, ax=ax,
                vmin=0.6, vmax=1.0, annot_kws={"size": 8.5}, cbar_kws={"shrink": 0.5})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title("All 36 Fusion Pairs — Metric Heatmap", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/heatmap_metrics_all.png", bbox_inches="tight"); plt.close()
    print("  Saved: summary_auc_all.png, summary_metrics_all.png, heatmap_metrics_all.png")


# ── Post-training comparison plots ───────────────────────────────────────────

def _load_seed_means(pairs, dirs=None):
    rows = []
    for p in pairs:
        f = f"{p['out_dir']}/{p['tag']}_seed_means.csv"
        if not os.path.exists(f): continue
        df = pd.read_csv(f); df["tag"] = p["tag"]; df["label"] = p["label"]
        df["group"] = p["group"]; rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_seed_stability(all_fusion_pairs):
    df = _load_seed_means(all_fusion_pairs)
    if df.empty: return
    order = df.groupby("tag")["auc_mean"].mean().sort_values(ascending=False).index.tolist()
    labels_ordered = [ALL_LABELS.get(t, t) for t in order]
    fig, ax = plt.subplots(figsize=(max(14, len(order)*0.45), 5))
    data = [df[df["tag"] == t]["auc_mean"].values for t in order]
    vp = ax.violinplot(data, positions=range(len(order)), showmeans=True, widths=0.7)
    grp_colors = {"hetero": "#4C8EDA", "homo": "#E8622A",
                  "same_mri": "#2CA02C", "same_pet": "#9467BD"}
    for body, t in zip(vp["bodies"], order):
        grp = df[df["tag"] == t]["group"].iloc[0]
        body.set_facecolor(grp_colors.get(grp, "#888")); body.set_alpha(0.55)
    for part in ("cbars","cmins","cmaxes","cmeans"):
        vp[part].set_color("#333"); vp[part].set_lw(1.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels_ordered, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("5-fold mean AUC (per seed)"); ax.set_ylim(0.3, 1.05)
    ax.set_title("Seed Stability — 5 Seeds × 5-Fold AUC\n"
                 "Blue=Hetero · Orange=Homo · Green=MRI×MRI · Purple=PET×PET")
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{STATS_DIR}/seed_stability.png", bbox_inches="tight"); plt.close()
    print("  Saved: seed_stability.png")


def plot_ablation_comparison(all_fusion_pairs, cross_res, same_res, solo_res, lr_res, avg_res):
    rows = []
    for p in all_fusion_pairs:
        tag = p["tag"]
        for result_dict, kind in [(cross_res if p["group"] in ("hetero","homo") else same_res, "MLP"),
                                   (lr_res, "LR-Stack"), (avg_res, "Avg")]:
            key = tag if kind == "MLP" else tag + ("_lr" if kind == "LR-Stack" else "_avg")
            if key not in result_dict: continue
            fr = result_dict[key]
            auc_v = float(np.mean([r["auc"] for r in fr]))
            rows.append({"label": p["label"], "group": p["group"], "kind": kind, "auc": auc_v})
    if not rows: return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(18, len(all_fusion_pairs)*0.8), 5))
    width = 0.27; kinds = ["MLP", "LR-Stack", "Avg"]
    kind_colors = {"MLP": "#4C8EDA", "LR-Stack": "#E8622A", "Avg": "#2CA02C"}
    pair_labels_ord = df["label"].unique().tolist()
    x = np.arange(len(pair_labels_ord))
    for ki, kind in enumerate(kinds):
        sub = df[df["kind"] == kind].set_index("label")
        vals = [float(sub.loc[lbl, "auc"]) if lbl in sub.index else np.nan
                for lbl in pair_labels_ord]
        ax.bar(x+(ki-1)*width, vals, width, label=kind,
               color=kind_colors[kind], alpha=0.8, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(pair_labels_ord, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean AUC (5-fold)"); ax.set_ylim(0.4, 1.08)
    ax.set_title("Ablation: MLP Fusion vs LR Stacking vs Probability Averaging")
    ax.legend(fontsize=10); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{STATS_DIR}/ablation_comparison.png", bbox_inches="tight"); plt.close()
    df.to_csv(f"{STATS_DIR}/ablation_summary.csv", index=False)
    print("  Saved: ablation_comparison.png, ablation_summary.csv")


def plot_solo_vs_fusion(archs, solo_res, cross_res):
    rows = []
    for arch in archs:
        mri_tag = f"{arch}_mri_solo"; pet_tag = f"{arch}_pet_solo"
        mri_auc = float(np.mean([r["auc"] for r in solo_res[mri_tag]])) if mri_tag in solo_res else np.nan
        pet_auc = float(np.mean([r["auc"] for r in solo_res[pet_tag]])) if pet_tag in solo_res else np.nan
        best_auc = max(
            (float(np.mean([r["auc"] for r in cross_res[f"{arch}_{pa}"]]))
             for pa in archs if pa != arch and f"{arch}_{pa}" in cross_res),
            default=np.nan)
        rows.append({"arch": ARCH_DISPLAY[arch],
                     "MRI Solo": mri_auc, "PET Solo": pet_auc,
                     "Best Fusion (as MRI)": best_auc})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(archs)); width = 0.25
    kind_colors = {"MRI Solo": "#4C8EDA", "PET Solo": "#E8622A", "Best Fusion (as MRI)": "#2CA02C"}
    for ki, kind in enumerate(["MRI Solo", "PET Solo", "Best Fusion (as MRI)"]):
        vals = df[kind].values
        ax.bar(x+(ki-1)*width, vals, width, label=kind, color=kind_colors[kind],
               alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["arch"], fontsize=11)
    ax.set_ylabel("Mean AUC (5-fold)"); ax.set_ylim(0.5, 1.08)
    ax.set_title("Solo Backbone vs Best Fusion (MRI as first branch)")
    ax.legend(fontsize=10); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{STATS_DIR}/solo_vs_fusion.png", bbox_inches="tight"); plt.close()


def plot_group_summary(cross_pairs, same_pairs, solo_models, cross_res, same_res, solo_res):
    groups_data = {}
    for p in cross_pairs:
        if p["tag"] not in cross_res: continue
        g = p["group"]
        groups_data.setdefault(g, []).append(float(np.mean([r["auc"] for r in cross_res[p["tag"]]])))
    for p in same_pairs:
        if p["tag"] not in same_res: continue
        g = p["group"]
        groups_data.setdefault(g, []).append(float(np.mean([r["auc"] for r in same_res[p["tag"]]])))
    for s in solo_models:
        if s["tag"] not in solo_res: continue
        g = s["group"]
        groups_data.setdefault(g, []).append(float(np.mean([r["auc"] for r in solo_res[s["tag"]]])))
    if not groups_data: return
    grp_colors = {"hetero": "#4C8EDA", "homo": "#E8622A",
                  "same_mri": "#2CA02C", "same_pet": "#9467BD",
                  "solo_mri": "#E69F00", "solo_pet": "#CC79A7"}
    groups = list(groups_data.keys())
    labels_used = [g.replace("_", " ").title() for g in groups]
    counts = [len(groups_data[g]) for g in groups]
    fig, ax = plt.subplots(figsize=(max(10, len(groups)*1.5), 5))
    for i, g in enumerate(groups):
        vals = groups_data[g]; col = grp_colors.get(g, "#888")
        vp = ax.violinplot([vals], positions=[i], showmeans=True)
        for body in vp["bodies"]: body.set_facecolor(col); body.set_alpha(0.5)
        for part in ("cbars","cmins","cmaxes","cmeans"): vp[part].set_color("#222"); vp[part].set_lw(1.5)
        jit = rng.uniform(-0.07, 0.07, len(vals))
        ax.scatter(np.full(len(vals), i)+jit, vals, s=40,
                   color=grp_colors.get(g, "#888"), edgecolors="white", lw=0.5, zorder=5)
        ax.text(i, max(vals)+0.015, f"μ={np.mean(vals):.3f}\nn={counts[i]}",
                ha="center", fontsize=9, color=grp_colors.get(g, "#888"))
    ax.set_xticks(range(len(labels_used)))
    ax.set_xticklabels([f"{g}\n(n={c})" for g, c in zip(labels_used, counts)], fontsize=10)
    ax.set_ylabel("Mean AUC (5-fold CV)"); ax.set_ylim(0.3, 1.1)
    ax.set_title("AUC by Experimental Group")
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{STATS_DIR}/group_summary.png", bbox_inches="tight"); plt.close()
    print("  Saved: group_summary.png")


# ── Stats plots ───────────────────────────────────────────────────────────────

def plot_calibration_grid(tags, y_true, probs, pair_info):
    nc = 6; nr = (len(tags) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(nc*3.5, nr*3.5), squeeze=False)
    bins = np.linspace(0, 1, 11)
    for mi, tag in enumerate(tags):
        ax = axes[mi//nc][mi%nc]; p = probs[mi]
        col = "#4C8EDA" if pair_info.loc[tag, "group"] == "hetero" else "#E8622A"
        bm, bf = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0: continue
            bm.append(p[mask].mean()); bf.append(y_true[mask].mean())
        ax.plot([0, 1], [0, 1], "--", color="#ccc", lw=1)
        ax.plot(bm, bf, "o-", color=col, lw=1.5, ms=5)
        ax.fill_between(bm, bm, bf, alpha=0.12, color=col)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_title(pair_info.loc[tag, "label"], fontsize=8, pad=3)
        ax.set_xlabel("Predicted", fontsize=7); ax.set_ylabel("Observed", fontsize=7)
        ax.tick_params(labelsize=7)
    for k in range(len(tags), nr*nc): axes[k//nc][k%nc].set_visible(False)
    fig.suptitle("Calibration Reliability Diagrams\nBlue=Hetero · Orange=Homo", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{STATS_DIR}/calibration_grid.png", bbox_inches="tight"); plt.close()


def plot_pairwise_hm(tags, pair_df, col, title, fname, pair_info, cmap="RdYlGn",
                     vmin=0, vmax=3, log_scale=False, center=None):
    n = len(tags); ti = {t: i for i, t in enumerate(tags)}; mat = np.zeros((n, n))
    for _, row in pair_df.iterrows():
        if row["model_a"] not in ti or row["model_b"] not in ti: continue
        v = row[col]
        if log_scale and v > 0: v = -np.log10(v + 1e-15)
        mat[ti[row["model_a"]], ti[row["model_b"]]] = v
        mat[ti[row["model_b"]], ti[row["model_a"]]] = v
    short = [pair_info.loc[t, "label"].replace(" × ", "×").replace(" (Homo)"," H") for t in tags]
    fig, ax = plt.subplots(figsize=(16, 14))
    kw = dict(annot=False, cmap=cmap, xticklabels=short, yticklabels=short, ax=ax,
              linewidths=0.3, linecolor="#eee", cbar_kws={"shrink": 0.6})
    if center is not None: kw["center"] = center
    else: kw["vmin"] = vmin; kw["vmax"] = vmax
    sns.heatmap(mat, **kw)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(f"{STATS_DIR}/{fname}", bbox_inches="tight"); plt.close()


def plot_cd_diagram(tags, avg_ranks, nem_p, pair_info):
    n = len(tags); order = np.argsort(avg_ranks)
    fig, ax = plt.subplots(figsize=(14, max(6, n*0.35)))
    ymax = n+1; ystep = 1.; ypos = {}
    for ri, mi in enumerate(order):
        y = ymax - ri*ystep - 1; ypos[mi] = y
        col = "#4C8EDA" if pair_info.loc[tags[mi], "group"] == "hetero" else "#E8622A"
        ax.plot(avg_ranks[mi], y, "o", color=col, ms=9, zorder=5)
        ax.text(avg_ranks[mi]+0.06, y, pair_info.loc[tags[mi], "label"],
                va="center", fontsize=8, color=col)
    drawn = set()
    for i, j in itertools.combinations(range(n), 2):
        if nem_p[i, j] >= ALPHA:
            key = (min(avg_ranks[i], avg_ranks[j]), max(avg_ranks[i], avg_ranks[j]))
            if key not in drawn:
                drawn.add(key)
                ax.plot([avg_ranks[i], avg_ranks[j]], [ypos[i], ypos[j]],
                        "-", color="#ccc", lw=1.5, zorder=1, alpha=0.5)
    ax.set_xlabel("Average Rank (lower=better)", fontsize=11)
    ax.set_title(f"Critical Difference Diagram (Nemenyi α={ALPHA})\n"
                 "Lines=not significantly different  Blue=Hetero · Orange=Homo", fontsize=11)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlim(avg_ranks.min()-0.5, avg_ranks.max()+0.5); ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(f"{STATS_DIR}/cd_diagram.png", bbox_inches="tight"); plt.close()


def plot_group_violin(df_grp, paired_diffs, tags, folds, pair_info):
    h = [float(folds[t]["auc"].mean()) for t in tags if pair_info.loc[t, "group"] == "hetero"]
    o = [float(folds[t]["auc"].mean()) for t in tags if pair_info.loc[t, "group"] == "homo"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, (vals, col) in enumerate(zip([h, o], ["#4C8EDA", "#E8622A"])):
        vp = axes[0].violinplot([vals], positions=[i], showmeans=True)
        for body in vp["bodies"]: body.set_facecolor(col); body.set_alpha(0.5)
        for part in ("cbars","cmins","cmaxes","cmeans"): vp[part].set_color("#222"); vp[part].set_lw(1.8)
        jit = rng.uniform(-0.07, 0.07, len(vals))
        axes[0].scatter(np.full(len(vals), i)+jit, vals, s=50, color=col, edgecolors="white", lw=0.7, zorder=5)
        axes[0].text(i, max(vals)+0.012, f"μ={np.mean(vals):.3f}", ha="center", fontsize=11,
                     fontweight="bold", color=col)
    mw = df_grp[df_grp["test"] == "Mann-Whitney U"]
    if not mw.empty:
        axes[0].text(0.5, max(max(h), max(o))+0.025, f"Mann-Whitney p={mw.iloc[0]['p_value']:.4f}",
                     ha="center", fontsize=10, color="#333")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels([f"Hetero\n(n={len(h)})", f"Homo\n(n={len(o)})"], fontsize=12)
    axes[0].set_ylabel("Mean AUC (5-fold CV)"); axes[0].set_title("AUC by Group")
    axes[0].yaxis.grid(True, alpha=0.3); axes[0].set_axisbelow(True)
    if paired_diffs:
        pd_arr = sorted(paired_diffs)
        axes[1].axhline(0, color="#ccc", lw=1.5, ls="--")
        axes[1].bar(range(len(pd_arr)), pd_arr,
                    color=["#4C8EDA" if d >= 0 else "#E8622A" for d in pd_arr],
                    alpha=0.75, edgecolor="white")
        axes[1].set_xlabel("Pair index (sorted)", fontsize=11)
        axes[1].set_ylabel("AUC_hetero − mean(AUC_homo components)", fontsize=10)
        axes[1].set_title("Per-pair AUC gain over homo baseline")
        axes[1].yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{STATS_DIR}/group_violin.png", bbox_inches="tight"); plt.close()


def plot_diversity_scatter(df_dg):
    if df_dg.empty or "disagreement" not in df_dg.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, xcol, xtitle in zip(axes, ["disagreement", "double_fault"],
                                 ["Disagreement (homo branch probs)", "Double-fault"]):
        sub = df_dg.dropna(subset=[xcol, "auc_gain"])
        if sub.empty: continue
        ax.scatter(sub[xcol], sub["auc_gain"], s=55, color="#4C8EDA", edgecolors="white", lw=0.7, zorder=5)
        if len(sub) >= 3:
            m, b, *_ = scipy.stats.linregress(sub[xcol], sub["auc_gain"])
            x_ = np.linspace(sub[xcol].min(), sub[xcol].max(), 100)
            rcol = "spearman_dis_r" if "dis" in xcol.split("_")[0] else "spearman_df_r"
            if rcol in sub.columns:
                ax.plot(x_, m*x_+b, "--", color="#DC267F", lw=1.8,
                        label=f"ρ={sub[rcol].iloc[0]:.3f}"); ax.legend(fontsize=10)
        ax.axhline(0, color="#ccc", lw=1, ls=":")
        ax.set_xlabel(xtitle, fontsize=11)
        ax.set_ylabel("AUC gain over best component", fontsize=11)
        ax.set_title(f"{xtitle} vs AUC gain"); ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{STATS_DIR}/diversity_scatter.png", bbox_inches="tight"); plt.close()


def plot_power_bars(df_cal, pair_info):
    order = df_cal.sort_values("power", ascending=False)
    fig, ax = plt.subplots(figsize=(22, 5))
    ax.bar(range(len(order)), order["power"],
           color=["#4C8EDA" if pair_info.loc[t, "group"] == "hetero" else "#E8622A"
                  for t in order["tag"]], alpha=0.85, edgecolor="white")
    ax.axhline(0.8, color="#333", lw=1.5, ls="--", label="80% power")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order["label"].tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hanley–McNeil Power"); ax.set_ylim(0, 1.08)
    ax.set_title("Statistical Power per Model (α=0.05, H₀: AUC=0.5)\nBlue=Hetero · Orange=Homo")
    ax.legend(fontsize=10); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{STATS_DIR}/power_bars.png", bbox_inches="tight"); plt.close()
