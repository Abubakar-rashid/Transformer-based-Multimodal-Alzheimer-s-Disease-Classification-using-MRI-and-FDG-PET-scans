"""
run_fusion.py — Main orchestration for CAFNet cross-architecture fusion experiments.

Usage (Kaggle):
    python -m fusion.run_fusion               # full pipeline (training + stats)
    SKIP_TRAINING=1 python -m fusion.run_fusion  # stats only (needs existing CSVs)

Pipeline phases:
    1  Pre-cache backbone features (24 .npz files)
    1b Backbone inference timing
    2a Solo backbone baselines (12 × LogReg, 5-fold CV)
    2b Cross-modality MLP fusion (36 pairs × 5 seeds × 5 folds)
    2c Same-modality MLP fusion  (30 pairs × 5 seeds × 5 folds)
    2d LR-stacking baselines     (66 pairs)
    2e Probability averaging baselines (66 pairs)
    3  Summary plots and CSVs
    4  Full statistical analysis
"""
import os, gc, json, time
import numpy as np
import pandas as pd
import torch

from .config import (
    SKIP_TRAINING, DEVICE, ARCHS, ARCH_IMG, MRI_CKPTS, PET_CKPTS,
    CROSS_MOD_PAIRS, SAME_MOD_PAIRS, ALL_FUSION_PAIRS, SOLO_MODELS,
    CROSS_DIR, SAME_DIR, SOLO_DIR, BASE_DIR, STATS_DIR, OUT_DIR,
    COLORS, ALL_LABELS, PAIR_LABELS,
    MRI_CSV, PET_CSV,
)
from .data import load_df, get_arch_cache
from .train import run_pair
from .baselines import eval_solo_lr, eval_lr_stacking, compute_averaging_baseline
from .plots import (
    plot_roc_folds, plot_confusion, plot_fold_auc,
    plot_roc_grid_by_mri, plot_auc_by_mri, plot_auc_heatmap_6x6,
    plot_hetero_vs_homo, plot_summary_all,
    plot_seed_stability, plot_ablation_comparison, plot_solo_vs_fusion, plot_group_summary,
    plot_calibration_grid, plot_pairwise_hm, plot_cd_diagram,
    plot_group_violin, plot_diversity_scatter, plot_power_bars,
)
from .stats import (
    load_stats_inputs, build_aligned,
    stats_per_model, stats_pairwise, stats_omnibus,
    stats_group, stats_diversity_gain, stats_ablation,
)
from .models import Extractor


def measure_backbone_inference_times(n_warmup=10, n_runs=100):
    """Time each backbone (arch × mod) with a single dummy image."""
    def _cuda_sync():
        if torch.cuda.is_available(): torch.cuda.synchronize()

    print("\n" + "=" * 65)
    print("  Backbone inference timing (batch_size=1, single sample)")
    print("=" * 65)
    rows = []
    for arch in ARCHS:
        for mod in ["mri", "pet"]:
            ckpt     = MRI_CKPTS[arch] if mod == "mri" else PET_CKPTS[arch]
            img_size = ARCH_IMG[arch]
            tag      = f"{arch}_{mod}"
            try:
                ext   = Extractor(arch, ckpt, img_size=img_size).to(DEVICE)
                dummy = torch.randn(1, 3, img_size, img_size).to(DEVICE)
                with torch.no_grad():
                    for _ in range(n_warmup): ext(dummy)
                _cuda_sync()
                times = []
                with torch.no_grad():
                    for _ in range(n_runs):
                        _cuda_sync()
                        t0 = time.perf_counter()
                        ext(dummy)
                        _cuda_sync()
                        times.append((time.perf_counter() - t0) * 1000.0)
                rows.append({"arch": arch, "mod": mod, "tag": tag,
                             "mean_ms": round(float(np.mean(times)), 3),
                             "std_ms":  round(float(np.std(times)),  3),
                             "min_ms":  round(float(np.min(times)),  3),
                             "max_ms":  round(float(np.max(times)),  3),
                             "device":  str(DEVICE), "n_runs": n_runs,
                             "img_size": img_size})
                ext.cpu(); del ext; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                print(f"    {tag:<25} {np.mean(times):.2f} ± {np.std(times):.2f} ms/sample")
            except Exception as e:
                print(f"    WARNING: {tag} timing failed: {e}")
                rows.append({"arch": arch, "mod": mod, "tag": tag,
                             "mean_ms": float("nan"), "std_ms": float("nan"),
                             "device": str(DEVICE)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{STATS_DIR}/backbone_inference_times.csv", index=False)
    print(f"  Saved backbone_inference_times.csv  ({len(df)} rows)\n")
    return df


def save_summary_csv(all_results):
    from .config import ALL_PAIRS
    rows = []
    for pair in ALL_PAIRS:
        tag = pair["tag"]
        if tag not in all_results: continue
        row = {"pair": PAIR_LABELS[tag], "tag": tag, "group": pair["group"],
               "mri_arch": pair.get("mri_arch",""), "pet_arch": pair.get("pet_arch","")}
        for met in ["auc","acc","sens","spec","f1","pr_auc","bal_acc"]:
            vs = [r[met] for r in all_results[tag]]
            row[f"{met}_mean"] = round(np.mean(vs), 4); row[f"{met}_std"]  = round(np.std(vs),  4)
            row[f"{met}_min"]  = round(np.min(vs),  4); row[f"{met}_max"]  = round(np.max(vs),  4)
        rows.append(row)
        print(f"  [{pair['group']}]  {PAIR_LABELS[tag]:<40}  AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f}")
    pd.DataFrame(rows).to_csv("/kaggle/working/kfold_h36_summary.csv", index=False)


def _save_model_registry(cross_pairs, same_pairs, solo_models, all_fusion_pairs):
    rows = []
    for p in cross_pairs:
        rows.append({"tag": p["tag"], "label": p["label"], "group": p["group"],
                     "b1_arch": p["b1_arch"], "b1_mod": p["b1_mod"],
                     "b2_arch": p["b2_arch"], "b2_mod": p["b2_mod"], "out_dir": p["out_dir"]})
    for p in same_pairs:
        rows.append({"tag": p["tag"], "label": p["label"], "group": p["group"],
                     "b1_arch": p["b1_arch"], "b1_mod": p["b1_mod"],
                     "b2_arch": p["b2_arch"], "b2_mod": p["b2_mod"], "out_dir": p["out_dir"]})
    for s in solo_models:
        rows.append({"tag": s["tag"], "label": s["label"], "group": s["group"],
                     "b1_arch": s["arch"], "b1_mod": s["mod"],
                     "b2_arch": None, "b2_mod": None, "out_dir": SOLO_DIR})
    for p in all_fusion_pairs:
        rows.append({"tag": p["tag"]+"_lr", "label": ALL_LABELS[p["tag"]+"_lr"],
                     "group": "lr_stack", "b1_arch": p["b1_arch"], "b1_mod": p["b1_mod"],
                     "b2_arch": p["b2_arch"], "b2_mod": p["b2_mod"], "out_dir": BASE_DIR})
        rows.append({"tag": p["tag"]+"_avg", "label": ALL_LABELS[p["tag"]+"_avg"],
                     "group": "avg", "b1_arch": p["b1_arch"], "b1_mod": p["b1_mod"],
                     "b2_arch": p["b2_arch"], "b2_mod": p["b2_mod"], "out_dir": BASE_DIR})
    pd.DataFrame(rows).to_csv("/kaggle/working/model_info.csv", index=False)
    pd.DataFrame([{"tag": p["tag"], "label": p["label"],
                   "mri_arch": p["b1_arch"], "pet_arch": p["b2_arch"], "group": p["group"]}
                  for p in cross_pairs]).to_csv("/kaggle/working/pair_info.csv", index=False)
    print(f"  Saved: model_info.csv ({len(rows)} models)")


def build_fusion_inference_table(all_fusion_pairs):
    bt_path = f"{STATS_DIR}/backbone_inference_times.csv"
    if not os.path.exists(bt_path):
        print("  WARNING: backbone_inference_times.csv not found — skipping fusion timing table.")
        return
    bt = pd.read_csv(bt_path).set_index(["arch", "mod"])
    sc_path = f"{STATS_DIR}/slice_counts.json"
    avg_mri_slices = avg_pet_slices = float("nan")
    if os.path.exists(sc_path):
        with open(sc_path) as _f:
            _sc = json.load(_f)
        avg_mri_slices = float(_sc["avg_mri_slices"])
        avg_pet_slices = float(_sc["avg_pet_slices"])
    rows = []
    for pair in all_fusion_pairs:
        tag = pair["tag"]; od = pair["out_dir"]; tp = f"{od}/{tag}_timing.csv"
        if not os.path.exists(tp): continue
        td = pd.read_csv(tp).iloc[0]
        mlp_ms  = float(td["mlp_ms_per_sample"])
        mlp_std = float(td.get("mlp_std_ms", float("nan")))
        b1 = (pair["b1_arch"], pair["b1_mod"]); b2 = (pair["b2_arch"], pair["b2_mod"])
        def _get(key, col): return float(bt.loc[key, col]) if key in bt.index else float("nan")
        b1_ms = _get(b1, "mean_ms"); b1_std = _get(b1, "std_ms")
        b2_ms = _get(b2, "mean_ms"); b2_std = _get(b2, "std_ms")
        rows.append({
            "tag": tag, "label": pair["label"], "group": pair["group"],
            "b1_arch": pair["b1_arch"], "b1_mod": pair["b1_mod"],
            "b2_arch": pair["b2_arch"], "b2_mod": pair["b2_mod"],
            "b1_ms": round(b1_ms, 3), "b1_std_ms": round(b1_std, 3),
            "b2_ms": round(b2_ms, 3), "b2_std_ms": round(b2_std, 3),
            "mlp_ms": round(mlp_ms, 3), "mlp_std_ms": round(mlp_std, 3),
            "total_ms": round(b1_ms + b2_ms + mlp_ms, 3),
            "total_std_ms": round(float(np.sqrt(b1_std**2 + b2_std**2 + mlp_std**2)), 3),
            "avg_mri_slices": avg_mri_slices, "avg_pet_slices": avg_pet_slices,
            "per_subject_ms": round(avg_mri_slices * b1_ms + avg_pet_slices * b2_ms + mlp_ms, 1),
            "per_subject_std_ms": round(float(np.sqrt(
                (avg_mri_slices * b1_std)**2 + (avg_pet_slices * b2_std)**2 + mlp_std**2)), 1),
        })
    if rows:
        df = pd.DataFrame(rows).sort_values("total_ms")
        df.to_csv(f"{STATS_DIR}/fusion_inference_times.csv", index=False)
        print(f"  Saved fusion_inference_times.csv  ({len(df)} pairs)")
        print(f"    Fastest: {df.iloc[0]['label']}  {df.iloc[0]['total_ms']:.1f} ms")
        print(f"    Slowest: {df.iloc[-1]['label']}  {df.iloc[-1]['total_ms']:.1f} ms")


def run_statistical_analysis():
    """Phase 4: load saved CSVs and run all statistical tests + plots."""
    print("\n" + "=" * 65)
    print("PHASE 4 — Statistical Analysis")
    print(f"  Writing to:   {STATS_DIR}")
    print("=" * 65)
    print("  Note: ECE uses equal-mass (quantile) bins, n_bins=5 (more stable at N=150).\n")

    tags, predictions, folds, pair_info = load_stats_inputs()

    cross_tags = [t for t in tags if pair_info.loc[t, "group"] in ("hetero","homo")]
    solo_tags  = [t for t in tags if "solo" in pair_info.loc[t, "group"]]
    mlp_tags   = [t for t in tags if pair_info.loc[t, "group"] not in ("avg","lr_stack")
                  and "solo" not in pair_info.loc[t, "group"]]
    stat_tags  = mlp_tags + solo_tags

    y_true, probs, pred_arr = build_aligned(stat_tags, predictions)
    print(f"  MLP models: {len(mlp_tags)}  Solo: {len(solo_tags)}")
    print(f"  Common subjects for stats: {len(y_true)}\n")

    df_ci, df_cal = stats_per_model(stat_tags, y_true, probs, pred_arr, pair_info, predictions)

    cross_idx      = [i for i, t in enumerate(stat_tags) if pair_info.loc[t, "group"] in ("hetero","homo")]
    cross_sub_tags = [stat_tags[i] for i in cross_idx]

    df_dl, df_mc, df_div, df_nb = stats_pairwise(
        stat_tags, y_true, probs, pred_arr, folds, pair_info)

    mlp_row_idx = [i for i, t in enumerate(stat_tags) if t in mlp_tags]
    avg_ranks, nem_p = stats_omnibus(mlp_tags, y_true, pred_arr[mlp_row_idx], folds, pair_info)

    df_grp, paired_diffs = stats_group(stat_tags, y_true, probs, pred_arr, folds, pair_info, predictions)

    df_dg = stats_diversity_gain(cross_tags, probs[cross_idx], y_true, folds, pair_info)

    stats_ablation(tags, predictions, pair_info)

    print("  [stats 7/7] Generating plots...")
    plot_calibration_grid(stat_tags, y_true, probs, pair_info)
    plot_pairwise_hm(stat_tags, df_dl, "p_raw_bh",
                     "DeLong p-values (BH-corrected, −log₁₀)", "heatmap_delong_bh.png",
                     pair_info, cmap="RdYlGn", log_scale=True)
    plot_pairwise_hm(stat_tags, df_mc, "p_exact_bh",
                     "McNemar p-values (BH-corrected, −log₁₀)", "heatmap_mcnemar_bh.png",
                     pair_info, cmap="RdYlGn", log_scale=True)
    plot_pairwise_hm(cross_sub_tags, df_div, "kappa",
                     "Cohen's κ (cross-modality pairs)", "heatmap_kappa.png",
                     pair_info, cmap="coolwarm", center=0.0)
    plot_pairwise_hm(cross_sub_tags, df_div, "disagreement",
                     "Disagreement (cross-modality)", "heatmap_disagreement.png",
                     pair_info, cmap="YlOrRd", vmin=0, vmax=0.5)
    plot_group_violin(df_grp, paired_diffs, stat_tags, folds, pair_info)
    plot_cd_diagram(mlp_tags, avg_ranks, nem_p, pair_info)
    plot_diversity_scatter(df_dg)
    plot_power_bars(df_cal, pair_info)

    print(f"\n  All stats outputs → {STATS_DIR}/")
    for f in sorted(f for f in os.listdir(STATS_DIR) if f.endswith(".csv")):
        print(f"    {f}")


if __name__ == "__main__":

    if not SKIP_TRAINING:
        print(f"All paths set.  Device: {DEVICE}\n")

        mri_df = load_df(MRI_CSV); pet_df = load_df(PET_CSV)

        # ── Phase 1: cache backbone features ─────────────────────────────────
        print("=" * 65)
        print("PHASE 1 — Pre-caching backbone features (24 files)")
        print("=" * 65)
        all_caches = {}
        for arch in ARCHS:
            sz = ARCH_IMG[arch]; print(f"\n  [{arch.upper()}]  img={sz}")
            all_caches[(arch,"mri","aug")] = get_arch_cache(arch,"mri",sz,MRI_CKPTS[arch],mri_df,True)
            all_caches[(arch,"mri","det")] = get_arch_cache(arch,"mri",sz,MRI_CKPTS[arch],mri_df,False)
            all_caches[(arch,"pet","aug")] = get_arch_cache(arch,"pet",sz,PET_CKPTS[arch],pet_df,True)
            all_caches[(arch,"pet","det")] = get_arch_cache(arch,"pet",sz,PET_CKPTS[arch],pet_df,False)
        global_subjects = sorted(set.intersection(*[set(c.keys()) for c in all_caches.values()]))
        glab = np.array([all_caches[(ARCHS[0],"mri","aug")][s][1] for s in global_subjects])
        ad_g = int(glab.sum()); cn_g = len(global_subjects) - ad_g
        print(f"\nGlobal pool: {len(global_subjects)} subjects  AD={ad_g}  CN={cn_g}")

        _mri_sc = mri_df[mri_df["subject_id"].isin(global_subjects)].groupby("subject_id").size()
        _pet_sc = pet_df[pet_df["subject_id"].isin(global_subjects)].groupby("subject_id").size()
        _sc = {"avg_mri_slices": round(float(_mri_sc.mean()), 1),
               "avg_pet_slices": round(float(_pet_sc.mean()), 1)}
        with open(f"{STATS_DIR}/slice_counts.json", "w") as _f:
            json.dump(_sc, _f)

        # ── Phase 1b: backbone inference timing ───────────────────────────────
        measure_backbone_inference_times()

        # ── Phase 2a: solo backbone baselines ─────────────────────────────────
        print("\n" + "=" * 65)
        print("PHASE 2a — Solo backbone baselines (12 × LogReg, 5-fold CV)")
        print("=" * 65)
        solo_fold_preds = {}; solo_results = {}
        for solo in SOLO_MODELS:
            fr, fpr = eval_solo_lr(solo, all_caches, global_subjects)
            solo_results[solo["tag"]]    = fr
            solo_fold_preds[solo["tag"]] = pd.DataFrame(fpr)

        # ── Phase 2b: cross-modality MLP fusion ───────────────────────────────
        print("\n" + "=" * 65)
        print(f"PHASE 2b — Cross-modality MLP fusion  "
              f"({len(CROSS_MOD_PAIRS)} pairs × 5 seeds × 5 folds"
              f" = {len(CROSS_MOD_PAIRS)*5*5} runs)")
        print("=" * 65)
        cross_results = {}
        for idx, pair in enumerate(CROSS_MOD_PAIRS):
            kfold_rows, s42 = run_pair(pair, all_caches, global_subjects)
            cross_results[pair["tag"]] = s42
            color = COLORS[idx % len(COLORS)]
            plot_roc_folds(s42, pair["tag"], color)
            plot_confusion(s42, pair["tag"])
            plot_fold_auc(s42, pair["tag"], color)

        # ── Phase 2c: same-modality MLP fusion ────────────────────────────────
        print("\n" + "=" * 65)
        print(f"PHASE 2c — Same-modality MLP fusion  "
              f"({len(SAME_MOD_PAIRS)} pairs × 5 seeds × 5 folds"
              f" = {len(SAME_MOD_PAIRS)*5*5} runs)")
        print("=" * 65)
        same_results = {}
        for idx, pair in enumerate(SAME_MOD_PAIRS):
            kfold_rows, s42 = run_pair(pair, all_caches, global_subjects)
            same_results[pair["tag"]] = s42

        # ── Phase 2d: LR-stacking baselines ───────────────────────────────────
        print("\n" + "=" * 65)
        print(f"PHASE 2d — LR stacking baselines ({len(ALL_FUSION_PAIRS)} pairs)")
        print("=" * 65)
        lr_results = {}
        for pair in ALL_FUSION_PAIRS:
            lr_results[pair["tag"] + "_lr"] = eval_lr_stacking(pair, all_caches, global_subjects)

        # ── Phase 2e: averaging baselines ─────────────────────────────────────
        print("\n" + "=" * 65)
        print(f"PHASE 2e — Averaging baselines ({len(ALL_FUSION_PAIRS)} pairs)")
        print("=" * 65)
        avg_results = {}
        for pair in ALL_FUSION_PAIRS:
            fr = compute_averaging_baseline(pair, solo_fold_preds)
            if fr: avg_results[pair["tag"] + "_avg"] = fr

        # ── Phase 3: summary plots and CSVs ───────────────────────────────────
        print("\n" + "=" * 65)
        print("PHASE 3 — Summary plots and CSVs")
        print("=" * 65)
        all_results = cross_results
        save_summary_csv(all_results)
        for ma in ARCHS:
            plot_auc_by_mri(all_results, ma)
            plot_roc_grid_by_mri(all_results, ma)
        plot_auc_heatmap_6x6(all_results)
        plot_hetero_vs_homo(all_results)
        plot_summary_all(all_results)
        plot_seed_stability(CROSS_MOD_PAIRS + SAME_MOD_PAIRS)
        plot_ablation_comparison(ALL_FUSION_PAIRS, cross_results, same_results,
                                 solo_results, lr_results, avg_results)
        plot_solo_vs_fusion(ARCHS, solo_results, cross_results)
        plot_group_summary(CROSS_MOD_PAIRS, SAME_MOD_PAIRS, SOLO_MODELS,
                           cross_results, same_results, solo_results)
        _save_model_registry(CROSS_MOD_PAIRS, SAME_MOD_PAIRS, SOLO_MODELS, ALL_FUSION_PAIRS)
        build_fusion_inference_table(ALL_FUSION_PAIRS)

    # ── Phase 4: statistical analysis (always runs) ───────────────────────────
    run_statistical_analysis()
