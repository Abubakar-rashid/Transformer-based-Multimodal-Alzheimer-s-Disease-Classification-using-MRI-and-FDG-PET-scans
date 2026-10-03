import os
import torch
import torch.nn as nn

from config import DEVICE, CFG, OUT_DIR, SPLIT_DIR
from data import load_split_csv, make_single_loaders_backbone, make_multimodal_loaders
from model import ConvNeXtV2Extractor, MultimodalFusionModel
from train import fit, evaluate, print_results
from inference import measure_inference_time, measure_inference_time_multimodal
from plots import plot_curves, plot_roc_pr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n=== Loading Pre-Computed Splits ===")
    mri_backbone = load_split_csv(os.path.join(SPLIT_DIR, "mri_backbone_splits.csv"), "MRI Backbone")
    pet_backbone = load_split_csv(os.path.join(SPLIT_DIR, "pet_backbone_splits.csv"), "PET Backbone")
    mri_fusion   = load_split_csv(os.path.join(SPLIT_DIR, "mri_fusion_splits.csv"),   "MRI Fusion")
    pet_fusion   = load_split_csv(os.path.join(SPLIT_DIR, "pet_fusion_splits.csv"),   "PET Fusion")

    mri_bb_test_subjs = set(mri_backbone[mri_backbone["split"] == "test"]["subject_id"])
    print(f"\n  MRI backbone: {len(set(mri_backbone[mri_backbone['split']=='train']['subject_id']))} train subjects, "
          f"{len(mri_bb_test_subjs)} test subjects")
    print(f"  Fusion: {mri_fusion['subject_id'].nunique()} overlap subjects")

    # ── STAGE 1: MRI ConvNeXtV2 ───────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1 — MRI ConvNeXtV2")
    print("="*60)

    mri_loaders = make_single_loaders_backbone(
        mri_backbone, CFG["batch_size"], CFG["img_size"], "MRI")

    mri_model = ConvNeXtV2Extractor(
        CFG["dense_units"], CFG["dropout"], CFG["num_classes"]).to(DEVICE)

    mri_model, mri_hist = fit(
        mri_model, mri_loaders,
        save_path=os.path.join(OUT_DIR, "mri_convnext_best.pth"),
        name="MRI", lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], device=DEVICE,
        weight_decay=CFG["weight_decay"], warmup_epochs=CFG["warmup_epochs"],
    )
    plot_curves(mri_hist, "MRI")

    _, _, mri_preds, mri_lbls, mri_probs, mri_subjects = evaluate(
        mri_model, mri_loaders["test"], nn.CrossEntropyLoss(), DEVICE)
    mri_acc, mri_su_acc, mri_sl_roc, mri_su_roc, mri_sl_pr, mri_su_pr, \
        mri_su_preds, mri_su_lbls, mri_su_probs = print_results(
            "MRI_overlap_test", mri_preds, mri_lbls, mri_probs, mri_subjects)
    plot_roc_pr(mri_lbls, mri_probs, "MRI_slice")
    plot_roc_pr(mri_su_lbls, mri_su_probs, "MRI_subject")

    measure_inference_time(
        mri_model, mri_backbone[mri_backbone["split"] == "test"],
        CFG["img_size"], "MRI_ConvNeXtV2")

    # ── STAGE 2: PET ConvNeXtV2 ───────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2 — PET ConvNeXtV2")
    print("="*60)

    pet_loaders = make_single_loaders_backbone(
        pet_backbone, CFG["batch_size"], CFG["img_size"], "PET")

    pet_model = ConvNeXtV2Extractor(
        CFG["dense_units"], CFG["dropout"], CFG["num_classes"]).to(DEVICE)

    pet_model, pet_hist = fit(
        pet_model, pet_loaders,
        save_path=os.path.join(OUT_DIR, "pet_convnext_best.pth"),
        name="PET", lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], device=DEVICE,
        weight_decay=CFG["weight_decay"], warmup_epochs=CFG["warmup_epochs"],
    )
    plot_curves(pet_hist, "PET")

    _, _, pet_preds, pet_lbls, pet_probs, pet_subjects = evaluate(
        pet_model, pet_loaders["test"], nn.CrossEntropyLoss(), DEVICE)
    pet_acc, pet_su_acc, pet_sl_roc, pet_su_roc, pet_sl_pr, pet_su_pr, \
        pet_su_preds, pet_su_lbls, pet_su_probs = print_results(
            "PET_overlap_test", pet_preds, pet_lbls, pet_probs, pet_subjects)
    plot_roc_pr(pet_lbls, pet_probs, "PET_slice")
    plot_roc_pr(pet_su_lbls, pet_su_probs, "PET_subject")

    measure_inference_time(
        pet_model, pet_backbone[pet_backbone["split"] == "test"],
        CFG["img_size"], "PET_ConvNeXtV2")

    # ── STAGE 3: Multimodal Fusion ────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3 — Multimodal Fusion (ConvNeXtV2 backbones frozen)")
    print("="*60)

    mri_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "mri_convnext_best.pth"), map_location=DEVICE))
    pet_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "pet_convnext_best.pth"), map_location=DEVICE))
    for p in mri_model.backbone.parameters(): p.requires_grad = False
    for p in pet_model.backbone.parameters(): p.requires_grad = False

    mm_model = MultimodalFusionModel(
        mri_model, pet_model,
        CFG["dense_units"], CFG["dropout"], CFG["num_classes"],
    ).to(DEVICE)

    print("\n  Multimodal loaders:")
    mm_loaders = make_multimodal_loaders(
        mri_fusion, pet_fusion, CFG["batch_size"], CFG["img_size"])

    mm_model, mm_hist = fit(
        mm_model, mm_loaders,
        save_path=os.path.join(OUT_DIR, "multimodal_best.pth"),
        name="Multimodal", lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], device=DEVICE,
        weight_decay=CFG["weight_decay"], warmup_epochs=CFG["warmup_epochs"],
        multimodal=True,
    )
    plot_curves(mm_hist, "Multimodal")

    _, _, mm_preds, mm_lbls, mm_probs, mm_subjects = evaluate(
        mm_model, mm_loaders["test"], nn.CrossEntropyLoss(), DEVICE, multimodal=True)
    mm_acc, mm_su_acc, mm_sl_roc, mm_su_roc, mm_sl_pr, mm_su_pr, \
        mm_su_preds, mm_su_lbls, mm_su_probs = print_results(
            "Multimodal", mm_preds, mm_lbls, mm_probs, mm_subjects)
    plot_roc_pr(mm_lbls, mm_probs, "Multimodal_slice")
    plot_roc_pr(mm_su_lbls, mm_su_probs, "Multimodal_subject")

    measure_inference_time_multimodal(
        mm_model,
        mri_fusion[mri_fusion["split"] == "test"],
        pet_fusion[pet_fusion["split"] == "test"],
        CFG["img_size"], "Multimodal_ConvNeXtV2")

    # ── Final summary ─────────────────────────────────────────
    n_bb_test  = len(mri_bb_test_subjs)
    n_fus_test = mri_fusion[mri_fusion["split"] == "test"]["subject_id"].nunique()

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"  {'Model':<25} {'Test Subjects':<15} {'Slice Acc':>10} {'Subj Acc':>10} {'Subj ROC':>10}")
    print(f"  {'-'*75}")
    print(f"  {'MRI ConvNeXtV2':<25} {n_bb_test:<15} {mri_acc:>10.4f} {mri_su_acc:>10.4f} {mri_su_roc:>10.4f}")
    print(f"  {'PET ConvNeXtV2':<25} {n_bb_test:<15} {pet_acc:>10.4f} {pet_su_acc:>10.4f} {pet_su_roc:>10.4f}")
    print(f"  {'Multimodal ConvNeXtV2':<25} {n_fus_test:<15} {mm_acc:>10.4f} {mm_su_acc:>10.4f} {mm_su_roc:>10.4f}")
    print(f"\n  All models + plots saved to {OUT_DIR}")

    torch.save(mm_model.state_dict(),
               os.path.join(OUT_DIR, "multimodal_final.pth"))


if __name__ == "__main__":
    main()
