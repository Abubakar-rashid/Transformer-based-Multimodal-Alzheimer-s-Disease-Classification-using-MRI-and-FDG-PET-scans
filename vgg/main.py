import os
import torch
import torch.nn as nn

from config import DEVICE, CFG, OUT_DIR, SPLIT_DIR
from data import load_split_csv, make_single_loaders_backbone, make_multimodal_loaders
from model import VGG19Extractor, MultimodalFusionModel
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

    mri_bb_train_subjs = set(mri_backbone[mri_backbone["split"] == "train"]["subject_id"])
    mri_bb_test_subjs  = set(mri_backbone[mri_backbone["split"] == "test"]["subject_id"])
    overlap_subjs      = set(mri_fusion["subject_id"])
    print(f"\n  MRI backbone: {len(mri_bb_train_subjs)} train subjects, "
          f"{len(mri_bb_test_subjs)} test subjects")
    print(f"  Fusion: {mri_fusion['subject_id'].nunique()} overlap subjects")

    # ── STAGE 1: MRI VGG19 ────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1 — MRI VGG19")
    print("="*60)

    mri_loaders = make_single_loaders_backbone(
        mri_backbone, CFG["batch_size"], CFG["img_size"], "MRI")

    mri_model = VGG19Extractor(
        CFG["dense_units"], CFG["dropout"], CFG["num_classes"]).to(DEVICE)

    mri_model, mri_hist = fit(
        mri_model, mri_loaders,
        save_path=os.path.join(OUT_DIR, "mri_vgg19_best.pth"),
        name="MRI", lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], device=DEVICE,
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
        CFG["img_size"], "MRI_VGG19")

    # ── STAGE 2: PET VGG19 ────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2 — PET VGG19")
    print("="*60)

    pet_loaders = make_single_loaders_backbone(
        pet_backbone, CFG["batch_size"], CFG["img_size"], "PET")

    pet_model = VGG19Extractor(
        CFG["dense_units"], CFG["dropout"], CFG["num_classes"]).to(DEVICE)

    pet_model, pet_hist = fit(
        pet_model, pet_loaders,
        save_path=os.path.join(OUT_DIR, "pet_vgg19_best.pth"),
        name="PET", lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], device=DEVICE,
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
        CFG["img_size"], "PET_VGG19")

    # ── STAGE 3: Multimodal Fusion ────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3 — Multimodal Fusion (VGG19 backbones frozen)")
    print("="*60)

    mri_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "mri_vgg19_best.pth"), map_location=DEVICE))
    pet_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "pet_vgg19_best.pth"), map_location=DEVICE))
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
        save_path=os.path.join(OUT_DIR, "multimodal_best.pth"),
        name="Multimodal", lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], device=DEVICE, multimodal=True,
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
        CFG["img_size"], "Multimodal_VGG19")

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"  {'Model':<20} {'Test Subjects':<15} {'Slice Acc':>10} {'Subj Acc':>10} {'Subj ROC':>10}")
    print(f"  {'-'*75}")
    print(f"  {'MRI VGG19':<20} {len(mri_bb_test_subjs):<15} {mri_acc:>10.4f} {mri_su_acc:>10.4f} {mri_su_roc:>10.4f}")
    print(f"  {'PET VGG19':<20} {len(mri_bb_test_subjs):<15} {pet_acc:>10.4f} {pet_su_acc:>10.4f} {pet_su_roc:>10.4f}")
    print(f"  {'Multimodal':<20} {mri_fusion['subject_id'].nunique():<15} {mm_acc:>10.4f} {mm_su_acc:>10.4f} {mm_su_roc:>10.4f}")
    print(f"\n  Backbones tested on SAME {len(overlap_subjs)} subjects used in fusion!")
    print(f"  All models + plots saved to {OUT_DIR}")

    torch.save(mm_model.state_dict(),
               os.path.join(OUT_DIR, "multimodal_final.pth"))


if __name__ == "__main__":
    main()
