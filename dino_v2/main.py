import os
import torch
import torch.nn as nn

from config import (DEVICE, CFG_SHARED, CFG_MRI, CFG_PET, CFG_FUSION,
                    OUT_DIR, SPLIT_DIR)
from data import load_split_csv, remap_paths, make_single_loaders_backbone, make_multimodal_loaders
from model import DINOv2Extractor, MultimodalFusionModel
from train import fit, evaluate, print_results
from inference import measure_inference_time, measure_inference_time_multimodal
from plots import plot_curves, plot_roc_pr


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
    print(f"        Replace CFG_FUSION values in config.py if HPO yields a better trial.\n")

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

    # ── STAGE 1: MRI DINOv2 ───────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1 — MRI DINOv2 ViT-B/14  [HPO Trial #1, val_acc=0.7692]")
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

    measure_inference_time(
        mri_model, mri_backbone[mri_backbone["split"] == "test"],
        CFG_SHARED["img_size"], "MRI_DINOv2")

    # ── STAGE 2: PET DINOv2 ───────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2 — PET DINOv2 ViT-B/14  [HPO Trial #14, val_acc=0.8753]")
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

    measure_inference_time(
        pet_model, pet_backbone[pet_backbone["split"] == "test"],
        CFG_SHARED["img_size"], "PET_DINOv2")

    # ── STAGE 3: Multimodal Fusion ────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3 — Multimodal Fusion  [HPO Trial #0 (best so far), val_acc=0.7126]")
    print(f"  DINOv2 backbone frozen — only projection heads + fusion MLP train")
    print("="*60)

    mri_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "mri_dinov2_best.pth"), map_location=DEVICE))
    pet_model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "pet_dinov2_best.pth"), map_location=DEVICE))
    for p in mri_model.backbone.parameters(): p.requires_grad = False
    for p in pet_model.backbone.parameters(): p.requires_grad = False

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
    print(f"\n  All models + plots saved to {OUT_DIR}")

    torch.save(mm_model.state_dict(),
               os.path.join(OUT_DIR, "multimodal_dinov2_final.pth"))


if __name__ == "__main__":
    main()
