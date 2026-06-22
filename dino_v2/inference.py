import os
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import OUT_DIR
from data import get_transforms


@torch.no_grad()
def measure_inference_time(model, test_df, img_size, name):
    """Per-subject inference timing on GPU (if available) and CPU."""
    print(f"\n{'='*70}")
    print(f"  INFERENCE TIMING — {name}")
    print(f"{'='*70}")

    transform = get_transforms(img_size, split="val")
    subjects  = sorted(test_df["subject_id"].unique())
    has_gpu   = torch.cuda.is_available()

    subject_data = {}
    for subj in subjects:
        subj_df = test_df[test_df["subject_id"] == subj]
        imgs = [transform(Image.open(row["slice_path"]).convert("RGB"))
                for _, row in subj_df.iterrows()]
        subject_data[subj] = torch.stack(imgs)

    results   = []
    gpu_times = {}
    cpu_times = {}

    if has_gpu:
        gpu_device = torch.device("cuda")
        model_gpu  = model.to(gpu_device)
        model_gpu.eval()
        _ = model_gpu(subject_data[subjects[0]].to(gpu_device))
        torch.cuda.synchronize()

        for subj in subjects:
            batch = subject_data[subj].to(gpu_device)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); _ = model_gpu(batch); e.record()
            torch.cuda.synchronize()
            gpu_times[subj] = s.elapsed_time(e)

    cpu_device = torch.device("cpu")
    model_cpu  = model.to(cpu_device)
    model_cpu.eval()
    _ = model_cpu(subject_data[subjects[0]])

    for subj in subjects:
        batch = subject_data[subj]
        t0 = time.perf_counter()
        _ = model_cpu(batch)
        cpu_times[subj] = (time.perf_counter() - t0) * 1000.0

    print(f"  {'Subject':<18} | {'#Slices':>7} | {'GPU (ms)':>10} | {'CPU (ms)':>10}")
    print(f"  {'-'*57}")
    for subj in subjects:
        n_slices = len(subject_data[subj])
        gpu_ms   = gpu_times.get(subj, float("nan"))
        cpu_ms   = cpu_times[subj]
        gpu_str  = f"{gpu_ms:>10.2f}" if has_gpu else f"{'N/A':>10}"
        print(f"  {subj:<18} | {n_slices:>7} | {gpu_str} | {cpu_ms:>10.2f}")
        results.append({"subject": subj, "n_slices": n_slices,
                        "gpu_ms": gpu_ms if has_gpu else None, "cpu_ms": cpu_ms})

    avg_slices = np.mean([r["n_slices"] for r in results])
    avg_cpu    = np.mean([r["cpu_ms"]   for r in results])
    std_cpu    = np.std( [r["cpu_ms"]   for r in results])
    print(f"  {'-'*57}")
    if has_gpu:
        avg_gpu = np.mean([r["gpu_ms"] for r in results])
        std_gpu = np.std( [r["gpu_ms"] for r in results])
        print(f"  {'MEAN':<18} | {avg_slices:>7.1f} | {avg_gpu:>10.2f} | {avg_cpu:>10.2f}")
        print(f"  {'STD':<18} | {'':>7} | {std_gpu:>10.2f} | {std_cpu:>10.2f}")
    else:
        print(f"  {'MEAN':<18} | {avg_slices:>7.1f} | {'N/A':>10} | {avg_cpu:>10.2f}")
        print(f"  {'STD':<18} | {'':>7} | {'N/A':>10} | {std_cpu:>10.2f}")

    csv_path = os.path.join(OUT_DIR, f"{name}_inference_times.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    if has_gpu:
        model.to(torch.device("cuda"))
    return results


@torch.no_grad()
def measure_inference_time_multimodal(model, mri_test_df, pet_test_df, img_size, name):
    """Per-subject inference timing for the multimodal fusion model."""
    print(f"\n{'='*70}")
    print(f"  INFERENCE TIMING — {name} (Multimodal)")
    print(f"{'='*70}")

    transform = get_transforms(img_size, split="val")
    common    = sorted(set(mri_test_df["subject_id"]) & set(pet_test_df["subject_id"]))
    has_gpu   = torch.cuda.is_available()

    subject_data = {}
    for subj in common:
        mri_paths = mri_test_df[mri_test_df["subject_id"] == subj]["slice_path"].tolist()
        pet_paths = pet_test_df[pet_test_df["subject_id"] == subj]["slice_path"].tolist()
        mri_imgs, pet_imgs = [], []
        for mp in mri_paths:
            for pp in pet_paths:
                mri_imgs.append(transform(Image.open(mp).convert("RGB")))
                pet_imgs.append(transform(Image.open(pp).convert("RGB")))
        subject_data[subj] = (torch.stack(mri_imgs), torch.stack(pet_imgs))

    results   = []
    gpu_times = {}
    cpu_times = {}

    if has_gpu:
        gpu_device = torch.device("cuda")
        model_gpu  = model.to(gpu_device)
        model_gpu.eval()
        mri_d, pet_d = subject_data[common[0]]
        _ = model_gpu(mri_d.to(gpu_device), pet_d.to(gpu_device))
        torch.cuda.synchronize()

        for subj in common:
            mri_b, pet_b = subject_data[subj]
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = model_gpu(mri_b.to(gpu_device), pet_b.to(gpu_device))
            e.record()
            torch.cuda.synchronize()
            gpu_times[subj] = s.elapsed_time(e)

    cpu_device = torch.device("cpu")
    model_cpu  = model.to(cpu_device)
    model_cpu.eval()
    mri_d, pet_d = subject_data[common[0]]
    _ = model_cpu(mri_d, pet_d)

    for subj in common:
        mri_b, pet_b = subject_data[subj]
        t0 = time.perf_counter()
        _ = model_cpu(mri_b, pet_b)
        cpu_times[subj] = (time.perf_counter() - t0) * 1000.0

    print(f"  {'Subject':<18} | {'#Pairs':>7} | {'GPU (ms)':>10} | {'CPU (ms)':>10}")
    print(f"  {'-'*57}")
    for subj in common:
        n_pairs = len(subject_data[subj][0])
        gpu_ms  = gpu_times.get(subj, float("nan"))
        cpu_ms  = cpu_times[subj]
        gpu_str = f"{gpu_ms:>10.2f}" if has_gpu else f"{'N/A':>10}"
        print(f"  {subj:<18} | {n_pairs:>7} | {gpu_str} | {cpu_ms:>10.2f}")
        results.append({"subject": subj, "n_pairs": n_pairs,
                        "gpu_ms": gpu_ms if has_gpu else None, "cpu_ms": cpu_ms})

    avg_pairs = np.mean([r["n_pairs"] for r in results])
    avg_cpu   = np.mean([r["cpu_ms"]  for r in results])
    std_cpu   = np.std( [r["cpu_ms"]  for r in results])
    print(f"  {'-'*57}")
    if has_gpu:
        avg_gpu = np.mean([r["gpu_ms"] for r in results])
        std_gpu = np.std( [r["gpu_ms"] for r in results])
        print(f"  {'MEAN':<18} | {avg_pairs:>7.1f} | {avg_gpu:>10.2f} | {avg_cpu:>10.2f}")
        print(f"  {'STD':<18} | {'':>7} | {std_gpu:>10.2f} | {std_cpu:>10.2f}")
    else:
        print(f"  {'MEAN':<18} | {avg_pairs:>7.1f} | {'N/A':>10} | {avg_cpu:>10.2f}")
        print(f"  {'STD':<18} | {'':>7} | {'N/A':>10} | {std_cpu:>10.2f}")

    csv_path = os.path.join(OUT_DIR, f"{name}_inference_times.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    if has_gpu:
        model.to(torch.device("cuda"))
    return results
