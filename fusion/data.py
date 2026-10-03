"""
data.py — Data loading, feature caching, and tensor construction.

cache_features() extracts and saves backbone features per subject to .npz.
build_tensors() concatenates MRI+PET feature vectors into training tensors.
"""
import os, gc
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import DEVICE, LABEL_MAP, CACHE_DIR, OLD_PFX, NEW_PFX, AUG_VIEWS
from .models import Extractor, get_transforms


def load_df(csv_path):
    df = pd.read_csv(csv_path)
    df["slice_path"] = df["slice_path"].str.replace(OLD_PFX, NEW_PFX, regex=False)
    if "subject_id" not in df.columns and "subject" in df.columns:
        df = df.rename(columns={"subject": "subject_id"})
    return df


@torch.no_grad()
def cache_features(df, extractor, transforms_list, cache_path):
    if os.path.exists(cache_path):
        print(f"    [cache hit] {os.path.basename(cache_path)}")
        return np.load(cache_path, allow_pickle=True)["data"].item()
    print(f"    [extracting] {os.path.basename(cache_path)}")
    extractor.to(DEVICE).eval()
    result = {}; first_err = [False]
    for subj, grp in df.groupby("subject_id"):
        label = LABEL_MAP[grp.group.iloc[0]]
        views = []
        for tf in transforms_list:
            vecs = []
            for p in grp.slice_path.tolist():
                try:
                    img = Image.open(p).convert("RGB")
                    f   = extractor(tf(img).unsqueeze(0).to(DEVICE)).squeeze(0).cpu().float().numpy()
                    vecs.append(f)
                except Exception as e:
                    if not first_err[0]:
                        print(f"    [WARNING] {type(e).__name__}: {e}")
                        first_err[0] = True
            if vecs:
                views.append(np.mean(vecs, axis=0))
        if views:
            result[subj] = (np.stack(views), label)
    extractor.cpu(); gc.collect(); torch.cuda.empty_cache()
    np.savez_compressed(cache_path, data=result)
    print(f"    cached {len(result)} subjects")
    if len(result) == 0:
        raise RuntimeError(f"0 subjects cached for {os.path.basename(cache_path)}")
    return result


def get_arch_cache(arch, modality, img_size, ckpt, df, aug):
    path = f"{CACHE_DIR}/{arch}_{modality}_{'aug' if aug else 'det'}.npz"
    ext  = Extractor(arch, ckpt, img_size=img_size)
    tfs  = [get_transforms(img_size, aug)] * (AUG_VIEWS if aug else 1)
    cache = cache_features(df, ext, tfs, path)
    del ext; gc.collect(); torch.cuda.empty_cache()
    return cache


def build_tensors(mri_cache, pet_cache, subjects, use_aug):
    X, y = [], []
    for s in subjects:
        if s not in mri_cache or s not in pet_cache:
            continue
        mri_views, label = mri_cache[s]; pet_views, _ = pet_cache[s]
        n = min(len(mri_views), len(pet_views))
        if use_aug:
            for v in range(n):
                X.append(np.concatenate([mri_views[v], pet_views[v]])); y.append(label)
        else:
            X.append(np.concatenate([mri_views[0], pet_views[0]])); y.append(label)
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.long))
