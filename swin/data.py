import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import LABEL_MAP


def scan_modality(root_dir):
    """Scans root_dir/AD/<subject>/*.png and root_dir/CN/<subject>/*.png."""
    records = []
    for group in ["AD", "CN"]:
        group_dir = os.path.join(root_dir, group)
        if not os.path.isdir(group_dir):
            print(f"  WARNING: {group_dir} not found")
            continue
        for subject in sorted(os.listdir(group_dir)):
            subj_dir = os.path.join(group_dir, subject)
            if not os.path.isdir(subj_dir):
                continue
            for fname in sorted(os.listdir(subj_dir)):
                if fname.lower().endswith(".png"):
                    records.append({
                        "subject_id": subject,
                        "group":      group,
                        "slice_path": os.path.join(subj_dir, fname),
                    })
    df = pd.DataFrame(records)
    print(f"  {root_dir.split('/')[-1]}: "
          f"{df['subject_id'].nunique()} subjects | "
          f"{len(df)} slices | "
          f"{df['group'].value_counts().to_dict()}")
    return df


def get_transforms(img_size, split="train"):
    """Swin V2-B was pretrained at 256×256 with ImageNet-1K stats."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class SliceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["slice_path"]).convert("RGB")
        label = LABEL_MAP[row["group"]]
        if self.transform:
            img = self.transform(img)
        return img, label, row["subject_id"]


class MultimodalDataset(Dataset):
    """Cartesian-product MRI×PET pairs for overlap subjects."""
    def __init__(self, mri_df, pet_df, transform=None):
        self.transform = transform
        pairs  = []
        common = set(mri_df["subject_id"]) & set(pet_df["subject_id"])
        for subj in sorted(common):
            mri_slices = mri_df[mri_df["subject_id"] == subj]["slice_path"].tolist()
            pet_slices = pet_df[pet_df["subject_id"] == subj]["slice_path"].tolist()
            group      = mri_df[mri_df["subject_id"] == subj]["group"].iloc[0]
            for m in mri_slices:
                for p in pet_slices:
                    pairs.append((m, p, group, subj))
        self.pairs = pairs
        print(f"    {len(common)} subjects | {len(pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mri_path, pet_path, group, subj = self.pairs[idx]
        label   = LABEL_MAP[group]
        mri_img = Image.open(mri_path).convert("RGB")
        pet_img = Image.open(pet_path).convert("RGB")
        if self.transform:
            mri_img = self.transform(mri_img)
            pet_img = self.transform(pet_img)
        return mri_img, pet_img, label, subj


def make_single_loaders_backbone(df, batch_size, img_size, name):
    loaders = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        ds  = SliceDataset(sub, transform=get_transforms(img_size, split if split != "test" else "val"))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2, pin_memory=True,
        )
        print(f"    [{name}][{split}] "
              f"{sub['subject_id'].nunique()} subjects | {len(sub)} slices")
    return loaders


def make_multimodal_loaders(mri_df, pet_df, batch_size, img_size):
    loaders = {}
    for split in ["train", "val", "test"]:
        m  = mri_df[mri_df["split"] == split]
        p  = pet_df[pet_df["split"] == split]
        print(f"  [{split}] ", end="")
        ds = MultimodalDataset(m, p, transform=get_transforms(img_size, split if split != "test" else "val"))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2, pin_memory=True,
        )
    return loaders
