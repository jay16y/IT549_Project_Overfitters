# pill_dataset_v2.py — SOTA Dataset for Pill Recognition
# Includes: CutMix, MixUp, class-balanced sampling, advanced augmentation

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import random
import os


# ══════════════════════════════════════════════════════
# ADVANCED AUGMENTATION TRANSFORMS
# ══════════════════════════════════════════════════════
def get_train_transform(img_size=224):
    """
    SOTA augmentation pipeline for fine-grained classification.
    DINOv2 uses 224 or 518 — we use 224 for speed.
    """
    return transforms.Compose([
        transforms.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        # ImageNet normalization (DINOv2 uses this)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def get_eval_transform(img_size=224):
    """Clean transform for validation/test/inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ══════════════════════════════════════════════════════
# CUTMIX & MIXUP — SOTA augmentation for fine-grained tasks
# ══════════════════════════════════════════════════════
def cutmix(images, labels, alpha=1.0):
    """
    CutMix augmentation — replaces a random rectangle in image A
    with content from image B. Labels are mixed proportionally.
    """
    batch_size = images.size(0)
    indices = torch.randperm(batch_size, device=images.device)

    lam = np.random.beta(alpha, alpha)

    H, W = images.shape[2], images.shape[3]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]

    # Adjust lambda based on actual cut area
    lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))

    return images, labels, labels[indices], lam


def mixup(images, labels, alpha=0.2):
    """
    MixUp augmentation — linear interpolation of two images.
    Better at generalization than just CutMix.
    """
    batch_size = images.size(0)
    indices = torch.randperm(batch_size, device=images.device)

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensures we keep more of the original

    mixed_images = lam * images + (1 - lam) * images[indices]
    return mixed_images, labels, labels[indices], lam


# ══════════════════════════════════════════════════════
# DATASET CLASSES
# ══════════════════════════════════════════════════════
class PillDatasetV2(Dataset):
    """
    Pill dataset with optional class-balanced sampling weights.
    Returns (image, pill_id) for ArcFace training.
    """

    def __init__(self, csv_path, split="train", img_size=224):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.img_size = img_size

        if split == "train":
            self.transform = get_train_transform(img_size)
        else:
            self.transform = get_eval_transform(img_size)

        self.labels = self.df["pill_id"].values
        self.num_classes = self.df["pill_id"].nunique()

        # For class-balanced sampling
        self.class_counts = self.df["pill_id"].value_counts().to_dict()

    def get_sampler_weights(self):
        """
        Returns weights for WeightedRandomSampler.
        Underrepresented pills get higher weights → balanced training.
        """
        weights = []
        for label in self.labels:
            weights.append(1.0 / self.class_counts[label])
        return torch.DoubleTensor(weights)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            img = Image.open(row["img_path"]).convert("RGB")
        except Exception as e:
            # Failed to load — use random other image
            new_idx = random.randint(0, len(self.df) - 1)
            return self.__getitem__(new_idx)

        img = self.transform(img)
        return img, int(row["pill_id"])


class ReferenceDatasetV2(Dataset):
    """Dataset for reference images — used for FAISS index building."""

    def __init__(self, csv_path, img_size=224):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.transform = get_eval_transform(img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            img = Image.open(row["img_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size))

        img = self.transform(img)
        return img, int(row["pill_id"]), str(row["ndc_clean"])


# ══════════════════════════════════════════════════════
# DATALOADER FACTORY
# ══════════════════════════════════════════════════════
def get_dataloaders_v2(consumer_csv, reference_csv, batch_size=64,
                       img_size=224, num_workers=8, balanced_sampling=True):
    """
    Build train/val/test/reference dataloaders.

    Args:
        balanced_sampling: If True, uses WeightedRandomSampler for training.
                          Helps when some pills have many more images than others.
    """
    print("📦 Building datasets...")

    train_set = PillDatasetV2(consumer_csv, "train", img_size)
    val_set   = PillDatasetV2(consumer_csv, "val",   img_size)
    test_set  = PillDatasetV2(consumer_csv, "test",  img_size)
    ref_set   = ReferenceDatasetV2(reference_csv, img_size)

    # Sampler for training
    if balanced_sampling:
        sampler = WeightedRandomSampler(
            weights=train_set.get_sampler_weights(),
            num_samples=len(train_set),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(num_workers > 0),
        ),
        "val": DataLoader(
            val_set,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        ),
        "reference": DataLoader(
            ref_set,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        ),
    }

    print(f"  Train     : {len(train_set):>6} images, {len(loaders['train']):>4} batches")
    print(f"  Val       : {len(val_set):>6} images, {len(loaders['val']):>4} batches")
    print(f"  Test      : {len(test_set):>6} images, {len(loaders['test']):>4} batches")
    print(f"  Reference : {len(ref_set):>6} images, {len(loaders['reference']):>4} batches")
    print(f"  Sampling  : {'class-balanced' if balanced_sampling else 'random'}")

    return loaders, train_set.num_classes


if __name__ == "__main__":
    loaders, num_classes = get_dataloaders_v2(
        "data/consumer_mapping.csv",
        "data/reference_mapping.csv",
        batch_size=32,
        num_workers=0,
    )
    print(f"\nNum classes: {num_classes}")

    for imgs, labels in loaders["train"]:
        print(f"Batch shape: {imgs.shape}, labels: {labels[:5]}")

        # Test CutMix
        mixed, l1, l2, lam = cutmix(imgs, labels)
        print(f"CutMix: lambda={lam:.3f}")

        # Test MixUp
        mixed, l1, l2, lam = mixup(imgs, labels)
        print(f"MixUp:  lambda={lam:.3f}")
        break
