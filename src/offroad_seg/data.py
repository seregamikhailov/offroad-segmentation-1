"""Датасет сегментации: списки «кадр<TAB>маска» (см. scripts/prepare_rugd.py)."""
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_classes(root):
    meta = json.loads((Path(root) / "classes.json").read_text())
    return meta["classes"], meta["ignore_index"]


def build_transforms(size, train):
    h, w = size
    norm = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    if not train:
        return A.Compose([A.Resize(h, w)] + norm)
    return A.Compose([
        A.RandomResizedCrop(size=(h, w), scale=(0.4, 1.0), ratio=(1.0, 1.6)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03, p=0.8),
        A.GaussNoise(p=0.2),
    ] + norm)


class SegDataset(Dataset):
    def __init__(self, root, split, size, train=False, limit=None, stride=1):
        lines = (Path(root) / "splits" / f"{split}.txt").read_text().split("\n")
        self.pairs = [ln.split("\t") for ln in lines if ln.strip()][::max(1, stride)]
        if limit:
            self.pairs = self.pairs[:limit]
        self.tf = build_transforms(size, train)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_p, mask_p = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(img_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].long()


def class_frequencies(root, split="train", num_classes=None, ignore_index=255, max_files=800):
    """Доля пикселей каждого класса (по подвыборке) — для весов лосса и анализа дисбаланса."""
    lines = (Path(root) / "splits" / f"{split}.txt").read_text().split("\n")
    masks = [ln.split("\t")[1] for ln in lines if ln.strip()]
    step = max(1, len(masks) // max_files)
    counts = np.zeros(num_classes, dtype=np.int64)
    for p in masks[::step]:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        m = m[m != ignore_index]
        counts += np.bincount(m.ravel(), minlength=num_classes)[:num_classes]
    return counts / max(counts.sum(), 1)
