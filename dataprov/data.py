"""Datasets for evaluation and inverse-decoder finetuning.

Images are always handled in ``[0, 1]`` at the dataset level; conversion to a
model's native value range (and any robustness attack) happens in the
evaluation / finetuning loops. This keeps a single, attack-friendly convention.
"""

from __future__ import annotations

import glob
import os
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["ImageFolder", "GeneratedTargetDataset", "list_images"]

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_images(folder: str) -> List[str]:
    files: List[str] = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(files))


class ImageFolder(Dataset):
    """A flat folder of images, returned as ``[0, 1]`` tensors.

    Used for both belonging (generated) and non-belonging (natural / other
    model) evaluation sets.
    """

    def __init__(self, folder: str, transform: Callable, limit: Optional[int] = None):
        self.paths = list_images(folder)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {folder}")
        if limit is not None:
            self.paths = self.paths[:limit]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class GeneratedTargetDataset(Dataset):
    """Pairs of (generated image, finetuning target feature map).

    Produced by ``scripts/generate_data.py``: for every generated image
    ``{i:06d}.png`` there is a target tensor ``{i:06d}_target.pt`` holding the
    quantized feature map ``f_Z`` the tokens were sampled from. The finetuning
    objective trains the encoder so that ``encode_feature(image) ~= f_Z``.
    """

    def __init__(self, folder: str, transform: Callable, limit: Optional[int] = None):
        imgs = list_images(folder)
        self.pairs: List[Tuple[str, str]] = []
        for p in imgs:
            target = os.path.splitext(p)[0] + "_target.pt"
            if os.path.exists(target):
                self.pairs.append((p, target))
        if not self.pairs:
            raise FileNotFoundError(
                f"No (image, *_target.pt) pairs found in {folder}. "
                "Run scripts/generate_data.py first."
            )
        if limit is not None:
            self.pairs = self.pairs[:limit]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, target_path = self.pairs[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        target = torch.load(target_path, map_location="cpu")
        return img, target
