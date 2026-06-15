"""Image post-processing transforms used for robustness.

These serve two purposes:

1. **Robustness evaluation** (``apply_attack``): a single transform at a fixed
   strength is applied to a suspect image before computing provenance signals.
2. **Robustness finetuning** (``sample_train_augmentation``): for RAR and Taming
   we optionally finetune the inverse decoder while applying a progressively
   stronger schedule of these transforms, which makes the resulting signals far
   more robust to post-processing (see the paper's robustness section).

All transforms operate on float tensors in ``[0, 1]`` with shape ``(C, H, W)``
or ``(B, C, H, W)`` and return the same shape/range. The default attack
strengths (``ATTACK_DEFAULTS``) and the sweep ranges (``ATTACK_RANGES``) match
the values reported in the paper.
"""

from __future__ import annotations

import io
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image

__all__ = [
    "jpeg",
    "gaussian_blur",
    "gaussian_noise",
    "brightness",
    "contrast",
    "saturation",
    "resize_back",
    "ATTACKS",
    "ATTACK_DEFAULTS",
    "ATTACK_RANGES",
    "apply_attack",
    "AUG_SCHEDULE",
    "sample_train_augmentation",
]


# --------------------------------------------------------------------------- #
# Individual transforms (single image, C,H,W, float in [0, 1]).
# --------------------------------------------------------------------------- #
def _to_pil(img: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(img.detach().cpu().clamp(0, 1))


def _jpeg_single(img: torch.Tensor, quality: int) -> torch.Tensor:
    buffer = io.BytesIO()
    _to_pil(img).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return TF.to_tensor(Image.open(buffer).convert("RGB")).to(img.device)


def _batched(fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
    """Apply a per-image transform to a single image or a batch."""

    def wrapper(img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if img.dim() == 4:
            return torch.stack([wrapper(img[i], *args, **kwargs) for i in range(img.shape[0])])
        return fn(img, *args, **kwargs)

    return wrapper


@_batched
def jpeg(img: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG compression to the given quality (1-100; lower = stronger)."""
    return _jpeg_single(img.clamp(0, 1), quality).clamp(0, 1)


@_batched
def gaussian_blur(img: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Gaussian blur with an odd kernel size (0 = identity)."""
    kernel_size = int(kernel_size)
    if kernel_size <= 0:
        return img
    if kernel_size % 2 == 0:
        kernel_size += 1
    return TF.gaussian_blur(img, kernel_size).clamp(0, 1)


@_batched
def gaussian_noise(img: torch.Tensor, std: float) -> torch.Tensor:
    """Additive Gaussian noise with the given standard deviation (in [0, 1])."""
    return (img + torch.randn_like(img) * float(std)).clamp(0, 1)


@_batched
def brightness(img: torch.Tensor, factor: float) -> torch.Tensor:
    return TF.adjust_brightness(img, float(factor)).clamp(0, 1)


@_batched
def contrast(img: torch.Tensor, factor: float) -> torch.Tensor:
    return TF.adjust_contrast(img, float(factor)).clamp(0, 1)


@_batched
def saturation(img: torch.Tensor, factor: float) -> torch.Tensor:
    return TF.adjust_saturation(img, float(factor)).clamp(0, 1)


@_batched
def resize_back(img: torch.Tensor, ratio: float) -> torch.Tensor:
    """Downscale by ``ratio`` then upscale back to the original resolution."""
    h, w = img.shape[-2:]
    new_h, new_w = max(1, int(h * ratio)), max(1, int(w * ratio))
    img = TF.resize(img, [new_h, new_w], antialias=True)
    img = TF.resize(img, [h, w], antialias=True)
    return img.clamp(0, 1)


# Registry keyed by the attack names used in the paper/tables.
ATTACKS: Dict[str, Callable[..., torch.Tensor]] = {
    "noise": gaussian_noise,
    "kernel": gaussian_blur,
    "jpeg": jpeg,
    "brightness": brightness,
    "contrast": contrast,
    "saturation": saturation,
    "resize": resize_back,
}

# Default strengths used for the main robustness table.
ATTACK_DEFAULTS: Dict[str, float] = {
    "noise": 0.05,
    "kernel": 9,
    "jpeg": 60,
    "brightness": 1.6,
    "contrast": 2.0,
    "saturation": 2.0,
    "resize": 0.5,
}

# Strength sweeps used for the extended robustness analysis.
ATTACK_RANGES: Dict[str, List[float]] = {
    "kernel": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    "noise": [0.0, 0.05, 0.1, 0.15, 0.2],
    "resize": [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
    "jpeg": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
    "brightness": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    "saturation": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
    "contrast": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
}


def apply_attack(img01: torch.Tensor, name: str, strength: float) -> torch.Tensor:
    """Apply a single post-processing attack at a fixed strength to ``[0,1]`` images."""
    if name == "none":
        return img01
    if name not in ATTACKS:
        raise ValueError(f"Unknown attack '{name}'. Options: {list(ATTACKS)}")
    return ATTACKS[name](img01, strength)


# --------------------------------------------------------------------------- #
# Progressive augmentation schedule for robustness finetuning (RAR / Taming).
# Stages are applied over the finetuning epochs (weak -> medium -> strong),
# matching the recipe in the paper.
# --------------------------------------------------------------------------- #
AUG_SCHEDULE: Dict[str, Dict[str, List[float]]] = {
    "weak": {
        "jpeg": [90, 85, 80],
        "kernel": [1, 3],
        "noise": [0.005, 0.01, 0.02],
        "brightness": [1.0, 1.1, 1.2],
        "saturation": [1.0, 1.2, 1.5],
        "resize": [0.9, 0.85, 0.8],
        "contrast": [1.0, 1.2, 1.5],
    },
    "medium": {
        "jpeg": [80, 75, 70, 65],
        "kernel": [3, 5],
        "noise": [0.02, 0.03, 0.04],
        "brightness": [1.3, 1.4, 1.5],
        "saturation": [1.5, 1.7, 2.0],
        "resize": [0.8, 0.75, 0.7],
        "contrast": [1.5, 1.7, 2.0],
    },
    "strong": {
        "jpeg": [60, 55, 50],
        "kernel": [5, 7, 9],
        "noise": [0.03, 0.04, 0.05],
        "brightness": [1.5, 1.7, 2.0],
        "saturation": [2.0, 2.2, 2.5],
        "resize": [0.7, 0.6, 0.5],
        "contrast": [2.0, 2.2, 2.4],
    },
}


def stage_for_epoch(epoch: int, schedule: List[Tuple[str, int]]) -> Optional[str]:
    """Map a 0-indexed epoch to an augmentation stage given a schedule.

    ``schedule`` is a list of ``(stage_name, num_epochs)`` pairs, e.g.
    ``[("none", 5), ("weak", 5), ("medium", 20), ("strong", 20)]``. Returns the
    stage name (or ``None`` for the no-augmentation warmup).
    """
    cursor = 0
    for stage, n in schedule:
        if epoch < cursor + n:
            return None if stage == "none" else stage
        cursor += n
    return schedule[-1][0] if schedule else None


def sample_train_augmentation(
    img01: torch.Tensor, stage: Optional[str], p: float, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Randomly apply one transform from ``stage`` with probability ``p``.

    Used during robustness finetuning. ``stage`` is one of ``AUG_SCHEDULE`` keys
    (or ``None`` for no augmentation).
    """
    if stage is None:
        return img01

    def rand() -> float:
        return torch.rand(1, generator=generator).item()

    if rand() > p:
        return img01

    stage_cfg = AUG_SCHEDULE[stage]
    names = list(stage_cfg.keys())
    name = names[int(rand() * len(names)) % len(names)]
    choices = stage_cfg[name]
    strength = choices[int(rand() * len(choices)) % len(choices)]
    return apply_attack(img01, name, strength)
