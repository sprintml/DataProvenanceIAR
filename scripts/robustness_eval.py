"""Robustness of provenance signals to image post-processing.

Applies a post-processing attack (JPEG, blur, noise, brightness, contrast,
saturation, resize) to both belonging and non-belonging images, then reports
TPR@1%FPR. The paper uses QuantLoss as the most robust signal, especially when
the inverse decoder was finetuned with augmentations (``--encoder finetuned`` on
an augmentation-finetuned checkpoint).

Examples
--------
    # all attacks at the paper's default strengths
    python scripts/robustness_eval.py rar --signal quant_loss \
        --belonging data/rar_generated --nonbelonging data/imagenet

    # sweep one attack across its strength range
    python scripts/robustness_eval.py rar --attacks jpeg --strength sweep \
        --belonging data/rar_generated --nonbelonging data/imagenet
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from _common import build_from_args, add_common_args  # noqa: E402
from dataprov import augmentations as A
from dataprov import metrics, signals  # noqa: E402
from dataprov.data import list_images  # noqa: E402


def attacked_scores(model, folder, attack, strength, signal, batch_size, limit) -> np.ndarray:
    """Score a folder under one attack, faithfully to the paper (demo_ours.py):
    open -> resize((S, S)) -> attack on the PIL image -> ToTensor -> encode.
    """
    paths = list_images(folder)
    if limit is not None:
        paths = paths[:limit]
    size = model.image_size
    out: List[np.ndarray] = []
    batch: List[torch.Tensor] = []

    def flush() -> np.ndarray:
        native = model.to_model_range(torch.stack(batch).to(model.device))
        return signals.provenance_signals(model, native)[signal]

    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size))
        img = A.apply_attack_pil(img, attack, strength, size)
        batch.append(TF.to_tensor(img))
        if len(batch) == batch_size:
            out.append(flush())
            batch = []
    if batch:
        out.append(flush())
    return np.concatenate(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--belonging", required=True)
    p.add_argument("--nonbelonging", required=True)
    p.add_argument("--signal", default="quant_loss", choices=list(signals.SIGNAL_NAMES))
    p.add_argument("--attacks", nargs="+", default=list(A.ATTACK_DEFAULTS),
                   help="attacks to evaluate (default: all)")
    p.add_argument("--strength", default="default", choices=["default", "sweep"],
                   help="'default' = paper strength; 'sweep' = full range per attack")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model, cfg = build_from_args(args)

    results: List[tuple] = []
    print(f"\nTPR@1%FPR (%)  |  model={args.model}  signal={args.signal}  encoder={args.encoder}\n")
    print(f"{'attack':<14}{'strength':>10}{'TPR@1%FPR':>12}")
    print("-" * 36)
    for attack in args.attacks:
        strengths = A.ATTACK_RANGES[attack] if args.strength == "sweep" else [A.ATTACK_DEFAULTS[attack]]
        for s in strengths:
            bel = attacked_scores(model, args.belonging, attack, s, args.signal, args.batch_size, args.limit)
            non = attacked_scores(model, args.nonbelonging, attack, s, args.signal, args.batch_size, args.limit)
            tpr = 100.0 * metrics.tpr_at_fpr(bel, non, 0.01, members_lower=True)
            results.append((attack, s, tpr))
            print(f"{attack:<14}{str(s):>10}{tpr:>12.1f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["attack", "strength", "tpr_at_1fpr"])
            w.writerows(results)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
