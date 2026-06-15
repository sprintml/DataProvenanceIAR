"""Finetune a model's encoder into an inverse decoder (key experiment #1).

Starts from the model's *original* encoder and trains it (decoder + codebook
frozen) so that re-encoding a generated image recovers the feature map the image
was generated from. The result is saved to
``checkpoints/<model>/encoder_final.pth`` by default.

Examples
--------
    # standard finetuning (hyper-parameters come from configs/<model>.yaml)
    python scripts/finetune_encoder.py rar --data data/rar_generated

    # robustness finetuning with the progressive augmentation schedule (RAR/Taming)
    python scripts/finetune_encoder.py rar --data data/rar_generated --augment \
        --out checkpoints/rar_aug

    # override any finetuning hyper-parameter
    python scripts/finetune_encoder.py var --data data/var_generated \
        --set finetune.epochs=10 finetune.lr=5e-5
"""

from __future__ import annotations

import argparse
import os

from _common import build_from_args, add_common_args  # noqa: E402
from dataprov.config import checkpoints_dir  # noqa: E402
from dataprov.finetune import finetune_inverse_decoder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--data", required=True, help="directory of generated images + *_target.pt")
    p.add_argument("--out", default=None, help="output dir (default checkpoints/<model>)")
    p.add_argument("--augment", action="store_true", help="enable robustness augmentation schedule")
    p.add_argument("--limit", type=int, default=None, help="cap number of finetuning samples")
    args = p.parse_args()

    # Finetuning always starts from the original encoder.
    args.encoder = "original"
    model, cfg = build_from_args(args)

    ft_cfg = cfg.get("finetune", {})
    out_dir = args.out or os.path.join(checkpoints_dir(), args.model)
    aug_schedule = None
    if args.augment:
        sched = ft_cfg.get("aug_schedule", None)
        if sched is not None:
            aug_schedule = [(s[0], int(s[1])) for s in sched]

    path = finetune_inverse_decoder(
        model, args.data, out_dir, ft_cfg,
        augment=args.augment, aug_schedule=aug_schedule, limit=args.limit,
    )
    print(f"\nfinetuned inverse decoder saved to: {path}")


if __name__ == "__main__":
    main()
