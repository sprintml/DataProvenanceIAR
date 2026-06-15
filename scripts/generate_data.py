"""Generate belonging + finetuning data with a target IAR.

Saves, for each sample, ``{i:06d}.png`` (the generated image) and
``{i:06d}_target.pt`` (the quantized feature map ``f_Z`` the tokens were sampled
from, used as the inverse-decoder finetuning target).

Examples
--------
    # 1000 images to use both as the belonging eval set and finetuning data
    python scripts/generate_data.py var --n 1000 --out data/var_generated

    # smaller set, custom sampling
    python scripts/generate_data.py rar --n 200 --set cfg_scale=4.0 top_k=900
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image
from tqdm import tqdm

from _common import build_from_args, add_common_args  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--n", type=int, default=1000, help="number of images to generate")
    p.add_argument("--batch-size", type=int, default=None, help="override generation batch size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--no-targets", action="store_true", help="save only images (no *_target.pt)")
    args = p.parse_args()

    # Generation always uses the original tokenizer + the AR model.
    args.encoder = "original"
    model, cfg = build_from_args(args, need_generator=True)
    if model.generator is None:
        raise RuntimeError(f"{args.model} has no AR generator wired; cannot generate.")

    os.makedirs(args.out, exist_ok=True)
    bs = args.batch_size or int(cfg.get("batch_size", 8))
    gen_kwargs = {
        k: cfg[k] for k in ("cfg_scale", "top_k", "top_p", "temperature") if k in cfg
    }

    i = 0
    with tqdm(total=args.n, desc=f"generating {args.model}") as bar:
        for images01, targets in model.generate(args.n, bs, args.seed, **gen_kwargs):
            images01 = images01.clamp(0, 1).cpu()
            for b in range(images01.shape[0]):
                if i >= args.n:
                    break
                arr = (images01[b].permute(1, 2, 0).numpy() * 255).astype("uint8")
                Image.fromarray(arr).save(os.path.join(args.out, f"{i:06d}.png"))
                if not args.no_targets and targets is not None:
                    torch.save(targets[b].cpu(), os.path.join(args.out, f"{i:06d}_target.pt"))
                i += 1
                bar.update(1)
            if i >= args.n:
                break
    print(f"wrote {i} images to {args.out}")


if __name__ == "__main__":
    main()
