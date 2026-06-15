"""Convert an existing finetuned-encoder checkpoint into the release format.

The paper's finetuned encoders come in a few shapes depending on the model:
  * full_vae            -- a full tokenizer/VAE state dict (e.g. VAR)
  * encoder_state_dict  -- a dict with an "encoder_state_dict" key (LlamaGen, Infinity)
  * delta               -- (finetuned - original) encoder tensors (RAR, Taming)

This script loads any of them, applies it to the model's encoder, and re-saves it
as the unified ``encoder_final.pth`` that ``--encoder finetuned`` expects, ready
to upload to HuggingFace with ``scripts/upload_checkpoints.py``.

Run in the model's own environment. Examples
--------
    python scripts/prep_release_checkpoint.py var --from <paper_full_vae>.pth \
        --format full_vae --set vae_path=<orig_vae>.pth
    python scripts/prep_release_checkpoint.py rar --from <delta>.pth --format delta \
        --set tokenizer_path=<tok>.bin
"""

from __future__ import annotations

import argparse
import os

import torch

from _common import build_from_args, add_common_args  # noqa: E402
from dataprov.config import checkpoints_dir  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--from", dest="src", required=True, help="path to the existing finetuned checkpoint")
    p.add_argument("--format", choices=["full_vae", "encoder_state_dict", "delta"], required=True)
    p.add_argument("--out", default=None, help="output path (default checkpoints/<model>/encoder_final.pth)")
    args = p.parse_args()

    args.encoder = "original"  # start from the original encoder, then apply the checkpoint
    model, _ = build_from_args(args, need_generator=False)
    enc = model.trainable_encoder_modules()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    if args.format == "full_vae":
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.tokenizer.load_state_dict(sd, strict=False)
    elif args.format == "encoder_state_dict":
        enc[0].load_state_dict(ckpt["encoder_state_dict"])
    else:  # delta
        delta = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        sd = enc[0].state_dict()
        applied = 0
        for k, v in delta.items():
            kk = k[len("encoder."):] if k.startswith("encoder.") else k
            if kk in sd:
                sd[kk] = sd[kk] + v.to(sd[kk].device)
                applied += 1
        enc[0].load_state_dict(sd)
        print(f"applied delta to {applied} encoder tensors")

    out = args.out or os.path.join(checkpoints_dir(), args.model, "encoder_final.pth")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(model.encoder_state_dict(), out)
    print(f"wrote release encoder -> {out}")


if __name__ == "__main__":
    main()
