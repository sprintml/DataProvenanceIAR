"""Download base model weights and/or finetuned inverse-decoder encoders.

Base weights for VAR, RAR and LlamaGen download automatically from their
official HuggingFace repos the first time you build the model; this script just
triggers that. Taming and Infinity weights must be fetched manually -- see
docs/MODELS.md.

Finetuned encoders (our released inverse decoders) are pulled from a single
HuggingFace model repo with layout ``<model>/encoder_final.pth``.

Examples
--------
    python scripts/download_checkpoints.py var --what base
    python scripts/download_checkpoints.py var --what encoder \
        --hf-repo <user>/dataprovenance-iar-encoders
"""

from __future__ import annotations

import argparse
import os

import torch

from _common import add_common_args  # noqa: E402
from dataprov import build_model  # noqa: E402
from dataprov.config import checkpoints_dir, load_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--what", choices=["base", "encoder", "all"], default="all")
    p.add_argument("--hf-repo", default=None, help="HF repo id holding finetuned encoders")
    p.add_argument("--revision", default="main")
    args = p.parse_args()
    cfg = load_config(args.model, args.overrides)

    if args.what in ("base", "all"):
        print(f"[base] building {args.model} to trigger base-weight download ...")
        build_model(args.model, cfg, device="cpu")
        print("[base] done")

    if args.what in ("encoder", "all"):
        repo = args.hf_repo or cfg.get("hf_encoder_repo", None)
        if not repo:
            raise SystemExit("Provide --hf-repo or set hf_encoder_repo in the config.")
        from huggingface_hub import hf_hub_download

        dst = os.path.join(checkpoints_dir(), args.model)
        os.makedirs(dst, exist_ok=True)
        path = hf_hub_download(
            repo_id=repo, filename=f"{args.model}/encoder_final.pth",
            revision=args.revision, local_dir=dst,
        )
        # normalize to checkpoints/<model>/encoder_final.pth
        final = os.path.join(dst, "encoder_final.pth")
        if os.path.abspath(path) != os.path.abspath(final):
            torch.save(torch.load(path, map_location="cpu"), final)
        print(f"[encoder] downloaded -> {final}")


if __name__ == "__main__":
    main()
