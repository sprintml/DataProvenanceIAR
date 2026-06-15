"""Shared CLI plumbing for the scripts."""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Make `import dataprov` work when running scripts directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprov import build_model  # noqa: E402
from dataprov.config import checkpoints_dir, load_config  # noqa: E402
from dataprov.models import MODEL_NAMES  # noqa: E402


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", choices=MODEL_NAMES, help="target IAR model")
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[],
        metavar="key=value", help="config overrides, e.g. --set finetune.epochs=5",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--encoder", choices=["original", "finetuned"], default="finetuned",
        help="'original' reproduces the Reconstruction/AEDR baselines; "
             "'finetuned' uses the inverse decoder (our method)",
    )


def finetuned_encoder_path(cfg, model_name: str) -> str:
    """Where this model's finetuned encoder lives (config override or default)."""
    p = cfg.get("finetuned_encoder", None)
    if p:
        return p
    return os.path.join(checkpoints_dir(), model_name, "encoder_final.pth")


def build_from_args(args, need_generator: bool = False):
    """Load config, build the model, and apply the encoder choice.

    ``need_generator`` controls whether the (large) AR transformer is loaded;
    only data generation needs it.
    """
    cfg = load_config(args.model, args.overrides)
    if "load_generator" not in cfg:
        cfg["load_generator"] = need_generator
    model = build_model(args.model, cfg, device=args.device)
    model.eval()

    if args.encoder == "finetuned":
        path = finetuned_encoder_path(cfg, args.model)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Finetuned encoder not found at {path}.\n"
                f"Either finetune it (scripts/finetune_encoder.py), download it "
                f"(scripts/download_checkpoints.py), or pass --encoder original."
            )
        sd = torch.load(path, map_location="cpu")
        model.load_encoder_state_dict(sd)
        print(f"loaded finetuned encoder: {path}")
    else:
        print("using the model's ORIGINAL encoder (baseline)")

    return model, cfg
