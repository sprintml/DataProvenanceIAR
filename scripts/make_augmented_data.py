"""Create on-disk post-processed copies of an image folder for robustness.

For each requested attack/strength, writes a sibling folder named
``<input>_<attack><strength>`` containing the attacked images. Useful to
pre-materialize robustness evaluation sets (an alternative to applying attacks
on-the-fly in robustness_eval.py).

Examples
--------
    # paper default strengths for all attacks
    python scripts/make_augmented_data.py data/rar_generated

    # specific attacks/strengths
    python scripts/make_augmented_data.py data/imagenet --attacks jpeg resize \
        --strengths 60 0.5
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataprov import augmentations as A  # noqa: E402
from dataprov.data import list_images  # noqa: E402


def _suffix(attack: str, strength) -> str:
    s = str(strength).replace(".", "")
    return f"{attack}{s}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="folder of source images")
    p.add_argument("--attacks", nargs="+", default=list(A.ATTACK_DEFAULTS))
    p.add_argument("--strengths", nargs="+", default=None,
                   help="one strength per attack (default: paper values)")
    p.add_argument("--out-root", default=None, help="where to write (default: alongside input)")
    args = p.parse_args()

    paths = list_images(args.input)
    if not paths:
        raise SystemExit(f"No images in {args.input}")

    strengths: List[Optional[float]] = (
        [float(s) for s in args.strengths] if args.strengths else [None] * len(args.attacks)
    )
    assert len(strengths) == len(args.attacks), "give one --strengths value per attack"

    base = args.out_root or os.path.dirname(os.path.abspath(args.input.rstrip("/")))
    name = os.path.basename(args.input.rstrip("/"))

    for attack, strength in zip(args.attacks, strengths):
        strength = A.ATTACK_DEFAULTS[attack] if strength is None else strength
        out_dir = os.path.join(base, f"{name}_{_suffix(attack, strength)}")
        os.makedirs(out_dir, exist_ok=True)
        for path in tqdm(paths, desc=f"{attack}={strength}", leave=False):
            img = TF.to_tensor(Image.open(path).convert("RGB"))
            att = A.apply_attack(img, attack, strength)
            TF.to_pil_image(att.clamp(0, 1)).save(os.path.join(out_dir, os.path.basename(path)))
        print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
