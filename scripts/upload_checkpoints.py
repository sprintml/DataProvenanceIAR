"""Upload finetuned inverse-decoder encoders to the HuggingFace Hub.

Run this once per model after finetuning, under your own HF account. Authenticate
first with ``huggingface-cli login`` (or set ``HF_TOKEN``).

All encoders go into a single model repo with layout ``<model>/encoder_final.pth``,
so users can fetch any of them with scripts/download_checkpoints.py.

Examples
--------
    huggingface-cli login
    python scripts/upload_checkpoints.py var \
        --encoder-path checkpoints/var/encoder_final.pth \
        --hf-repo <user>/dataprovenance-iar-encoders --create
"""

from __future__ import annotations

import argparse
import os

_CARD = """---
license: mit
tags:
  - image-provenance
  - autoregressive-image-generation
  - data-provenance
---

# Inverse-Decoder Encoders for "Data Provenance for Image Auto-Regressive Generation"

Finetuned encoders (inverse decoders) for the provenance framework. Each file
``<model>/encoder_final.pth`` is a full state dict of the finetuned encoder for
one IAR model (LlamaGen, RAR, VAR, Taming, Infinity).

Use with the code at <PROJECT_REPO_URL>:

```bash
python scripts/download_checkpoints.py <model> --what encoder --hf-repo <THIS_REPO>
python scripts/evaluate.py <model> --belonging ... --nonbelonging ...
```
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="model name, used as the folder inside the repo")
    p.add_argument("--encoder-path", required=True, help="local encoder_final.pth to upload")
    p.add_argument("--hf-repo", required=True, help="target HF repo id, e.g. user/repo")
    p.add_argument("--create", action="store_true", help="create the repo if missing")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.encoder_path):
        raise SystemExit(f"No file at {args.encoder_path}")

    from huggingface_hub import HfApi

    api = HfApi()
    if args.create:
        api.create_repo(args.hf_repo, repo_type="model", private=args.private, exist_ok=True)
        # Seed/refresh the model card.
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(_CARD)
            card_path = f.name
        api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md", repo_id=args.hf_repo)

    api.upload_file(
        path_or_fileobj=args.encoder_path,
        path_in_repo=f"{args.model}/encoder_final.pth",
        repo_id=args.hf_repo,
    )
    print(f"uploaded {args.encoder_path} -> {args.hf_repo}:{args.model}/encoder_final.pth")


if __name__ == "__main__":
    main()
