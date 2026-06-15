"""Helpers shared by the model wrappers: vendored-code path + weight resolution."""

from __future__ import annotations

import os
import sys
from typing import Optional

from ..config import checkpoints_dir, repo_root


def add_third_party(name: str) -> str:
    """Put ``third_party/<name>`` on ``sys.path`` and return the path."""
    p = os.path.join(repo_root(), "third_party", name)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def resolve_weight(
    cfg, model_name: str, explicit_key: str, filename: str, hf_repo: Optional[str] = None
) -> str:
    """Locate a weight file, preferring (1) an explicit config path, then
    (2) ``<checkpoints>/<model>/<filename>``, then (3) a HuggingFace download.
    """
    explicit = cfg.get(explicit_key, None)
    if explicit and os.path.exists(explicit):
        return explicit

    local = os.path.join(checkpoints_dir(), model_name, filename)
    if os.path.exists(local):
        return local

    if hf_repo:
        from huggingface_hub import hf_hub_download

        os.makedirs(os.path.dirname(local), exist_ok=True)
        return hf_hub_download(
            repo_id=hf_repo, filename=filename,
            local_dir=os.path.join(checkpoints_dir(), model_name),
        )

    raise FileNotFoundError(
        f"Could not find {filename} for '{model_name}'. Set '{explicit_key}' in the "
        f"config to an absolute path, or place it at {local}."
    )
