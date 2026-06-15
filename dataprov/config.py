"""Lightweight configuration (no heavy dependencies).

Model configs live in ``configs/<name>.yaml`` and are parsed with PyYAML into a
small attribute-accessible ``DotDict``. We deliberately avoid OmegaConf/Hydra so
that ``dataprov`` slots into each model's own conda environment with only the
dependencies those environments already have.

Paths are resolved from environment variables so one config works everywhere:

* ``DATAPROV_CHECKPOINTS`` -- base + finetuned weight cache (default ``<repo>/checkpoints``)
* ``DATAPROV_DATA``        -- generated / natural datasets   (default ``<repo>/data``)
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import yaml

__all__ = ["DotDict", "repo_root", "checkpoints_dir", "data_dir", "load_config"]


class DotDict(dict):
    """A dict whose keys are also accessible as attributes (recursively)."""

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return cls(v)
        if isinstance(v, list):
            return [cls._wrap(x) for x in v]
        return v

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = self._wrap(v)

    def setdefault(self, k: str, default: Any) -> Any:  # type: ignore[override]
        if k not in self:
            self[k] = self._wrap(default)
        return self[k]


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def checkpoints_dir() -> str:
    return os.environ.get("DATAPROV_CHECKPOINTS", os.path.join(repo_root(), "checkpoints"))


def data_dir() -> str:
    return os.environ.get("DATAPROV_DATA", os.path.join(repo_root(), "data"))


def _apply_override(cfg: DotDict, dotted: str) -> None:
    """Apply a single ``a.b.c=value`` override (value parsed as YAML scalar)."""
    key, _, raw = dotted.partition("=")
    value = yaml.safe_load(raw)
    node: dict = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, DotDict())
    node[parts[-1]] = DotDict._wrap(value)


def load_config(model_name: str, overrides: Optional[List[str]] = None) -> DotDict:
    """Load ``configs/<model_name>.yaml`` and apply ``a.b=value`` overrides."""
    path = os.path.join(repo_root(), "configs", f"{model_name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No config for '{model_name}' at {path}")
    with open(path, "r") as f:
        cfg = DotDict(yaml.safe_load(f) or {})
    cfg.setdefault("name", model_name)
    cfg.setdefault("checkpoints_dir", checkpoints_dir())
    cfg.setdefault("data_dir", data_dir())
    for o in overrides or []:
        _apply_override(cfg, o)
    return cfg
