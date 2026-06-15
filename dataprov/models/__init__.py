"""Model registry.

Each model wrapper is imported lazily so that pulling in one model's (vendored)
upstream code never forces the dependencies of the others.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseIAR

__all__ = ["BaseIAR", "build_model", "MODEL_NAMES"]

#: name -> "module:ClassName" (under dataprov.models)
_REGISTRY: Dict[str, str] = {
    "llamagen": "llamagen:LlamaGenIAR",
    "rar": "rar:RarIAR",
    "var": "var:VarIAR",
    "taming": "taming:TamingIAR",
    "infinity": "infinity:InfinityIAR",
}

MODEL_NAMES = tuple(_REGISTRY)


def _load_class(spec: str) -> Type[BaseIAR]:
    import importlib

    module_name, class_name = spec.split(":")
    module = importlib.import_module(f"dataprov.models.{module_name}")
    return getattr(module, class_name)


def build_model(name: str, cfg, device: str = "cuda") -> BaseIAR:
    """Instantiate the wrapper registered under ``name``."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {MODEL_NAMES}")
    cls = _load_class(_REGISTRY[name])
    return cls(cfg, device=device)
