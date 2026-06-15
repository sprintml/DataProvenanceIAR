"""Data provenance for image auto-regressive generation.

A post-hoc, model-agnostic framework that traces an image back to the IAR that
generated it, using three signals derived from the model's vector-quantized
autoencoder: QuantLoss, EncLoss and their Combined product. See the project
README for usage.
"""

from __future__ import annotations

from . import augmentations, baselines, metrics, signals
from .config import load_config
from .models import MODEL_NAMES, build_model
from .signals import SIGNAL_NAMES, provenance_signals

__all__ = [
    "augmentations",
    "baselines",
    "metrics",
    "signals",
    "load_config",
    "build_model",
    "MODEL_NAMES",
    "provenance_signals",
    "SIGNAL_NAMES",
]

__version__ = "0.1.0"
