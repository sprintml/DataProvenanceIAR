"""Baseline provenance methods compared against in the paper.

Two of the three baselines reuse the exact signal code, only with the model's
*original* (non-finetuned) encoder, so they live in the evaluation script rather
than here:

* **Reconstruction** -- the naive reconstruction loss
  ``||x - D(Q^{-1}(Q(E(x))))||``. This is ``provenance_signals(...)["reconstruction"]``
  computed with ``--encoder original``.
* **AEDR** -- the calibrated double-reconstruction ratio with the original
  encoder. This is ``provenance_signals(...)["enc_loss"]`` computed with
  ``--encoder original``.

The remaining baseline, **LatentTracer**, is a separate optimization and is
implemented below.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .models.base import BaseIAR
from .signals import mse_per_sample

__all__ = ["latent_tracer"]


def latent_tracer(
    model: BaseIAR,
    images: Tensor,
    iters: int = 100,
    lr: float = 1e-2,
    lr_decay_every: int = 50,
    lr_decay: float = 0.5,
) -> np.ndarray:
    """LatentTracer baseline (Wang et al., 2024).

    Initializes the latent at the quantized encoding of each image and optimizes
    it (Adam) to minimize the reconstruction error, halving the learning rate
    every ``lr_decay_every`` steps. Returns the final per-sample reconstruction
    MSE; belonging images reach a lower value.
    """
    images = images.to(model.device)

    with torch.no_grad():
        _, _, _, fhat = model.img_to_reconstructed_img(images, use_quant=True)
    latent = torch.nn.Parameter(fhat.detach().clone())
    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_per_sample = torch.zeros(images.shape[0])
    for i in range(iters):
        optimizer.zero_grad()
        rec = model.decode_feature(latent)
        per_sample = mse_per_sample(rec, images)
        per_sample.mean().backward()
        optimizer.step()
        loss_per_sample = per_sample.detach().cpu()
        if (i + 1) % lr_decay_every == 0:
            for g in optimizer.param_groups:
                g["lr"] *= lr_decay

    return loss_per_sample.numpy()
