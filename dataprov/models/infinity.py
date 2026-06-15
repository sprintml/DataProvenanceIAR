"""Infinity wrapper -- bit-wise, next-scale prediction IAR (high resolution).

Infinity quantizes with a multi-scale lookup-free quantizer (LFQ). For provenance
we use QuantLoss directly (= ||h - z||, the mean codebook error); the paper shows
this is Infinity's strongest signal once the encoder is finetuned. Eval and
finetuning need only the bit-wise VAE (bsq_vae).

Generation requires the full Infinity transformer + FLAN-T5 text encoder; it is
not wired here (build belonging data with the official Infinity repo, or evaluate
against pre-generated images).

Vendored upstream code: ``third_party/infinity/infinity`` (adapted from the
official Infinity repo, https://github.com/FoundationVision/Infinity).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .base import BaseIAR
from ._utils import add_third_party, resolve_weight


class InfinityIAR(BaseIAR):
    image_size = 1024
    value_range = "pm1"
    name = "infinity"

    def load_models(self):
        add_third_party("infinity")
        from infinity.models.bsq_vae.vae import vae_model  # vendored
        from infinity.utils.dynamic_resolution import (  # vendored
            dynamic_resolution_h_w, h_div_w_templates,
        )

        c = self.cfg
        vae_type = int(c.get("vae_type", 32))
        patchify = bool(int(c.get("apply_spatial_patchify", 0)))
        patch_size = 8 if patchify else 16
        ch_mult = [1, 2, 4, 4] if patchify else [1, 2, 4, 4, 4]

        vae_path = resolve_weight(c, "infinity", "vae_path", "infinity_vae_d32reg.pth")
        vae = vae_model(
            vae_path, schedule_mode="dynamic", codebook_dim=vae_type, codebook_size=2 ** vae_type,
            patch_size=patch_size, encoder_ch_mult=ch_mult, decoder_ch_mult=ch_mult, test_mode=True,
        )
        vae.eval().to(self.device)

        # Fixed scale schedule for square images at the configured resolution preset.
        h_div_w = 1.0
        tmpl = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        sched = dynamic_resolution_h_w[tmpl][str(c.get("pn", "1M"))]["scales"]
        self.scale_schedule = [(1, h, w) for (_, h, w) in sched]
        return vae, None

    # --- finetuning hooks ------------------------------------------------- #
    def encode_feature(self, images: Tensor) -> Tensor:
        return self.tokenizer.encode_for_raw_features(images, self.scale_schedule)[0]

    def decode_feature(self, z: Tensor) -> Tensor:
        return self.tokenizer.decode(z).clamp(-1, 1)

    def trainable_encoder_modules(self):
        return [self.tokenizer.encoder]

    # --- core primitive --------------------------------------------------- #
    @torch.no_grad()
    def img_to_reconstructed_img(self, images, use_quant: bool = True):
        vae = self.tokenizer
        indices, h, z, _ = vae.encode_with_internals(images, self.scale_schedule)
        if use_quant:
            rec, fhat = vae.decode(z), z
        else:
            indices, rec, fhat = None, vae.decode(h), h
        return rec.clamp(-1, 1), indices, h, fhat

    @torch.no_grad()
    def generate(self, n, batch_size, seed, **kw):
        raise NotImplementedError(
            "Infinity generation (transformer + FLAN-T5) is not wired. Evaluate "
            "against pre-generated Infinity images, or generate with the official "
            "Infinity repo and place the images + targets in your data folder."
        )
