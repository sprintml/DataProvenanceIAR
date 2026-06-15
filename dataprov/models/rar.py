"""RAR wrapper -- randomized-order next-token IAR (single-scale).

RAR tokenizes with a MaskGIT-VQGAN: each spatial feature maps to a single
codebook entry, so quantization (and its inverse) is the plain nearest-neighbour
lookup -- no optimized search needed. The original encoder is a poor inverse of
the decoder, so inverse-decoder finetuning is essential here, and RAR also
supports augmentation finetuning + robustness evaluation.

Vendored upstream code: ``third_party/rar/rar`` (adapted from the RAR /
1d-tokenizer repos, Bytedance/Apache-2.0).
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
from torch import Tensor

from .base import BaseIAR
from ._utils import add_third_party, resolve_weight


class RarIAR(BaseIAR):
    image_size = 256
    value_range = "01"
    name = "rar"

    def load_models(self):
        tp = add_third_party("rar")
        from omegaconf import OmegaConf
        from rar.modeling.titok import PretrainedTokenizer  # vendored
        from rar.modeling.rar import RAR  # vendored

        c = self.cfg
        tok_path = resolve_weight(c, "rar", "tokenizer_path", c.tokenizer_ckpt, c.get("tokenizer_repo"))
        tokenizer = PretrainedTokenizer(tok_path)
        tokenizer.eval().to(self.device)

        generator = None
        if bool(c.get("load_generator", False)):
            rar_cfg = OmegaConf.load(os.path.join(tp, "rar/configs/training/generator/rar.yaml"))
            rar_cfg.model.generator.hidden_size = int(c.hidden_size)
            rar_cfg.model.generator.num_hidden_layers = int(c.num_hidden_layers)
            rar_cfg.model.generator.num_attention_heads = int(c.num_attention_heads)
            rar_cfg.model.generator.intermediate_size = int(c.intermediate_size)
            ar_path = resolve_weight(c, "rar", "ar_path", c.ar_ckpt, c.get("ar_repo"))
            generator = RAR(rar_cfg)
            generator.load_state_dict(torch.load(ar_path, map_location="cpu"))
            generator.eval().requires_grad_(False).to(self.device)
            generator.set_random_ratio(0)
        return tokenizer, generator

    # --- finetuning hooks ------------------------------------------------- #
    def encode_feature(self, images: Tensor) -> Tensor:
        return self.tokenizer.encoder(images)

    def decode_feature(self, f: Tensor) -> Tensor:
        return self.tokenizer.decoder(f).clamp(0, 1)

    def trainable_encoder_modules(self):
        return [self.tokenizer.encoder]

    # --- core primitive --------------------------------------------------- #
    @torch.no_grad()
    def img_to_reconstructed_img(self, images, use_quant: bool = True):
        tok = self.tokenizer
        if use_quant:
            tokens, f, fhat, _ = tok.encode_with_internals(images.clone())
            rec = tok.decode_tokens(tokens.clone())
        else:
            _, f, fhat, _ = tok.encode_without_quant(images.clone())
            tokens = None
            rec = tok.decode_states(f.clone())
        return rec.clamp(0, 1), tokens, f, fhat

    # --- generation ------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, n, batch_size, seed, guidance_scale=4.0, temperature=1.0,
                 num_sample_steps=8, guidance_scale_pow=3.0, guidance_decay="constant", **kw):
        tok, gen = self.tokenizer, self.generator
        if gen is None:
            raise RuntimeError("AR generator not loaded; build with load_generator=true.")
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)
            g = torch.Generator(device=self.device).manual_seed(seed + start)
            labels = torch.randint(0, 1000, (b,), generator=g, device=self.device)
            tokens = gen.generate(
                condition=labels, guidance_scale=float(guidance_scale),
                guidance_decay=str(guidance_decay), guidance_scale_pow=float(guidance_scale_pow),
                randomize_temperature=float(temperature),
                softmax_temperature_annealing=False, num_sample_steps=int(num_sample_steps),
            )
            tokens = tokens.view(b, -1)
            fhat = tok.quantize.get_codebook_entry(tokens)
            img = tok.decode_tokens(tokens).clamp(0, 1)
            yield img.cpu(), fhat.cpu()
