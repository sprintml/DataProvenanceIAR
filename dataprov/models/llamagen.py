"""LlamaGen wrapper -- next-token prediction IAR (single-scale).

LlamaGen's VQ tokenizer uses an L2-normalized codebook, so the continuous
feature is normalized (over the channel dim) before being compared to / decoded
from codebook entries -- this normalization is the only model-specific quirk.

Vendored upstream code: ``third_party/llamagen/LlamaGen`` (adapted from the
official LlamaGen repo, https://github.com/FoundationVision/LlamaGen).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .base import BaseIAR
from ._utils import add_third_party, resolve_weight

_DTYPE = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class LlamaGenIAR(BaseIAR):
    image_size = 384
    value_range = "pm1"
    name = "llamagen"

    def load_models(self):
        add_third_party("llamagen")
        from LlamaGen.tokenizer.tokenizer_image.vq_model import VQ_models  # vendored

        c = self.cfg
        self.latent_size = int(c.image_size) // int(c.downsample_size)
        vq = VQ_models[c.vq_model](
            codebook_size=int(c.codebook_size), codebook_embed_dim=int(c.codebook_embed_dim)
        )
        vq_path = resolve_weight(c, "llamagen", "vq_path", c.vq_ckpt, c.get("hf_repo"))
        vq.load_state_dict(torch.load(vq_path, map_location="cpu")["model"])
        vq.eval().to(self.device)

        gpt = None
        if bool(c.get("load_generator", False)):
            from LlamaGen.autoregressive.models.gpt import GPT_models  # vendored

            dtype = _DTYPE[str(c.get("precision", "bf16"))]
            gpt = GPT_models[c.gpt_model](
                vocab_size=int(c.codebook_size), block_size=self.latent_size ** 2,
                num_classes=int(c.num_classes), cls_token_num=int(c.cls_token_num),
                model_type=c.gpt_type,
            ).to(device=self.device, dtype=dtype)
            ckpt = torch.load(resolve_weight(c, "llamagen", "gpt_path", c.gpt_ckpt, c.get("hf_repo")),
                              map_location="cpu")
            weight = ckpt.get("model", ckpt.get("module", ckpt.get("state_dict", ckpt)))
            gpt.load_state_dict(weight, strict=False)
            gpt.eval()
        return vq, gpt

    # --- finetuning hooks ------------------------------------------------- #
    def encode_feature(self, images: Tensor) -> Tensor:
        vq = self.tokenizer
        return F.normalize(vq.quant_conv(vq.encoder(images)), p=2, dim=1)

    def decode_feature(self, f: Tensor) -> Tensor:
        return self.tokenizer.decode(f).clamp(-1, 1)

    def trainable_encoder_modules(self):
        return [self.tokenizer.encoder]

    # --- core primitive --------------------------------------------------- #
    @torch.no_grad()
    def img_to_reconstructed_img(self, images, use_quant: bool = True):
        vq = self.tokenizer
        tokens, h, quant, _ = vq.encode_with_internals(images)
        h = F.normalize(h, p=2, dim=1)
        if use_quant:
            rec = vq.decode(quant)
        else:
            tokens = None
            rec = vq.decode(h)
        return rec.clamp(-1, 1), tokens, h, quant

    # --- generation ------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, n, batch_size, seed, cfg_scale=4.0, top_k=0, top_p=1.0,
                 temperature=1.0, **kw):
        from LlamaGen.autoregressive.models.generate import generate as gpt_generate  # vendored

        vq, gpt = self.tokenizer, self.generator
        if gpt is None:
            raise RuntimeError("AR generator not loaded; build with load_generator=true.")
        ls, edim = self.latent_size, int(self.cfg.codebook_embed_dim)
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)
            g = torch.Generator(device=self.device).manual_seed(seed + start)
            labels = torch.randint(0, int(self.cfg.num_classes), (b,), generator=g, device=self.device)
            idx = gpt_generate(
                gpt, labels, ls * ls, cfg_scale=float(cfg_scale),
                temperature=float(temperature), top_k=int(top_k), top_p=float(top_p),
            )
            qzshape = [b, edim, ls, ls]
            fhat = vq.quantize.get_codebook_entry(idx, qzshape)
            img = vq.decode_code(idx, qzshape).clamp(-1, 1)
            yield ((img + 1) / 2).cpu(), fhat.cpu()
