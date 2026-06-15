"""Taming Transformers wrapper -- next-token prediction IAR (single-scale VQGAN).

Provenance only needs the VQGAN tokenizer (encoder + quantizer + decoder), which
we load directly from the VQGAN config/checkpoint. Like RAR, Taming benefits from
inverse-decoder finetuning and supports augmentation finetuning + robustness.

Generation uses the Taming (net2net) class-conditional transformer; set
``load_generator=true`` to load it (heavier). Eval/finetuning need only the VQGAN.

Vendored upstream code: ``third_party/taming/deps/taming`` (adapted from
CompVis/taming-transformers).
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from .base import BaseIAR
from ._utils import add_third_party, resolve_weight


class TamingIAR(BaseIAR):
    image_size = 256
    value_range = "pm1"
    name = "taming"

    def _path(self, key, default_rel):
        explicit = self.cfg.get(key, "")
        if explicit:
            return explicit
        model_dir = self.cfg.get("model_dir", "")
        return os.path.join(model_dir, default_rel) if model_dir else ""

    def load_models(self):
        add_third_party("taming")
        from omegaconf import OmegaConf
        from deps.taming.util import instantiate_from_config  # vendored

        cfg_path = self._path("config_path", "configs/vqgan.yaml")
        ckpt_path = self._path("ckpt_path", "checkpoints/vqgan.ckpt")
        if not (cfg_path and ckpt_path):
            raise FileNotFoundError(
                "Set model_dir (or config_path/ckpt_path) to the Taming VQGAN config + checkpoint."
            )
        vqcfg = OmegaConf.load(cfg_path)
        # The training loss (VQLPIPSWithDiscriminator) is irrelevant for provenance
        # and pulls heavy deps; replace it with a no-op so construction stays light.
        if "params" in vqcfg.model and "lossconfig" in vqcfg.model.params:
            vqcfg.model.params.lossconfig = OmegaConf.create({"target": "torch.nn.Identity"})
        vqgan = instantiate_from_config(vqcfg.model)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        vqgan.load_state_dict(sd, strict=False)
        vqgan.eval().to(self.device)

        generator = None
        if bool(self.cfg.get("load_generator", False)):
            generator = self._load_transformer(instantiate_from_config)
        return vqgan, generator

    def _load_transformer(self, instantiate_from_config):
        from omegaconf import OmegaConf

        cfg_path = self._path("net2net_config", "configs/net2net.yaml")
        ckpt_path = self._path("net2net_ckpt", "checkpoints/net2net.ckpt")
        net = instantiate_from_config(OmegaConf.load(cfg_path).model)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(sd.get("state_dict", sd))
        net.eval().to(self.device)
        return net

    # --- finetuning hooks ------------------------------------------------- #
    def encode_feature(self, images: Tensor) -> Tensor:
        vq = self.tokenizer
        return vq.quant_conv(vq.encoder(images))

    def decode_feature(self, z: Tensor) -> Tensor:
        vq = self.tokenizer
        return vq.decoder(vq.post_quant_conv(z)).clamp(-1, 1)

    def trainable_encoder_modules(self):
        return [self.tokenizer.encoder]

    # --- core primitive --------------------------------------------------- #
    @torch.no_grad()
    def img_to_reconstructed_img(self, images, use_quant: bool = True):
        vq = self.tokenizer
        z = vq.quant_conv(vq.encoder(images))
        if use_quant:
            z_q, _, _ = vq.quantize(z)
        else:
            z_q = z
        rec = vq.decoder(vq.post_quant_conv(z_q)).clamp(-1, 1)
        return rec, None, z, z_q

    # --- generation ------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, n, batch_size, seed, temperature=1.0, top_k=250, top_p=0.92, **kw):
        net = self.generator
        if net is None:
            raise RuntimeError("AR generator not loaded; build with load_generator=true.")
        from deps.taming.modules.transformer.mingpt import sample_with_past  # vendored

        vq = self.tokenizer
        steps = int(self.cfg.get("codes_size", 16)) ** 2
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)
            g = torch.Generator(device=self.device).manual_seed(seed + start)
            labels = torch.randint(0, 1000, (b, 1), generator=g, device=self.device)
            c_indices = net.cond_stage_model.encode(labels) if hasattr(net, "cond_stage_model") else labels
            codes = sample_with_past(c_indices, net.transformer, steps=steps,
                                     sample_logits=True, temperature=float(temperature),
                                     top_k=int(top_k), top_p=float(top_p))
            qzshape = [b, int(self.cfg.get("dim_z", 256)), 16, 16]
            z_q = vq.quantize.get_codebook_entry(codes.reshape(-1), shape=qzshape)
            img = vq.decoder(vq.post_quant_conv(z_q)).clamp(-1, 1)
            yield ((img + 1) / 2).cpu(), z_q.cpu()
