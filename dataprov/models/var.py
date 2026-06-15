"""VAR wrapper -- next-scale prediction IAR with optimized quantization.

VAR is multi-scale: each spatial feature is the sum of upsampled tokens from all
scales, so a greedy scale-wise quantization cannot recover the tokens an image
was generated from. We therefore search for the token combination by gradient
descent over soft codebook assignments (the optimized quantization), which is
what makes QuantLoss work for VAR.

Vendored upstream code: ``third_party/var/var_arch`` (adapted from the official
VAR repo, https://github.com/FoundationVision/VAR).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import BaseIAR
from ._utils import add_third_party, resolve_weight


class VarIAR(BaseIAR):
    image_size = 256
    value_range = "pm1"
    name = "var"

    def load_models(self):
        add_third_party("var")
        c = self.cfg
        self.patch_nums = tuple(c.patch_nums)
        # The AR transformer (several GB) is only needed for generation; skip it
        # for evaluation / finetuning by leaving load_generator=False.
        if bool(c.get("load_generator", False)):
            from var_arch import build_vae_var  # vendored

            vae, var = build_vae_var(
                device=self.device, patch_nums=self.patch_nums,
                V=int(c.V), Cvae=int(c.Cvae), ch=int(c.ch),
                share_quant_resi=int(c.share_quant_resi),
                num_classes=int(c.num_classes), depth=int(c.depth), shared_aln=False,
            )
            ar_path = resolve_weight(c, "var", "ar_path", c.ar_ckpt, c.get("hf_repo"))
            var.load_state_dict(torch.load(ar_path, map_location="cpu"), strict=True)
            var.eval().to(self.device)
        else:
            from var_arch import VQVAE  # vendored

            vae = VQVAE(
                vocab_size=int(c.V), z_channels=int(c.Cvae), ch=int(c.ch),
                test_mode=True, share_quant_resi=int(c.share_quant_resi),
                v_patch_nums=self.patch_nums,
            )
            var = None
        vae_path = resolve_weight(c, "var", "vae_path", c.vae_ckpt, c.get("hf_repo"))
        vae.load_state_dict(torch.load(vae_path, map_location="cpu"), strict=True)
        vae.eval().to(self.device)
        return vae, var

    # --- finetuning hooks ------------------------------------------------- #
    def encode_feature(self, images: Tensor) -> Tensor:
        vae = self.tokenizer
        return vae.quant_conv(vae.encoder(images))

    def decode_feature(self, f: Tensor) -> Tensor:
        vae = self.tokenizer
        return vae.decoder(vae.post_quant_conv(f)).clamp(-1, 1)

    def trainable_encoder_modules(self):
        return [self.tokenizer.encoder, self.tokenizer.quant_conv]

    # --- optimized quantization ------------------------------------------ #
    def _optimized_quant(self, f: Tensor) -> Tuple[List[Tensor], Tensor]:
        """Greedy init + layer-by-layer soft-assignment refinement.

        Total iterations = optim_iter x (1 + (9 - optim_stop_scale)); with the
        defaults (200 x 6) this is the paper's 1200 iterations.
        """
        c = self.cfg
        q = self.tokenizer.quantize
        common = dict(
            iters=int(c.optim_iter), lr=float(c.optim_lr), entropy_weight=0,
            tau_start=float(c.optim_tau_start), tau_end=float(c.optim_tau_end),
        )
        with torch.enable_grad():
            idx0 = q.f_to_idxBl_or_fhat(f.clone(), to_fhat=False, v_patch_nums=self.patch_nums)
            idxBl, fhat, _ = q.refine_soft_assign(f, init_idx_Bl=idx0, **common)
            for i in range(9, int(c.optim_stop_scale), -1):
                idxBl, fhat, _ = q.refine_soft_assign(f, init_idx_Bl=idxBl, fix_scale=i, **common)
        return idxBl, fhat.detach()

    # --- core primitive --------------------------------------------------- #
    @torch.no_grad()
    def img_to_reconstructed_img(self, images, use_quant: bool = True):
        vae = self.tokenizer
        f = vae.quant_conv(vae.encoder(images))
        if use_quant:
            if bool(self.cfg.get("token_optim", True)):
                idxBl, fhat = self._optimized_quant(f)
            else:
                idxBl = vae.quantize.f_to_idxBl_or_fhat(f.clone(), to_fhat=False, v_patch_nums=self.patch_nums)
                fhat = vae.quantize.f_to_idxBl_or_fhat(f.clone(), to_fhat=True, v_patch_nums=self.patch_nums)[-1]
        else:
            idxBl, fhat = None, f
        rec = vae.decoder(vae.post_quant_conv(fhat)).clamp(-1, 1)
        return rec, idxBl, f, fhat

    # --- generation ------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, n, batch_size, seed, cfg_scale=4.0, top_k=900, top_p=0.96, **kw):
        vae, var = self.tokenizer, self.generator
        if var is None:
            raise RuntimeError("AR generator not loaded; build with load_generator=true.")
        num_classes = int(self.cfg.num_classes)
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)
            g = torch.Generator(device=self.device).manual_seed(seed + start)
            labels = torch.randint(0, num_classes, (b,), generator=g, device=self.device)
            img01, all_idxBl, _ = var.autoregressive_infer_cfg_with_token_map(
                B=b, label_B=labels, g_seed=seed + start,
                cfg=float(cfg_scale), top_k=int(top_k), top_p=float(top_p), more_smooth=False,
            )
            ms_h = vae.idxBl_to_embedhat(all_idxBl)
            fhat = vae.quantize.embedhat_to_fhat(ms_h, all_to_max_scale=True, last_one=True)
            yield img01.clamp(0, 1).cpu(), fhat.cpu()

    # NOTE on preprocessing: we deliberately use the BaseIAR default transform
    # (Resize(image_size) on the shorter side + CenterCrop), which is a no-op for
    # images already at the model resolution. VAR's training-time mid-reso LANCZOS
    # resize must NOT be applied here: upsampling a native-256 generated image and
    # cropping resamples it (like a resize attack) and destroys the codebook-
    # alignment signal that QuantLoss relies on.
