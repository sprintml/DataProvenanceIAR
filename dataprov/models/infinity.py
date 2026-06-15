"""Infinity wrapper -- bit-wise, next-scale prediction IAR (high resolution).

Infinity quantizes with a multi-scale lookup-free quantizer (LFQ). For provenance
we use QuantLoss directly (= ||h - z||, the mean codebook error); the paper shows
this is Infinity's strongest signal once the encoder is finetuned. Eval and
finetuning need only the bit-wise VAE (bsq_vae).

Generation is text-to-image: it loads the Infinity transformer + a FLAN-T5 text
encoder (set ``load_generator=true``) and samples one image per prompt. The
quantized feature ``summed_codes`` the image was generated from is returned as the
finetuning target.

Vendored upstream code: ``third_party/infinity/infinity`` (adapted from the
official Infinity repo, https://github.com/FoundationVision/Infinity).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .base import BaseIAR
from ._utils import add_third_party, resolve_weight

# A small default prompt set for generating belonging data when no prompt file is
# given. For faithful reproduction, supply your own prompts via `prompt_file`.
_DEFAULT_PROMPTS = [
    "a photo of a golden retriever", "a photo of a tabby cat", "a photo of a red sports car",
    "a photo of a hot air balloon", "a photo of a wooden sailboat", "a photo of a steam locomotive",
    "a photo of a snowy mountain", "a photo of a tropical beach", "a photo of a bowl of fruit",
    "a photo of a city street at night", "a photo of a sunflower field", "a photo of a teddy bear",
]


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
        self.vae_type = int(c.get("vae_type", 32))
        patchify = bool(int(c.get("apply_spatial_patchify", 0)))
        patch_size = 8 if patchify else 16
        ch_mult = [1, 2, 4, 4] if patchify else [1, 2, 4, 4, 4]

        vae_path = resolve_weight(c, "infinity", "vae_path", "infinity_vae_d32reg.pth")
        vae = vae_model(
            vae_path, schedule_mode="dynamic", codebook_dim=self.vae_type, codebook_size=2 ** self.vae_type,
            patch_size=patch_size, encoder_ch_mult=ch_mult, decoder_ch_mult=ch_mult, test_mode=True,
        )
        vae.eval().to(self.device)

        tmpl = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 1.0))]
        sched = dynamic_resolution_h_w[tmpl][str(c.get("pn", "1M"))]["scales"]
        self.scale_schedule = [(1, h, w) for (_, h, w) in sched]

        generator = None
        if bool(c.get("load_generator", False)):
            generator = self._load_transformer(vae)
            self._load_text_encoder()
        return vae, generator

    # --- generation model loading ---------------------------------------- #
    def _load_transformer(self, vae):
        from infinity.models.infinity import Infinity  # vendored

        c = self.cfg
        kwargs = dict(depth=32, embed_dim=2048, num_heads=2048 // 128,
                      drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)  # infinity_2b
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
            inf = Infinity(
                vae_local=vae, text_channels=int(c.get("text_channels", 2048)), text_maxlen=512,
                shared_aln=True, raw_scale_schedule=None, checkpointing="full-block",
                customized_flash_attn=False, fused_norm=True, pad_to_multiplier=128, use_flex_attn=False,
                add_lvl_embeding_only_first_block=int(c.get("add_lvl_embeding_only_first_block", 1)),
                use_bit_label=int(c.get("use_bit_label", 1)),
                rope2d_each_sa_layer=int(c.get("rope2d_each_sa_layer", 1)),
                rope2d_normalized_by_hw=int(c.get("rope2d_normalized_by_hw", 2)),
                pn=str(c.get("pn", "1M")), apply_spatial_patchify=int(c.get("apply_spatial_patchify", 0)),
                inference_mode=True, train_h_div_w_list=[1.0], **kwargs,
            ).to(self.device)
            if bool(int(c.get("bf16", 1))):
                for block in inf.unregistered_blocks:
                    block.bfloat16()
            inf.eval().requires_grad_(False)
            sd = torch.load(resolve_weight(c, "infinity", "model_path", "infinity_2b_reg.pth"),
                            map_location=self.device)
            inf.load_state_dict(sd)
            inf.rng = torch.Generator(device=self.device)
        return inf

    def _load_text_encoder(self):
        from transformers import AutoTokenizer, T5EncoderModel

        t5 = self.cfg.get("text_encoder", "")
        if not t5:
            raise FileNotFoundError("Set 'text_encoder' to a FLAN-T5-XL path for generation.")
        self.text_tokenizer = AutoTokenizer.from_pretrained(t5, revision=None, legacy=True)
        self.text_tokenizer.model_max_length = 512
        self.text_encoder = T5EncoderModel.from_pretrained(t5, torch_dtype=torch.float16).to(self.device).eval()
        self.text_encoder.requires_grad_(False)

    def _encode_prompt(self, prompt: str):
        tok = self.text_tokenizer([prompt], max_length=512, padding="max_length",
                                  truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(self.device)
        mask = tok.attention_mask.to(self.device)
        feats = self.text_encoder(input_ids=ids, attention_mask=mask)["last_hidden_state"].float()
        lens = mask.sum(dim=-1).tolist()
        cu = F.pad(mask.sum(dim=-1).to(torch.int32).cumsum_(0), (1, 0))
        kv = torch.cat([f[:l] for l, f in zip(lens, feats.unbind(0))], dim=0)
        return (kv, lens, cu, max(lens))

    def _prompts(self, n: int):
        pf = self.cfg.get("prompt_file", "")
        if pf:
            with open(pf) as f:
                prompts = [ln.strip() for ln in f if ln.strip()]
        else:
            prompts = _DEFAULT_PROMPTS
        return [prompts[i % len(prompts)] for i in range(n)]

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

    # --- generation (text-to-image) -------------------------------------- #
    @torch.no_grad()
    def generate(self, n, batch_size, seed, cfg_scale=3.0, tau=1.0, top_p=0.97, **kw):
        if self.generator is None:
            raise RuntimeError("AR generator not loaded; build with load_generator=true.")
        vae = self.tokenizer
        prompts = self._prompts(n)
        cfg_list = [float(cfg_scale)] * len(self.scale_schedule)
        tau_list = [float(tau)] * len(self.scale_schedule)
        for i in range(n):
            text_cond = self._encode_prompt(prompts[i])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
                _, _, _img, summed = self.generator.autoregressive_infer_cfg(
                    vae=vae, scale_schedule=self.scale_schedule, label_B_or_BLT=text_cond,
                    g_seed=seed + i, B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
                    cfg_sc=3, cfg_list=cfg_list, tau_list=tau_list, top_k=0, top_p=float(top_p),
                    returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False, cfg_exp_k=0.0,
                    cfg_insertion_layer=[int(self.cfg.get("cfg_insertion_layer", 0))],
                    vae_type=self.vae_type, softmax_merge_topk=-1, ret_img=True, trunk_scale=1000,
                    gt_leak=-1, gt_ls_Bl=None, inference_mode=True,
                    sampling_per_bits=int(self.cfg.get("sampling_per_bits", 1)),
                )
            img01 = ((vae.decode(summed) + 1) / 2).clamp(0, 1).float()
            yield img01.cpu(), summed.float().cpu()
