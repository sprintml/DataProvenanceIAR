from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

from . import dist


# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
        
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False, ret_embeddings=False, token_forcing=None) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        if ret_embeddings:
            f_no_grad = f_BChw.clone().detach()
            f_rest = f_BChw
        else:
            f_no_grad = f_BChw.detach()
            f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        if token_forcing is not None:
            idx_Bl_target_all = token_forcing

        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            embed_scales: List[torch.Tensor] = []
            idx_Bl_all = []
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    # rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area') if (si != SN-1) else f_rest
                    embed_scales.append(rest_NC)
                    rest_NC = rest_NC.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                idx_Bl_all.append(idx_N.clone().detach().reshape(B, pn * pn))
                if token_forcing is not None:
                    idx_N = idx_Bl_target_all[si].detach().clone().reshape(B*pn*pn)
                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    if dist.initialized(): handler = tdist.all_reduce(hit_V, async_op=True)
                
                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat = f_hat + h_BChw
                # f_rest -= h_BChw
                f_rest = f_rest - h_BChw
                
                if self.training and dist.initialized():
                    handler.wait()
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        
        if tdist.is_initialized():
            margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        else:
            margin = (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        else: usages = None
        if ret_embeddings:
            return f_hat, usages, mean_vq_loss, embed_scales, idx_Bl_all
        else:
            return f_hat, usages, mean_vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw

    def embedhat_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            # downscaling
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, ph, pw)
            # upscaling
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw))
        
        return f_hat_or_idx_Bl
    
    def refine_soft_assign(
        self,
        f_BChw: torch.Tensor,
        init_idx_Bl=None,
        iters: int = 150,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
        entropy_weight: float = 1e-3,
        v_patch_nums=None,
        lr: float = 0.1,
        fix_scale=10,
    ):
        """
        Annealed soft-assign refinement of code indices:
        - optimize full MSE with soft assignments p over the codebook,
        - entropy penalty encourages one-hot,
        - temperature anneals from tau_start to tau_end,
        - finally snap to argmax and return indices, f_hat, mse.

        Returns: idx_Bl_all (list of (B, ph*pw)), f_hat (B,C,H,W), mse (scalar tensor)
        """
        device = f_BChw.device
        B, C, H, W = f_BChw.shape
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]
        assert patch_hws[-1] == (H, W)
        SN = len(patch_hws)

        V = self.vocab_size
        E = self.embedding.weight  # (V, C)

        # logits per scale: (B, ph*pw, V)
        logits = []
        for si, (ph, pw) in enumerate(patch_hws):
            n = ph * pw
            L = torch.zeros(B, n, V, device=device, dtype=torch.float32, requires_grad=True)
            if init_idx_Bl is not None:
                idx0 = init_idx_Bl[si]  # (B, n)
                with torch.no_grad():
                    L.scatter_(dim=2, index=idx0.unsqueeze(-1), value=0.5)  # bias towards init
            logits.append(nn.Parameter(L))

        opt = torch.optim.Adam(logits[:fix_scale], lr=lr)

        def build_f_hat_from_probs(prob_list):
            f_hat = torch.zeros(B, C, H, W, device=device, dtype=torch.float32)
            for si, (ph, pw) in enumerate(patch_hws):
                # p: (B, ph*pw, V) ; z: (B, ph*pw, C)
                p = prob_list[si]

                # idx = torch.argmax(p, dim=-1)  # (B, ph*pw)
                # z = E[idx]
                if si >= fix_scale:
                    # just take the embedding with the highest prob
                    idx = torch.argmax(p, dim=-1)  # (B, ph*pw)
                    z = E[idx]  # (B, ph*pw, C)
                else:
                    # z = p @ E  # expected embedding
                    # z = torch.matmul(p, E)
                    z = torch.einsum('bnv,vc->bnc', p, E)
    
                z = z.view(B, ph, pw, C).permute(0, 3, 1, 2).contiguous()  # B,C,ph,pw
                if si != SN - 1:
                    z = F.interpolate(z, size=(H, W), mode='bicubic')

                # z = self.quant_resi[si/(SN-1)](z)
                normalized_scale = si / (SN - 1)
                idx = np.argmin(np.abs(self.quant_resi.ticks - normalized_scale)).item()
                z = self.quant_resi.qresi_ls[idx](z)

                f_hat.add_(z)
            return f_hat

        # anneal schedule
        def temp(t):
            if iters <= 1: return tau_end
            a = t / (iters - 1)
            return tau_start * (tau_end / tau_start) ** a

        # Optional: compile for PyTorch 2.0+ (comment out if causing issues)
        try:
            if hasattr(torch, 'compile'):
                build_f_hat_from_probs = torch.compile(build_f_hat_from_probs, mode='default')
        except Exception:
            pass  # torch.compile unavailable (e.g. Python 3.12+) -> run uncompiled

        # training loop
        with torch.cuda.amp.autocast(enabled=False):
            for t in range(iters):
                tau = temp(t)
                opt.zero_grad(set_to_none=True)

                probs = [F.softmax(L / tau, dim=-1) for L in logits]  # list of (B, n, V)
                f_hat = build_f_hat_from_probs(probs)
                mse = 0.5 * F.mse_loss(f_hat, f_BChw, reduction='mean')

                # low-entropy encouragement -> one-hots
                # ent = 0.0
                # for p in probs:
                #     ent = ent - (p * (p.clamp_min(1e-12).log())).sum(dim=-1).mean()
                # loss = mse + entropy_weight * ent  # ent <= 0; adding pushes it toward 0 (peaky)
                loss = mse

                loss.backward()
                opt.step()

                # if t%50==0:
                #     for g in opt.param_groups:
                #         g['lr'] = g['lr']*0.5

        # snap to discrete codes
        with torch.no_grad():
            idx_Bl_all = []
            f_hat = torch.zeros(B, C, H, W, device=device, dtype=torch.float32)
            for si, (ph, pw) in enumerate(patch_hws):
                p = F.softmax(logits[si] / tau_end, dim=-1)  # final temperature
                idx = torch.argmax(p, dim=-1)  # (B, ph*pw)
                idx_Bl_all.append(idx)

                z = self.embedding(idx.view(B, ph, pw)).permute(0, 3, 1, 2).contiguous()
                if si != SN - 1:
                    z = F.interpolate(z, size=(H, W), mode='bicubic')
                z = self.quant_resi[si/(SN-1)](z)
                f_hat.add_(z)

            final_mse = F.mse_loss(f_hat, f_BChw)
        return idx_Bl_all, f_hat, final_mse

    def refine_soft_assign_topk(
        self,
        f_BChw: torch.Tensor,
        init_idx_Bl=None,
        iters: int = 150,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
        entropy_weight: float = 1e-3,
        v_patch_nums=None,
        lr: float = 0.1,
        fix_scale=10,
        K: int = 10,
    ):
        """
        Top-K refinement of code indices with annealed soft-assignment.

        Instead of optimizing over the full vocabulary (V=4096), we only consider
        the K nearest codes for each token position. This gives:
        - ~400x fewer parameters (K=10 vs V=4096)
        - Faster optimization (less computation per iteration)
        - Better gradients (concentrated on relevant codes)
        - Implicit regularization (prevents drifting to distant codes)

        Args:
            f_BChw: Target feature map to reconstruct (B, C, H, W)
            init_idx_Bl: Initial greedy token indices (list of (B, ph*pw))
            iters: Number of optimization iterations
            tau_start: Initial temperature for softmax
            tau_end: Final temperature for softmax
            entropy_weight: Weight for entropy regularization
            v_patch_nums: Patch numbers for each scale
            lr: Learning rate for Adam optimizer
            fix_scale: Scales >= fix_scale are frozen (hard assignment)
            K: Number of nearest codes to consider per token position

        Returns:
            idx_Bl_all: List of optimized token indices (B, ph*pw) per scale
            f_hat: Reconstructed feature map (B, C, H, W)
            final_mse: Final MSE loss value
        """
        device = f_BChw.device
        B, C, H, W = f_BChw.shape
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]
        assert patch_hws[-1] == (H, W)
        SN = len(patch_hws)

        V = self.vocab_size
        E = self.embedding.weight  # (V, C)

        # Step 1: Find top-K nearest codes for each token position at each scale
        topk_indices = []
        logits = []

        for si, (ph, pw) in enumerate(patch_hws):
            n = ph * pw

            # Get continuous features at this scale
            if si != SN - 1:
                z_NC = F.interpolate(f_BChw, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C)
            else:
                z_NC = f_BChw.permute(0, 2, 3, 1).reshape(-1, C)

            # Compute distances to all codebook entries: ||z - e_k||^2
            d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + \
                       torch.sum(E.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(z_NC, E.T, alpha=-2, beta=1)  # (B*n, V)

            # Get top-K nearest codes for each position
            _, topk_idx = torch.topk(d_no_grad, K, dim=1, largest=False)  # (B*n, K)
            topk_idx = topk_idx.view(B, n, K)

            # Ensure greedy token is in top-K (replace furthest if needed)
            if init_idx_Bl is not None:
                greedy_idx = init_idx_Bl[si].view(B, n, 1)  # (B, n, 1)
                # Check if greedy is already in top-K
                is_in_topk = (topk_idx == greedy_idx).any(dim=-1, keepdim=True)  # (B, n, 1)
                # If not, replace the last (furthest) code with greedy
                topk_idx = torch.where(
                    is_in_topk.expand(-1, -1, K),
                    topk_idx,
                    torch.cat([greedy_idx, topk_idx[:, :, :-1]], dim=-1)
                )

            topk_indices.append(topk_idx)

            # Initialize logits for top-K codes only
            L = torch.zeros(B, n, K, device=device, dtype=torch.float32, requires_grad=True)

            # Bias toward greedy token
            if init_idx_Bl is not None:
                greedy_idx = init_idx_Bl[si].view(B, n, 1)
                # Find position of greedy in top-K
                greedy_mask = (topk_idx == greedy_idx)  # (B, n, K)
                # Set higher logit for greedy position
                L = L.masked_fill(greedy_mask, 5.0)

            logits.append(nn.Parameter(L))

        # Only optimize scales [3, fix_scale)
        opt = torch.optim.Adam(logits[3:fix_scale], lr=lr)

        def build_f_hat_from_probs(prob_list):
            """Build feature map from probability distributions over top-K codes"""
            f_hat = torch.zeros(B, C, H, W, device=device, dtype=torch.float32)
            for si, (ph, pw) in enumerate(patch_hws):
                p = prob_list[si]  # (B, n, K) - probabilities over top-K only

                if si >= fix_scale:
                    # Frozen scales: hard assignment (argmax over top-K)
                    idx_in_topk = torch.argmax(p, dim=-1)  # (B, n)
                    idx_global = torch.gather(topk_indices[si], dim=2,
                                             index=idx_in_topk.unsqueeze(-1)).squeeze(-1)
                    z = E[idx_global]  # (B, n, C)
                else:
                    # Optimizing scales: soft assignment (weighted combination of top-K)
                    topk_embeddings = E[topk_indices[si]]  # (B, n, K, C)
                    z = torch.einsum('bnk,bnkc->bnc', p, topk_embeddings)  # (B, n, C)

                z = z.view(B, ph, pw, C).permute(0, 3, 1, 2).contiguous()  # (B, C, ph, pw)
                if si != SN - 1:
                    z = F.interpolate(z, size=(H, W), mode='bicubic')
                z = self.quant_resi[si/(SN-1)](z)
                f_hat.add_(z)
            return f_hat

        # Temperature annealing schedule
        def temp(t):
            if iters <= 1: return tau_end
            a = t / (iters - 1)
            return tau_start * (tau_end / tau_start) ** a

        # Optimization loop
        with torch.cuda.amp.autocast(enabled=False):
            for t in range(iters):
                tau = temp(t)
                opt.zero_grad(set_to_none=True)

                # Compute probabilities with current temperature
                probs = [F.softmax(L / tau, dim=-1) for L in logits]  # Each: (B, n, K)
                f_hat = build_f_hat_from_probs(probs)
                mse = 0.5 * F.mse_loss(f_hat, f_BChw, reduction='mean')

                # Entropy regularization (encourage peaky distributions)
                ent = 0.0
                for p in probs:
                    ent = ent - (p * (p.clamp_min(1e-12).log())).sum(dim=-1).mean()
                loss = mse + entropy_weight * ent

                loss.backward()
                opt.step()

        # Snap to discrete codes
        with torch.no_grad():
            idx_Bl_all = []
            f_hat = torch.zeros(B, C, H, W, device=device, dtype=torch.float32)
            for si, (ph, pw) in enumerate(patch_hws):
                p = F.softmax(logits[si] / tau_end, dim=-1)  # Final temperature
                idx_in_topk = torch.argmax(p, dim=-1)  # (B, n)
                idx_global = torch.gather(topk_indices[si], dim=2,
                                         index=idx_in_topk.unsqueeze(-1)).squeeze(-1)
                idx_Bl_all.append(idx_global)

                z = self.embedding(idx_global.view(B, ph, pw)).permute(0, 3, 1, 2).contiguous()
                if si != SN - 1:
                    z = F.interpolate(z, size=(H, W), mode='bicubic')
                z = self.quant_resi[si/(SN-1)](z)
                f_hat.add_(z)

            final_mse = F.mse_loss(f_hat, f_BChw)

        return idx_Bl_all, f_hat, final_mse

    def refine_soft_assign_remove(
        self,
        f_BChw: torch.Tensor,
        init_idx_Bl=None,
        iters: int = 150,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
        entropy_weight: float = 1e-3,
        v_patch_nums=None,
        lr: float = 0.1,
        fix_scale=10,
    ):
        """
        Annealed soft-assign refinement of code indices:
        - optimize full MSE with soft assignments p over the codebook,
        - entropy penalty encourages one-hot,
        - temperature anneals from tau_start to tau_end,
        - finally snap to argmax and return indices, f_hat, mse.

        Returns: idx_Bl_all (list of (B, ph*pw)), f_hat (B,C,H,W), mse (scalar tensor)
        """
        device = f_BChw.device
        B, C, H, W = f_BChw.shape
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]
        assert patch_hws[-1] == (H, W)
        SN = len(patch_hws)

        V = self.vocab_size
        E = self.embedding.weight  # (V, C)

        # logits per scale: (B, ph*pw, V)
        logits = []
        for si, (ph, pw) in enumerate(patch_hws):
            if si == fix_scale: break  # only optimize the first `fix_scale` scales
            n = ph * pw
            L = torch.zeros(B, n, V, device=device, dtype=torch.float32, requires_grad=True)
            if init_idx_Bl is not None:
                idx0 = init_idx_Bl[si]  # (B, n)
                with torch.no_grad():
                    L.scatter_(dim=2, index=idx0.unsqueeze(-1), value=0.5)  # bias towards init
            logits.append(nn.Parameter(L))

        opt = torch.optim.Adam(logits, lr=lr)

        def build_f_hat_from_probs(prob_list):
            f_hat = torch.zeros(B, C, H, W, device=device, dtype=torch.float32)
            for si, (ph, pw) in enumerate(patch_hws):
                if si == fix_scale: break  # only optimize the first `fix_scale` scales
                # p: (B, ph*pw, V) ; z: (B, ph*pw, C)
                p = prob_list[si]
                z = p @ E  # expected embedding
                z = z.view(B, ph, pw, C).permute(0, 3, 1, 2).contiguous()  # B,C,ph,pw
                if si != SN - 1:
                    z = F.interpolate(z, size=(H, W), mode='bicubic')
                z = self.quant_resi[si/(SN-1)](z)
                f_hat.add_(z)
            return f_hat

        # anneal schedule
        def temp(t):
            if iters <= 1: return tau_end
            a = t / (iters - 1)
            return tau_start * (tau_end / tau_start) ** a

        # training loop
        with torch.cuda.amp.autocast(enabled=False):
            for t in range(iters):
                tau = temp(t)
                opt.zero_grad(set_to_none=True)

                probs = [F.softmax(L / tau, dim=-1) for L in logits]  # list of (B, n, V)
                f_hat = build_f_hat_from_probs(probs)
                mse = 0.5 * F.mse_loss(f_hat, f_BChw, reduction='mean')

                # low-entropy encouragement -> one-hots
                ent = 0.0
                for p in probs[:fix_scale]:
                    ent = ent - (p * (p.clamp_min(1e-12).log())).sum(dim=-1).mean()
                loss = mse + entropy_weight * ent  # ent <= 0; adding pushes it toward 0 (peaky)

                loss.backward()
                opt.step()

        # snap to discrete codes
        with torch.no_grad():
            idx_Bl_all = []
            f_hat = torch.zeros(B, C, H, W, device=device, dtype=torch.float32)
            for si, (ph, pw) in enumerate(patch_hws):
                if si == fix_scale: break  # only optimize the first `fix_scale` scales
                p = F.softmax(logits[si] / tau_end, dim=-1)  # final temperature
                idx = torch.argmax(p, dim=-1)  # (B, ph*pw)
                idx_Bl_all.append(idx)

                z = self.embedding(idx.view(B, ph, pw)).permute(0, 3, 1, 2).contiguous()
                if si != SN - 1:
                    z = F.interpolate(z, size=(H, W), mode='bicubic')
                z = self.quant_resi[si/(SN-1)](z)
                f_hat.add_(z)

            final_mse = F.mse_loss(f_hat, f_BChw)
        return idx_Bl_all, f_hat, final_mse


    # --- helpers to expose Phi params cleanly ---
    def _phi_params(self, si: int):
        """Return (weight, bias, rho, pad) for Phi at scale si."""
        SN = len(self.v_patch_nums)
        phi = self.quant_resi[si/(SN-1)]
        # Identity branch (no residual conv)
        if isinstance(phi, nn.Identity) or getattr(phi, "resi_ratio", 0.0) == 0.0:
            return None, None, 0.0, 0
        # Phi is nn.Conv2d subclass with .weight/.bias and .resi_ratio
        W = phi.weight
        b = phi.bias
        rho = float(phi.resi_ratio)
        pad = (phi.kernel_size[0] // 2, phi.kernel_size[1] // 2) if isinstance(phi.kernel_size, tuple) else (phi.kernel_size // 2, phi.kernel_size // 2)
        return W, b, rho, pad

    def _patch_hw(self, si: int):
        pn = self.v_patch_nums[si]
        return (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
    

    def _apply_A(self, si: int, z_BChw_small: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Synthesis operator A_s: upsample z to (H,W), then apply quant_resi[si/(SN-1)].
        z_BChw_small: (B,C,ph,pw) at scale s
        returns: (B,C,H,W)
        """
        SN = len(self.v_patch_nums)
        ph, pw = self._patch_hw(si)
        if (ph, pw) != (H, W):
            h = F.interpolate(z_BChw_small, size=(H, W), mode='bicubic')
        else:
            h = z_BChw_small
        h = self.quant_resi[si/(SN-1)](h)
        return h

    # --- linear part only: upsample -> (1-rho)*I + rho*conv_no_bias ---
    def _apply_L(self, si: int, z_small: torch.Tensor, H: int, W: int) -> torch.Tensor:
        Wphi, bphi, rho, pad = self._phi_params(si)
        ph, pw = self._patch_hw(si)
        h = F.interpolate(z_small, size=(H, W), mode='bicubic') if (ph, pw) != (H, W) else z_small
        if rho == 0.0 or Wphi is None:
            return h  # pure identity
        # conv without bias for the linear part
        return h.mul(1.0 - rho) + F.conv2d(h, Wphi, bias=None, stride=1, padding=pad).mul(rho)

    def _bias_map(self, si: int, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return the constant map B_s = rho * b (broadcast to BxCxHxW)."""
        Wphi, bphi, rho, _ = self._phi_params(si)
        if rho == 0.0 or bphi is None:
            return torch.zeros(B, self.Cvae, H, W, device=device, dtype=torch.float32)
        return (bphi.to(torch.float32) * rho).view(1, -1, 1, 1).expand(B, -1, H, W).contiguous()

    # --- adjoint for the linear operator L only (grad-enabled) ---
    def _adjoint_L(self, si: int, r_full: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
        with torch.enable_grad():
            r = r_full.to(torch.float32)
            B, C, H, W = r.shape
            z = torch.zeros(B, C, ph, pw, device=r.device, dtype=torch.float32, requires_grad=True)
            h = self._apply_L(si, z, H, W)                   # linear path only
            s = (h * r.detach()).sum()
            g, = torch.autograd.grad(s, z, retain_graph=False, create_graph=False)
        return g  # (B,C,ph,pw)

    # --- exact recovery on the linearized operator; adds bias back at the end ---
    def recover_indices_exact(self, f_BChw: torch.Tensor, max_passes: int = 6, tol_rel: float = 1e-7,
                            use_atom_norms: bool = False, chunkV: int = 0):
        f = f_BChw.to(torch.float32)
        B, C, H, W = f.shape
        SN = len(self.v_patch_nums)
        device = f.device

        # precompute and subtract the constant bias (once)
        Bsum = torch.zeros(B, C, H, W, device=device, dtype=torch.float32)
        for si in range(SN):
            Bsum.add_(self._bias_map(si, B, H, W, device))
        f_lin = f - Bsum  # what the linear part must reconstruct

        # warm start from your greedy indices
        greedy = self.f_to_idxBl_or_fhat(f, to_fhat=False)
        z_ms, idx_ms = [], []
        for si in range(SN):
            ph, pw = self._patch_hw(si)
            idx_Bhw = greedy[si].view(B, ph, pw).long()
            z = self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous().to(torch.float32)
            z_ms.append(z)
            idx_ms.append(idx_Bhw)

        # optional ||L(e_k)||^2 term (location-agnostic approx)
        atom_norms = [None] * SN
        if use_atom_norms:
            V = self.vocab_size
            for si in range(SN):
                ph, pw = self._patch_hw(si)
                yx = (ph // 2, pw // 2)
                norms = []
                step = 1024 if chunkV == 0 else chunkV
                for a in range(0, V, step):
                    b = min(a + step, V)
                    z = torch.zeros(b - a, C, ph, pw, device=device, dtype=torch.float32)
                    z[:, :, yx[0], yx[1]] = self.embedding.weight[a:b].to(torch.float32)
                    d = self._apply_L(si, z, H, W).flatten(1)
                    norms.append(d.pow(2).sum(dim=1))
                atom_norms[si] = torch.cat(norms, dim=0)

        # residual for the linear problem: r_lin = sum_s L(z_s) - f_lin
        def synth_lin():
            y = f_lin.new_zeros(B, C, H, W)
            for si in range(SN):
                y.add_(self._apply_L(si, z_ms[si], H, W))
            return y

        y = synth_lin()
        r = y - f_lin
        base = f_lin.norm() + 1e-12
        E = self.embedding.weight.data.to(torch.float32)  # raw codebook

        for _ in range(max_passes):
            improved = False
            # sweep small->large then large->small
            for sweep in (range(SN), reversed(range(SN))):
                for si in sweep:
                    ph, pw = self._patch_hw(si)
                    # remove this scale's current linear contribution
                    r.add_(-self._apply_L(si, z_ms[si], H, W))

                    # adjoint of L
                    g = self._adjoint_L(si, r, ph, pw)                       # (B,C,ph,pw)
                    g_flat = g.permute(0, 2, 3, 1).reshape(-1, C)            # (B*ph*pw, C)

                    # select codes: minimize g·e_k (+ 0.5||L(e_k)||^2)
                    if chunkV and self.vocab_size > chunkV:
                        N = g_flat.shape[0]
                        best_val = None; best_idx = None; off = 0
                        while off < self.vocab_size:
                            end = min(off + chunkV, self.vocab_size)
                            logits = g_flat @ E[off:end].T
                            if use_atom_norms and (atom_norms[si] is not None):
                                logits = logits + 0.5 * atom_norms[si][off:end].view(1, -1)
                            vals, idxs = logits.min(dim=1)
                            if best_val is None:
                                best_val, best_idx = vals, idxs + off
                            else:
                                mask = vals < best_val
                                best_idx = torch.where(mask, idxs + off, best_idx)
                                best_val = torch.minimum(best_val, vals)
                            off = end
                        idx_flat = best_idx
                    else:
                        logits = g_flat @ E.T
                        if use_atom_norms and (atom_norms[si] is not None):
                            logits = logits + 0.5 * atom_norms[si].view(1, -1)
                        idx_flat = torch.argmin(logits, dim=1)

                    idx_Bhw_new = idx_flat.view(B, ph, pw).long()
                    if not torch.equal(idx_Bhw_new, idx_ms[si]):
                        improved = True
                        idx_ms[si] = idx_Bhw_new
                        z_ms[si] = self.embedding(idx_Bhw_new).permute(0, 3, 1, 2).contiguous().to(torch.float32)

                    # add back this scale's updated linear contribution
                    r.add_(self._apply_L(si, z_ms[si], H, W))

            if (r.norm() / base) <= tol_rel or not improved:
                break

        # pack outputs and form full residual (add bias back once)
        idx_Bl_all = [idx_ms[si].reshape(B, -1).contiguous() for si in range(SN)]
        full_residual = r  # r is on linearized target f_lin already
        return idx_Bl_all, full_residual


    
    def f_to_idxBl_or_fhat_with_embedding(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'

        embeddings = {
            "current_resolution":{
                "original": [],
                "quantized": []
            },
            "full_resolution":{
                "original": [],
                "quantized": []
            }
        }

        # embedding_scales, embedding_scales_hat = [], []
        # embedding_scales_full_resolution, embedding_scales_hat_full_resolution = [], []

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            # downscaling
            embeddings["full_resolution"]["original"].append(f_rest.clone().detach()) # full resolution: BChw
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            embeddings["current_resolution"]["original"].append(z_NC.clone().detach().view(B, ph, pw, C)) # current resolution: BhwC
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, ph, pw)
            # upscaling
            embeddings["current_resolution"]["quantized"].append(self.embedding(idx_Bhw.clone().detach()).detach()) # current resolution: BhwC
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            embeddings["full_resolution"]["quantized"].append(h_BChw.clone().detach()) # full resolution: BChw (after convolution)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw))
        
        return f_hat_or_idx_Bl, embeddings
    
    def f_to_idxBl_or_fhat_with_f_rest(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        embed_scales = []
        embedhat_scales = []
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_hwC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1) if (si != SN-1) else f_rest.permute(0, 2, 3, 1)
            embed_scales.append(z_hwC.clone()) #BhwC
            z_NC = z_hwC.reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, ph, pw)
            zhat_NC = self.embedding(idx_Bhw) #BhwC
            embedhat_scales.append(zhat_NC.clone())
            h_BChw = F.interpolate(zhat_NC.permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw))
        
        return f_hat_or_idx_Bl, f_rest, embed_scales, embedhat_scales
    
    def embed_to_embedhat(self, ms_h_BChw: List[torch.Tensor]) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_h_BChw = []
        idxBl = []
        B, C = ms_h_BChw[0].shape[0], ms_h_BChw[0].shape[1]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        for si, z_NC in enumerate(ms_h_BChw): # from small to large
            ph, pw = z_NC.shape[2], z_NC.shape[3]
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = z_NC.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            ls_h_BChw.append(h_BChw)
            idxBl.append(idx_N.reshape(B, ph*pw))
        return ls_h_BChw, idxBl

    def single_embed_to_embedhat(self, z_NC: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        B, C = z_NC.shape[0], z_NC.shape[1]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        ph, pw = z_NC.shape[2], z_NC.shape[3]
        # find the nearest embedding
        z_NC = z_NC.permute(0, 2, 3, 1).reshape(-1, C)
        if self.using_znorm:
            z_NC = F.normalize(z_NC, dim=-1)
            idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
        else:
            d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)
        
        idx_Bhw = idx_N.view(B, ph, pw)
        h_BChw = self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
        return h_BChw, idx_N.reshape(B, ph*pw)
    
    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = self.v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'





class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
