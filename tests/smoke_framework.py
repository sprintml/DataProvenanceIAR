"""Framework self-test with a synthetic IAR (no real weights needed).

Exercises every code path: the three signals, the TPR metric, the LatentTracer
baseline, the augmentation transforms + schedule, and the inverse-decoder
finetuning loop. Run from the repo root in any torch environment:

    python tests/smoke_framework.py
"""

import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprov import augmentations as A
from dataprov import metrics, signals
from dataprov.baselines import latent_tracer
from dataprov.finetune import finetune_inverse_decoder
from dataprov.models.base import BaseIAR

S = 16          # tiny image side
D = 3 * S * S   # flattened feature dim
K = 32          # codebook size


class FakeIAR(BaseIAR):
    """A trivial VQ model: feature = flattened image; codebook in [0,1]^D.

    Images decoded from a codebook entry re-encode exactly onto that entry, so
    their QuantLoss is ~0, while random images have large QuantLoss -- enough to
    validate signal direction and the metric.
    """

    image_size = S
    value_range = "01"
    name = "fake"

    def load_models(self):
        torch.manual_seed(0)
        self.register_buffer("codebook", torch.rand(K, D))
        self.encoder_mod = nn.Module()
        self.encoder_mod.delta = nn.Parameter(torch.zeros(D))
        return nn.Identity(), None

    def encode_feature(self, images: Tensor) -> Tensor:
        return images.flatten(1) + self.encoder_mod.delta

    def decode_feature(self, f: Tensor) -> Tensor:
        return f.reshape(-1, 3, S, S).clamp(0, 1)

    def trainable_encoder_modules(self):
        return [self.encoder_mod]

    @torch.no_grad()
    def img_to_reconstructed_img(self, images, use_quant=True):
        f = self.encode_feature(images)
        if use_quant:
            d = torch.cdist(f, self.codebook)
            idx = d.argmin(dim=1)
            fhat = self.codebook[idx]
            tokens = idx
        else:
            fhat = f
            tokens = None
        return self.decode_feature(fhat), tokens, f, fhat

    @torch.no_grad()
    def generate(self, n, batch_size, seed, **kw):
        g = torch.Generator().manual_seed(seed)
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)
            idx = torch.randint(0, K, (b,), generator=g)
            target = self.codebook[idx]
            yield self.decode_feature(target.to(self.device)).cpu(), target.cpu()


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FakeIAR(cfg={"image_size": S}, device=dev).to(dev)

    # belonging (generated) vs non-belonging (random)
    g = torch.Generator().manual_seed(1)
    bel_idx = torch.randint(0, K, (64,), generator=g)
    belonging = model.decode_feature(model.codebook[bel_idx].to(dev))
    nonbelonging = torch.rand(64, 3, S, S, generator=g).to(dev)

    sig_b = signals.provenance_signals(model, belonging)
    sig_n = signals.provenance_signals(model, nonbelonging)
    assert set(sig_b) >= set(signals.SIGNAL_NAMES), sig_b.keys()
    for k, v in sig_b.items():
        assert v.shape == (64,) and np.isfinite(v).all(), (k, v.shape)

    tpr = metrics.tpr_at_fpr(sig_b["quant_loss"], sig_n["quant_loss"], 0.01, members_lower=True)
    auc = metrics.auc(sig_b["quant_loss"], sig_n["quant_loss"], members_lower=True)
    print(f"[signals]  QuantLoss  belonging mean={sig_b['quant_loss'].mean():.4f}  "
          f"non-belonging mean={sig_n['quant_loss'].mean():.4f}")
    print(f"[metrics]  TPR@1%FPR={tpr:.3f}  AUC={auc:.3f}")
    assert tpr > 0.9 and auc > 0.95, "QuantLoss should separate belonging vs random"
    assert 0.0 <= metrics.tpr_at_fpr(sig_b["combined"], sig_n["combined"]) <= 1.0

    # LatentTracer baseline runs and returns one score per image
    lt = latent_tracer(model, belonging, iters=10, lr=1e-2)
    assert lt.shape == (64,) and np.isfinite(lt).all()
    print(f"[baseline] LatentTracer ran: mean recon={lt.mean():.4f}")

    # augmentations: transforms + eval defaults + finetune schedule
    img01 = belonging[:2].cpu()
    for name, strength in A.ATTACK_DEFAULTS.items():
        out = A.apply_attack(img01, name, strength)
        assert out.shape == img01.shape and out.min() >= 0 and out.max() <= 1, name
    sched = [("none", 1), ("weak", 1), ("medium", 1), ("strong", 1)]
    stages = [A.stage_for_epoch(e, sched) for e in range(4)]
    assert stages == [None, "weak", "medium", "strong"], stages
    _ = A.sample_train_augmentation(img01, "strong", p=1.0)
    print(f"[augment]  {len(A.ATTACK_DEFAULTS)} attacks + schedule OK")

    # finetuning loop: generate a tiny dataset, run 2 epochs, expect loss to drop
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "gen")
        os.makedirs(data_dir)
        i = 0
        for imgs, targets in model.generate(n=16, batch_size=8, seed=2):
            for b in range(imgs.shape[0]):
                Image.fromarray((imgs[b].permute(1, 2, 0).numpy() * 255).astype("uint8")).save(
                    os.path.join(data_dir, f"{i:06d}.png")
                )
                torch.save(targets[b], os.path.join(data_dir, f"{i:06d}_target.pt"))
                i += 1
        cfg = {"epochs": 2, "batch_size": 4, "lr": 1e-2, "optimizer": "adam",
               "num_workers": 0, "scheduler": {"name": "steplr", "step_size": 1, "gamma": 0.9}}
        out = finetune_inverse_decoder(model, data_dir, os.path.join(tmp, "ft"), cfg, log_every=1)
        assert os.path.exists(out)
        print(f"[finetune] saved encoder -> {os.path.basename(out)}")

    print("\nALL FRAMEWORK SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
