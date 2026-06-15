# Per-model setup

Each IAR model ships its (adapted, inference-only) upstream code under
`third_party/<model>/`, imported automatically by the wrapper — you do **not**
need to set `PYTHONPATH`. Because the models require different, mutually
incompatible PyTorch versions, run **each model in its own conda environment**.

The framework itself (`dataprov/`) needs only `torch, torchvision, numpy, scipy,
scikit-learn, pyyaml, pillow, tqdm, huggingface_hub` — all of which the
per-model environments below already include.

| Model | PyTorch | Extra deps | Base weights | Generation |
| --- | --- | --- | --- | --- |
| LlamaGen | ≥2.4 | – | HuggingFace (auto) | ✅ wired |
| RAR | ≥2.5 | `omegaconf`, `timm` | HuggingFace (auto) | ✅ wired |
| VAR | ≥2.2 | – | HuggingFace (auto) | ✅ wired |
| Taming | ≥2.5 | `omegaconf` | manual download | ✅ wired (net2net) |
| Infinity | ≥2.5 | `omegaconf`, `transformers` | official release | ✅ wired (text-to-image) |

After creating an environment, install the framework with `pip install -e .`.

---

## LlamaGen

```bash
conda create -n dpiar-llamagen python=3.10 -y && conda activate dpiar-llamagen
pip install torch torchvision && pip install -e .
```

Base weights (`vq_ds16_c2i.pt`, `c2i_XL_384.pt`) auto-download from
`FoundationVision/LlamaGen`. Note: LlamaGen's **original encoder is already a
good inverse decoder** (the paper achieves ~100% without finetuning), so the
`--encoder original` path alone reproduces its result.

```bash
python scripts/generate_data.py  llamagen --n 1000 --out data/llamagen_generated
python scripts/finetune_encoder.py llamagen --data data/llamagen_generated   # optional
python scripts/evaluate.py llamagen --encoder original \
    --belonging data/llamagen_generated --nonbelonging data/coco
```

## RAR

```bash
conda create -n dpiar-rar python=3.10 -y && conda activate dpiar-rar
pip install torch torchvision omegaconf timm && pip install -e .
```

Base weights auto-download: tokenizer from `fun-research/TiTok`
(`maskgit-vqgan-imagenet-f16-256.bin`), generator from `yucornetto/RAR`
(`rar_xxl.bin`). RAR needs inverse-decoder finetuning, and supports robustness
finetuning (`--augment`) + `scripts/robustness_eval.py`.

## VAR

```bash
conda create -n dpiar-var python=3.10 -y && conda activate dpiar-var
pip install torch torchvision && pip install -e .
```

Base weights (`vae_ch160v4096z32.pth`, `var_d30.pth`) auto-download from
`FoundationVision/var`. VAR uses the **optimized quantization** for QuantLoss
(`token_optim: true`); the total iteration count is `optim_iter × 6` (default
`200 × 6 = 1200`, the paper setting). Reduce `optim_iter` to trade accuracy for
speed (100 → ~87% TPR at ~10× lower latency).

## Taming

Taming weights are **not** on HuggingFace. Download the class-conditional
ImageNet release (`cin_transformer`) and lay it out as:

```
<taming_dir>/configs/vqgan.yaml          # VQGAN (f=16, 16384) config
<taming_dir>/configs/net2net.yaml        # transformer config (for generation)
<taming_dir>/checkpoints/vqgan.ckpt      # VQGAN weights
<taming_dir>/checkpoints/net2net.ckpt    # transformer weights
```

Sources (from the official taming-transformers repo): VQGAN
`https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/`, transformer
`https://app.koofr.net/links/90cbd5aa-ef70-4f5e-99bc-f12e5a89380e`.

```bash
conda create -n dpiar-taming python=3.10 -y && conda activate dpiar-taming
pip install torch torchvision omegaconf && pip install -e .
python scripts/evaluate.py taming --set model_dir=<taming_dir> \
    --belonging data/taming_generated --nonbelonging data/coco
```

Eval/finetuning load only the VQGAN (`configs/vqgan.yaml` + `checkpoints/vqgan.ckpt`);
generation additionally loads the net2net transformer.

## Infinity

Infinity weights come from the official Infinity release: the bit-wise VAE
`infinity_vae_d32reg.pth` and (for generation) `infinity_2b_reg.pth` +
FLAN-T5-XL. Point the config at the VAE:

```bash
conda create -n dpiar-infinity python=3.10 -y && conda activate dpiar-infinity
pip install torch torchvision omegaconf && pip install -e .
python scripts/evaluate.py infinity --signals quant_loss --set vae_path=<infinity_vae_d32reg.pth> \
    --belonging data/infinity_generated --nonbelonging data/coco
```

Infinity's strongest signal is **QuantLoss** with the finetuned encoder. Eval and
finetuning need only the VAE. Generation is **text-to-image**: set `model_path`
(the 2B transformer) and `text_encoder` (a FLAN-T5-XL path), then

```bash
python scripts/generate_data.py infinity --n 1000 --out data/infinity_generated \
    --set vae_path=<vae> model_path=<infinity_2b_reg.pth> \
          text_encoder=<flan-t5-xl> prompt_file=<prompts.txt>
```

One image is generated per prompt (`prompt_file`, one per line; a small built-in
prompt set is used if omitted). Requires `transformers` for FLAN-T5.

---

## Licensing

The vendored code under `third_party/` retains its upstream license (LlamaGen,
RAR/1d-tokenizer — Apache-2.0; VAR — MIT; taming-transformers — MIT; Infinity —
MIT). Only inference-relevant modules are kept; training-only code (losses,
discriminators, data loaders) was removed.
