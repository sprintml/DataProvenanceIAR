# Data Provenance for Image Auto-Regressive Generation

Official code for **"Data Provenance for Image Auto-Regressive Generation"**.

Image auto-regressive models (IARs) generate images as sequences of discrete
tokens drawn from a fixed codebook. This leaves a characteristic, model-specific
trace: when a generated image is inverted back through the decoder and
re-quantized, its features lie unusually close to the model's codebook entries.
We turn this observation into a **post-hoc, model-agnostic provenance framework**
that traces an image to the IAR that produced it — *without any modification to
the model's training or generation*, so it applies even to already-published,
unwatermarked content.

The framework derives three signals from a model's vector-quantized autoencoder:

| Signal | Definition | Code |
| --- | --- | --- |
| **QuantLoss** `L_Z` | `‖ f − f_Z ‖`, with `f = D⁻¹(x)` (inverse decoder) and `f_Z = Q⁻¹(Q(f))` | `dataprov/signals.py` |
| **EncLoss** `L_enc` | `‖x̂ − x‖ / ‖x̂̂ − x̂‖` (calibrated double reconstruction) | `dataprov/signals.py` |
| **Combined** `L_comb` | `L_Z × L_enc` | `dataprov/signals.py` |

The key experiment is **inverse-decoder finetuning**: each model's encoder is
finetuned (decoder + codebook frozen) so it inverts the decoder on generated
images, which sharpens all three signals. For the multi-scale model **VAR** we
use an **optimized quantization** (gradient-descent token search) for QuantLoss.
For **RAR** and **Taming** we additionally support **augmentation finetuning**
and **robustness evaluation** against common image post-processing.

---

## Models

| Model | Paradigm | Resolution | Range | Base weights | VAR-style optim |
| --- | --- | --- | --- | --- | --- |
| LlamaGen | next-token | 384 | [-1,1] | HuggingFace (auto) | – |
| RAR | random-order | 256 | [0,1] | HuggingFace (auto) | – |
| VAR | next-scale | 256 | [-1,1] | HuggingFace (auto) | ✅ optimized quant |
| Taming | next-token | 256 | [-1,1] | manual download | – |
| Infinity | next-scale (bit-wise) | 1024 | [-1,1] | official release | – |

Notes: LlamaGen's *original* encoder is already a good inverse decoder (it
reaches ~100% without finetuning, so use `--encoder original`); Infinity's
strongest signal is QuantLoss with the finetuned encoder, and its generation is
text-to-image (Infinity transformer + FLAN-T5). See [docs/MODELS.md](docs/MODELS.md).

Baselines included: **Reconstruction**, **AEDR** (both = a signal computed with
the *original* encoder, via `--encoder original`), and **LatentTracer**
(`dataprov/baselines.py`).

---

## Repository layout

```
dataprov/                 # the unified, model-agnostic framework
  signals.py              #   QuantLoss / EncLoss / Combined
  metrics.py              #   TPR@1%FPR, AUC
  finetune.py             #   inverse-decoder finetuning (+ augmentation schedule)
  augmentations.py        #   7 post-processing attacks + robustness schedule
  baselines.py            #   LatentTracer
  data.py                 #   image-folder datasets
  config.py               #   tiny YAML config (no heavy deps)
  models/
    base.py               #   BaseIAR interface every model implements
    {llamagen,rar,var,taming,infinity}.py   # per-model wrappers
configs/<model>.yaml      # per-model hyper-parameters (faithful to the paper)
scripts/                  # CLI: generate_data / finetune_encoder / evaluate / ...
third_party/<model>/      # vendored (adapted) upstream model code; see docs/MODELS.md
tests/smoke_framework.py  # framework self-test with a synthetic model
```

The framework (`dataprov/`) only needs `torch, torchvision, numpy, scipy,
scikit-learn, pyyaml, pillow, tqdm, huggingface_hub`, so it drops into each
model's own environment with no extra installs.

---

## Installation

The framework is dependency-light, but the **underlying IAR models require
different, mutually incompatible PyTorch versions**, so each model is run in its
**own conda environment**. General recipe:

```bash
# 1) clone
git clone <PROJECT_REPO_URL> DataProvenanceIAR && cd DataProvenanceIAR

# 2) create a per-model environment (example: VAR)
conda create -n dpiar-var python=3.10 -y
conda activate dpiar-var

# 3) install PyTorch matching the model (see docs/MODELS.md for per-model versions)
pip install torch torchvision            # or a specific +cuXXX build from pytorch.org

# 4) install the framework + that model's extra deps
pip install -e .
pip install -r third_party/var/requirements.txt   # per-model extras
```

Per-model PyTorch versions, extra dependencies, and the exact base-weight
sources are documented in **[docs/MODELS.md](docs/MODELS.md)**.

Optionally point the cache directories somewhere with space:

```bash
export DATAPROV_CHECKPOINTS=/path/to/checkpoints   # default ./checkpoints
export DATAPROV_DATA=/path/to/data                 # default ./data
```

---

## Quickstart: framework self-test (no weights needed)

```bash
python tests/smoke_framework.py
# -> TPR@1%FPR=1.000, finetuning loop runs, all signals/baselines/augmentations OK
```

---

## End-to-end workflow

All commands take the model name as the first argument (`llamagen`, `rar`,
`var`, `taming`, `infinity`) and read hyper-parameters from `configs/<model>.yaml`
(override anything with `--set key=value`).

### 1. Generate belonging + finetuning data

Generate images with the target IAR. Each image is saved with its quantized
feature map `f_Z` as the finetuning target.

```bash
python scripts/generate_data.py var --n 1000 --out data/var_generated
```

### 2. Finetune the inverse decoder (key experiment)

```bash
python scripts/finetune_encoder.py var --data data/var_generated
# -> checkpoints/var/encoder_final.pth
```

Robustness (augmentation) finetuning for RAR / Taming — applies the progressive
weak→medium→strong schedule:

```bash
python scripts/finetune_encoder.py rar --data data/rar_generated \
    --augment --out checkpoints/rar_aug
```

### 3. Evaluate provenance (main result: TPR@1%FPR)

Provide the belonging set (images the target model generated) and one or more
non-belonging sets (natural datasets or other models' images):

```bash
python scripts/evaluate.py var \
    --belonging    data/var_generated \
    --nonbelonging data/imagenet data/llamagen_generated data/rar_generated
```

```
TPR@1%FPR (%)  |  model=var  encoder=finetuned

signal                  imagenet   llamagen_generated   rar_generated
--------------------------------------------------------------------
reconstruction              ...           ...               ...
quant_loss                  ...           ...               ...
enc_loss                    ...           ...               ...
combined                    ...           ...               ...
```

### 4. Baselines

```bash
# Reconstruction + AEDR = signals with the ORIGINAL encoder; LatentTracer via --baselines
python scripts/evaluate.py var --encoder original --baselines \
    --belonging data/var_generated --nonbelonging data/imagenet
```

### 5. Robustness to post-processing (RAR / Taming)

Robustness uses QuantLoss with the **augmentation-finetuned** encoder
(`encoder_aug.pth`). Attacks are applied faithfully to the paper
(`demo_ours.py`): open → squash to the model resolution → attack on the **PIL**
image, with the Gaussian blur sampling a **random sigma in (0.1, 2.0)** per image.

```bash
# all attacks at the paper's default strengths, with the augmentation-finetuned encoder
python scripts/robustness_eval.py rar --signal quant_loss \
    --set finetuned_encoder=checkpoints/rar/encoder_aug.pth \
    --belonging data/rar_generated --nonbelonging data/coco

# sweep a single attack across its full strength range
python scripts/robustness_eval.py rar --attacks jpeg --strength sweep \
    --set finetuned_encoder=checkpoints/rar/encoder_aug.pth \
    --belonging data/rar_generated --nonbelonging data/coco
```

Pre-materialize attacked datasets instead of attacking on the fly:

```bash
python scripts/make_augmented_data.py data/rar_generated          # paper strengths
python scripts/make_augmented_data.py data/imagenet --attacks jpeg resize --strengths 60 0.5
```

---

## Finetuned encoder checkpoints (HuggingFace)

We release the finetuned inverse-decoder encoders as **full state dicts** in a
single HuggingFace model repo:

```
<model>/encoder_final.pth     # main-table inverse decoder (all 5 models)
rar/encoder_aug.pth           # augmentation-finetuned encoder (robustness)
taming/encoder_aug.pth        # augmentation-finetuned encoder (robustness)
```

Download and use:

```bash
python scripts/download_checkpoints.py var --what encoder \
    --hf-repo <user>/dataprovenance-iar-encoders
```

Maintainers: `scripts/prep_release_checkpoint.py` converts an existing finetuned
checkpoint into this format. A ready-to-upload folder (all seven files + a model
card documenting each one's source) can be uploaded in one shot:

```bash
huggingface-cli login
huggingface-cli upload <user>/dataprovenance-iar-encoders <folder> . --repo-type model
# or per file:  python scripts/upload_checkpoints.py var --encoder-path ... --hf-repo ... --create
```

---

## Datasets

* **Non-belonging (natural):** the 1,000-image validation subsets of ImageNet,
  LAION (LAION-POP), and MS-COCO 2014 used in the paper. Place each as a flat
  folder of images and pass it as a `--nonbelonging` argument.
* **Belonging / other-model:** generated with `scripts/generate_data.py`.

Any flat folder of images works; resolution/normalization is handled per model.

---

## Method ↔ code map

| Paper | Code |
| --- | --- |
| QuantLoss `L_Z` (Eq. 4) | `dataprov/signals.py::provenance_signals` |
| Calibrated EncLoss (Eq. 8) | `dataprov/signals.py::provenance_signals` |
| Combined Loss (Eq. 9) | `dataprov/signals.py::provenance_signals` |
| Inverse-decoder finetuning (Eq. 5) | `dataprov/finetune.py` |
| Optimized quantization for VAR (Alg. 3) | `dataprov/models/var.py` |
| Augmentation schedule (Tab. A2) | `dataprov/augmentations.py::AUG_SCHEDULE` |
| TPR@1%FPR | `dataprov/metrics.py::tpr_at_fpr` |

---

## Acknowledgements & licenses

The per-model wrappers build on the official model releases (LlamaGen, RAR/TiTok,
VAR, Taming Transformers, Infinity). The relevant model code is vendored under
`third_party/` with its original license; see `docs/MODELS.md` for sources. This
repository's own code is released under the MIT license.

## Citation

```bibtex
@inproceedings{zhao2026dataprovenance,
  title     = {Data Provenance for Image Auto-Regressive Generation},
  author    = {Zhao, Bihe and Kerner, Louis and Meintz, Michel and Bakr, Tameem
               and Boenisch, Franziska and Dziedzic, Adam},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
}
```
