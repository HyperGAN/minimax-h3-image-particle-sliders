# MiniMax-H3-Image Particle Sliders

Source-only integration scaffold for [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image), using the shared [Particle Sliders](https://github.com/HyperGAN/particle-sliders) library. This is the official **November 10, 2026 image release**, not Pruna P-H3-Image and not H3 video/audio.

**Not a trained slider or runnable native renderer.** This first version provides an offline request planner, packaged source provenance, an installed shared-lifecycle registration, and synthetic CPU software checks. Native downloading, loading, training, evaluation, export and rendering are unavailable. No model weights, native model implementation, generated images, or trained adapters are included.

The [model license](docs/LICENSE-REVIEW.md) restricts territories and commercial/hosted use. Model authorization has not been established by this repository. Source publication does not authorize model use.

## Install

Use an isolated Python >=3.11 environment and install your preferred CPU PyTorch >=2.6,<3 build first. Then:

```sh
python -m pip install '.[dev]'
h3-image-slider plan --prompt 'A documentary photograph of a ceramic cup beside a window.'
h3-image-slider provenance
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 h3-image-slider cpu-check
python -m pytest -q
```

The project pins `particle-sliders-core` **0.14.2** at full commit `61878caddeb626fbe77def072051276b7b1f92a1`; that source requires official `particlegan==0.9.5`. No second GAN implementation is vendored. ComfyUI is inspected at an exact source pin but is neither installed nor imported by this package. This is an explicitly new source cohort; old checkpoints and P-H3 results are not rebound to it.

The planner also works without Torch or native dependencies:

```sh
PYTHONPATH=src python -m minimax_h3_image_sliders.cli plan \
  --prompt 'Edit <Picture 1> to add warm window light.' \
  --reference target.png --reference palette.png --strength 0.5
```

Reference arguments are ordered **labels only**. This command does not open files. The first label is the proposed editing target; up to sixteen references are accepted. Strengths are proposed unipolar controls in [0,1], not a quality-qualified range.

## Native contract being planned

- Image-only 50-layer H3 transformer, with no audio stream.
- H3 Qwen3-VL conditioning, using the 50th-layer hidden states. Do not substitute Pruna's conditioning or silently expand paired prompts differently.
- Image VAE: 24 channels, 16× spatial reduction. A still-image latent is `[1,24,1,H/16,W/16]`, with 2×2 latent patches, hence 32×32 pixels per image token.
- Six-step distilled PDD head bank and its exact seven-value sigma grid, no CFG or negative prompt. P-H3's eight-step grid and H3 video's shift are not substitutes.
- Packed text, reference images, then target image. Preserve reference order and native modality tags/positions.
- The official release advertises approximately 16 MP and up to sixteen reference images. The inspected ComfyUI node defaults to 2048×2048 and exposes a larger UI reference limit. This planner defaults to 4096×4096, caps references at the model-card limit, and records that discrepancy rather than claiming either size was tested. Its 17-million-pixel budget is an offline planning guard, not a native memory guarantee.
- RAW weights only for future integration. No weight EMA, checkpoint averaging or EMA comparison.

See [native boundary and next steps](docs/NATIVE-INTEGRATION.md) and the installed [source lock](src/minimax_h3_image_sliders/source-lock.json).

## Shared lifecycle

```python
from particle_sliders.lifecycle import SessionConfig, backend
config = SessionConfig('minimax-h3-image', 'new-source-cohort', ema_decay=None)
print(backend('minimax-h3-image').describe(config))
```

The registered backend reports `operations=()`. All six common operations refuse **before** product validation, PG admission, preparation, CUDA selection or weight allocation. Direct prepare/validate hooks also refuse. There is no acknowledgment flag or environment-variable bypass.

The CPU fixture uses the actual shared `TinyParticleProjection` on a random 8×8 frozen Linear and its actual 128×4 particle bank. It checks zero-strength and initialization identity, fractional strength, gradient participation, single bank ownership, cleanup and RNG preservation. It performs no optimizer step and saves no adapter. These checks do not qualify H3 loading, training, GPU memory, quantized gradients, native rendering, useful concept control or image quality.

## Ownership and release scope

This product owns future native construction, conditioning, packing, PDD/sampler fidelity and model-specific admission. Shared lifecycle and adapter primitives remain in Particle Sliders; GAN mathematics remain in ParticleGAN. Existing `p-h3-image-particle-sliders` and `minimax-h3-particle-sliders` are separate products and were not modified.

Code is MIT. Model rights remain separate. No deployment or weight publication is part of this repository setup. See [validation](VALIDATION.md) for executed versus unrun checks.
