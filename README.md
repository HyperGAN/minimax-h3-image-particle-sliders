# MiniMax-H3-Image Particle Sliders

**Memory (calculated):** public planner: **0 GPU /1 GiB RAM**. Separate official native FP4 profile: training **32 GiB VRAM /32 GiB RAM** at2048² batch2/text≤256 (**40 GiB VRAM** at text4096); sampling **24 GiB VRAM /24 GiB RAM**; conditioning separately **32 GiB VRAM /64 GiB RAM**. See [MEMORY.md](MEMORY.md) for the complete budgets and assumptions.





Offline planner and native FP4 scheduling helper for [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image), using the shared [Particle Sliders](https://github.com/HyperGAN/particle-sliders) library. This is the official **November 10, 2026 image release**, not Pruna P-H3-Image and not H3 video/audio.

The installed shared lifecycle provides offline planning and provenance.
The optional scheduling helper runs inside a separately staged native FP4
pipeline. That pipeline supplies the denoiser, VAE, base weights and attached
RAW adapter. The package ships the helper and the measured comparison, with
model and adapter weights supplied by the caller.
[Native scheduling and measured result](#scheduling-on-an-owned-fp4-image-pipeline).

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

The project pins `particle-sliders-core` **0.15.2** at full commit `ec1161f5d74481be3540cb0dfa63d0e8f8b54e90`; that source requires official `particlegan==0.11.1`. No second GAN implementation is vendored. ComfyUI is inspected at an exact source pin but is neither installed nor imported by this package. This is an explicitly new source cohort; old checkpoints and P-H3 results are not rebound to it.

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

The separately staged native runner owns model construction, conditioning, packing and admission. The scheduling helper preserves its PDD sampler and adapter ownership. Shared lifecycle and adapter primitives remain in Particle Sliders; GAN mathematics remain in ParticleGAN. Existing `p-h3-image-particle-sliders` and `minimax-h3-particle-sliders` are separate products and were not modified.

Code is MIT. Model rights remain separate. No deployment or weight publication is part of this repository setup. See [archived validation](https://github.com/HyperGAN/minimax-h3-image-particle-sliders/blob/4aa24e62c44518b6eb854efdb1a034252fbac718/VALIDATION.md) for executed versus unrun checks.

## Scheduling on an owned FP4 image pipeline

The native FP4 execution owner can call
`minimax_h3_image_sliders.scheduled_native.denoise_with_gains` with its
original PDD denoiser, attached RAW adapter and six gains. Each gain uses
the original PDD head and its native timestep. The returned image latent
can be passed to the same native VAE stage. This helper requires the
separate native FP4 runner and its existing conditioning and resource
checks. Effective strengths must lie in `[0,1]`.

Early-heavy strength improved both matched cases of the Bioluminescence
G-half RAW8 profile. On the original 2048-pixel FP4 PDD6 route with its
FP32 VAE, the mean finished-image gain was +0.00226 early-heavy,
+0.00082 constant, and +0.00141 late-heavy at nominal strength 0.5.
All six images passed independent integrity.
[Per-case evidence](evidence/bioluminescence-fp4-raw8-scheduled-strength.json)

### Activation-dependent strength

Pass an `ImageGatePlan` as `gate_plan` to the same helper. At each native
PDD head, it averages the live packed image rows in FP32, yielding a
96-channel readout before entering that head's RAW adapter scope. CFP and CLABS
use a fitted unit direction for that readout. The matched random control
uses its calibrated target rate and a separate gate seed. The shared gate
chooses between OFF and the caller's positive nominal strength. The receipt
records each decision, the effective strengths and the sigma-integrated dose.
[Shared equations and active-budget calibration](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#cfp-and-clabs)

The separately staged FP4 runner can load the reviewed `scheduled_native.py`
by filename, as in the archived fixed-schedule run. The plan reads and verifies
the steering module in a wheel built from [core PR #269](https://github.com/HyperGAN/particle-sliders/pull/269).
The original owner keeps its installed dependencies and checkpoint bindings.

```python
from runpy import run_path

helper = run_path(verified_scheduling_source)
gate_plan = helper["ImageGatePlan"](
    core_wheel=reviewed_core_269_wheel,
    method="cfp",
    parameter=calibrated_threshold,
    direction=fitted_packed_image_direction,
    calibration_sha256=calibration_record_sha256,
)
latent, trace = helper["denoise_with_gains"](
    denoise, transformer, conditioning, noise,
    scale_adapter=raw_adapter.scaled, strength=qualified_strength,
    gate_plan=gate_plan, check=check,
)
```

CLABS takes an explicit bias, seed and `clabs` calibration fields. For
fixed-strength replay, divide each recorded effective strength by the
nominal strength to obtain its `step_gains` value. Fit the contrastive
packed-latent direction on separate matched attribute and neutral
trajectories, then calibrate the gate with steering active on disjoint cases.

The Bioluminescence RAW64 CFP profile gained +0.00145 over its equal-dose
constant on the two held-out 2048-pixel cases. It improved both cases and
passed independent integrity. The separate RAW8 fixed-schedule result
above uses its original early-heavy profile.
[Calibrated RAW64 CFP replay](docs/BIOLUMINESCENCE-RAW64-REPLAY.md) ·
[Per-case evidence](evidence/bioluminescence-fp4-raw64-adaptive-strength.json)
