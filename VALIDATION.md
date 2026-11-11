# Validation

Executed November 10, 2026 in an isolated CPU environment:

- Python 3.12.14, Torch 2.15.0+cpu, ParticleGAN 0.9.5, pytest 9.2.0.
- Shared core wheel built from clean public source commit `61878caddeb626fbe77def072051276b7b1f92a1`, version 0.14.2. Installed normally with its dependencies.
- Product sdist built, product wheel built from that sdist, and wheel installed with `--no-deps` after the exact shared source was installed. Tests use importlib mode with no source-path injection.
- **50 tests passed**, covering pure-Python planning, source-lock integrity, source/reference separation, exact six-step schedule, installed entry-point discovery, all six shared operation refusals before preparation/PG admission, direct-hook refusal, actual shared projection identity and gradients, bank ownership, nested strength cleanup and CPU RNG preservation.
- `pip check`: no broken requirements.
- No weights downloaded; no ComfyUI native runtime was installed or imported. Source inspection is not native execution.

The synthetic probe is not a GAN-training experiment: zero optimizer updates and no saved adapters. It does not instantiate H3 or generate an image.

Not run or established: native loading, GPU/device admission, quantized attachment/gradients, native PDD sampler parity, trained-slider quality, model rights, hosted CI, model publication or deployment. No CI workflow is configured in this initial scaffold. Python/Torch versions other than the above are declared compatibility ranges, not executed evidence.

Reproduce after installing the wheel:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m pip check
h3-image-slider cpu-check
```
