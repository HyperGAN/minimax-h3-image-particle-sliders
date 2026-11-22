# Replay the calibrated Bioluminescence RAW64 gates

On the two 2048-pixel Bioluminescence RAW64 cases, CFP gained +0.00250844
over OFF and +0.00145467 over its equal-dose constant. It improved both
cases over that constant. The original strength-one arm gained +0.00182409
over OFF. CLABS gained +0.00121295 over OFF and +0.00022143 over its own
equal-dose constant, with one case ahead and one behind. All ten distinct
images in the gated and dose-matched comparison passed independent integrity.
[Per-case results and calibrated settings](../evidence/bioluminescence-fp4-raw64-adaptive-strength.json)

CFP fired on 12 of 24 calibration intervals. CLABS fired on 13 of 24,
within the study's one-interval tolerance of the 50% target. Both examples
use the saved 96-channel direction and the exact RAW checkpoint. Select
`cfp` for the measured threshold or `clabs` for the measured bias and
feedback configuration. The replay runs inside the original FP4 PDD6
owner. Use the original
conditioning, noise preparation, adapter and guard from that owner. Set
`verified_scheduling_source` to the reviewed `scheduled_native.py`, and
`core_269_wheel` to the verified wheel from
[core PR #269](https://github.com/HyperGAN/particle-sliders/pull/269).

```python
from hashlib import file_digest
import json
from pathlib import Path
from runpy import run_path

import torch


def measured_plan(helper, replay, case, method, core_wheel):
    if method not in {"cfp", "clabs"}:
        raise ValueError("Choose the measured CFP or CLABS plan")
    settings = replay[method]
    return helper["ImageGatePlan"](
        core_wheel=core_wheel,
        method=method,
        parameter=settings["threshold" if method == "cfp" else "bias"],
        clabs=settings.get("parameters"),
        direction=replay["direction"],
        calibration_sha256=replay["calibration_sha256"],
        seed=case["seed"],
    )


profile = json.loads(Path("evidence/bioluminescence-fp4-raw64-adaptive-strength.json").read_text())
method = "clabs"  # Use "cfp" to replay that measured gate.
case = profile["methods"][method]["cases"][0]
replay = profile["replay"]
with Path(verified_scheduling_source).open("rb") as source:
    if file_digest(source, "sha256").hexdigest() != replay["scheduling_source_sha256"]:
        raise ValueError("Use the reviewed FP4 scheduling helper")
helper = run_path(verified_scheduling_source)
if list(helper["PDD_SIGMAS"]) != profile["actual_sigmas"]:
    raise ValueError("Use the original FP4 PDD6 sigma grid")

with Path(raw_path).open("rb") as checkpoint:
    if file_digest(checkpoint, "sha256").hexdigest() != profile["raw_sha256"]:
        raise ValueError("Use the measured Bioluminescence RAW64 checkpoint")
raw = torch.load(raw_path, map_location="cpu", weights_only=True)
attached = rt.adapter.state_dict()
if (
    raw["native_model"] != rt.model_identity
    or raw["ema_decay"] is not None
    or raw["weights"] != "raw"
    or raw["step"] != 64
    or raw["network"].keys() != attached.keys()
    or any(value.dtype != attached[name].dtype
           or not torch.equal(value, attached[name].detach().cpu())
           for name, value in raw["network"].items())
):
    raise ValueError("Use the original FP4 model and attached RAW64 tensors")

gate = measured_plan(helper, replay, case, method, core_269_wheel)
embedding = condition.embedding(case["prompt"])
noise = prepare_noise(
    embedding, height=2048, width=2048, min_generation_area=2048**2,
    device=device, generator=torch.Generator(device=device).manual_seed(case["seed"]),
)
with torch.no_grad():
    latent, trace = helper["denoise_with_gains"](
        denoise, rt.transformer, embedding, noise,
        scale_adapter=rt.network.scaled, strength=profile["nominal_strength"],
        gate_plan=gate, check=check,
    )
if (
    trace["sigmas"] != profile["actual_sigmas"]
    or trace["effective_strengths"] != case["gate"]["effective_strengths"]
    or trace["sigma_integrated_dose"] != case["equal_integrated_dose"]
):
    raise RuntimeError("The trace differs from the measured Bioluminescence case")
```

Pass the returned latent to the original FP32 VAE and retain the trace. For
the timing comparison, use the same case and set every PDD head to the
recorded sigma-dose-matched constant. The study separates this comparison
from gain over matched OFF and from the categorical integrity verdicts.

The CLABS gate fired at only the second PDD interval in both held-out cases.
The shared runtime reproduced the saved probabilities and decisions on all
24 intervals of the accepted CLABS calibration trial and all 12 held-out
CLABS intervals. Each measured test case uses its original seed.
