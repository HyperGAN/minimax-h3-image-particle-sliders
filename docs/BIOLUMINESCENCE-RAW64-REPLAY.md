# Replay the calibrated Bioluminescence RAW64 CFP gate

On the two 2048-pixel Bioluminescence RAW64 cases, CFP gained +0.00250844
over OFF and +0.00145467 over its equal-dose constant. It improved both
cases over that constant. The original strength-one arm gained +0.00182409
over OFF. CLABS gained +0.00121295 over OFF and +0.00022143 over its own
equal-dose constant, with one case ahead and one behind. All ten distinct
images in the gated and dose-matched comparison passed independent integrity.
[Per-case results and calibrated settings](../evidence/bioluminescence-fp4-raw64-adaptive-strength.json)

CFP reached 12 of 24 decisions on four separate calibration cases.
The example uses the saved 96-channel direction, measured threshold and exact
RAW checkpoint. It runs inside the original FP4 PDD6 owner. Use the original
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

profile = json.loads(Path("evidence/bioluminescence-fp4-raw64-adaptive-strength.json").read_text())
case = profile["methods"]["cfp"]["cases"][0]
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

gate = helper["ImageGatePlan"](
    core_wheel=core_269_wheel,
    method="cfp",
    parameter=replay["cfp"]["threshold"],
    direction=replay["direction"],
    calibration_sha256=replay["calibration_sha256"],
    seed=case["seed"],
)
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
```

Pass the returned latent to the original FP32 VAE and retain the trace. For
the timing comparison, use the same case and set every PDD head to the
recorded sigma-dose-matched constant. The study separates this comparison
from gain over matched OFF and from the categorical integrity verdicts.
