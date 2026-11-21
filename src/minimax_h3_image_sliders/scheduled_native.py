"""Opt-in gain control for a caller-owned native FP4 H3 Image denoiser."""

from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from itertools import pairwise
from math import fsum
from sys import float_info, modules
from threading import Lock
from types import ModuleType
from uuid import uuid4
from zipfile import ZipFile


# The fixed grid comes from the published H3 Image PDD configuration and is
# checked against the packaged source lock in tests. A staged RAW owner can
# load this helper by filename without resolving imports through a new package.
PDD_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
              0.5327739119529724, 0.23200887441635132, 0.0)
CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
CORE_MEMBER = "particle_sliders/adaptive_steering.py"
CORE_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"
PACKED_CHANNELS = 96  # The native image layout patches 24 channels over 1 x 2 x 2.
# Independent filename loads own independent locks and private module identities.
_MODULE_NAME = f"minimax_h3_image_sliders._verified_core_{CORE_REVISION}_{uuid4().hex}"
_LOAD_LOCK = Lock()


def gate_core(core_wheel):
    """Expose only the authenticated core #269 module from an explicit wheel."""
    with ZipFile(core_wheel) as archive:
        source = archive.read(CORE_MEMBER)
    if sha256(source).hexdigest() != CORE_SHA256:
        raise ValueError("H3 Image requires the exact adaptive steering source from core PR #269")
    with _LOAD_LOCK:
        if _MODULE_NAME not in modules:
            module = ModuleType(_MODULE_NAME)
            modules[_MODULE_NAME] = module
            try:
                exec(compile(source, CORE_MEMBER, "exec"), module.__dict__)
            except BaseException:
                del modules[_MODULE_NAME]
                raise
        return modules[_MODULE_NAME]


def packed_image_readout(hidden_states):
    """Mean of the live image rows in FP32, before applying the RAW adapter."""
    import torch

    if (not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_floating_point()
            or hidden_states.ndim != 3 or hidden_states.shape[0] != 1
            or hidden_states.shape[1] == 0 or hidden_states.shape[2] != PACKED_CHANNELS):
        raise ValueError("Expected one native H3 Image packed latent with 96 channels")
    return hidden_states.mean(dim=(0, 1), dtype=torch.float32)


class ImageGatePlan:
    """Explicit readout direction and active-budget calibration for FP4 PDD6."""

    def __init__(self, *, core_wheel, method, parameter, calibration_sha256,
                 direction=None, seed=None, clabs=None):
        if (not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64
                or any(c not in "0123456789abcdef" for c in calibration_sha256)):
            raise ValueError("Provide the SHA-256 of the H3 Image gate calibration record")
        if type(parameter) not in (int, float):
            raise ValueError("Provide an explicit numeric H3 Image gate threshold, bias, or rate")
        self._core = gate_core(core_wheel)
        self._clabs = self._core.CLABSParameters(**clabs) if clabs is not None else None
        self._method, self._parameter, self._seed = method, float(parameter), seed
        self._calibration_sha256 = calibration_sha256
        self._direction = None
        if method == "random":
            if direction is not None:
                raise ValueError("The matched random control does not use an H3 Image direction")
        else:
            if direction is None:
                raise ValueError("CFP and CLABS require a fitted H3 Image direction")
            import torch

            vector = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
            if (vector.shape != (PACKED_CHANNELS,) or not torch.isfinite(vector).all()
                    or not torch.isclose(vector.norm(), vector.new_tensor(1.0))):
                raise ValueError("H3 Image direction must be a finite 96-channel unit vector")
            self._direction = vector
        self._new_gate()

    def _new_gate(self):
        return self._core.ActivationGate(self._method, self._parameter, seed=self._seed, clabs=self._clabs)

    def _score(self, readout, strength):
        if self._direction is None:
            return None
        return float(self._core.activation_projection(readout, self._direction, strength=strength))

    def _receipt(self):
        return {
            "core_revision": CORE_REVISION,
            "core_source_sha256": CORE_SHA256,
            "readout": "fp32_mean_packed_image_rows_before_adapter",
            "calibration_sha256": self._calibration_sha256,
            "method": self._method,
            "parameter": self._parameter,
            "seed": self._seed,
            "clabs": asdict(self._clabs) if self._clabs is not None else None,
            "direction": self._direction.tolist() if self._direction is not None else None,
        }


def denoise_with_gains(denoise, transformer, conditioning, noise, *, scale_adapter,
                       step_gains=None, gate_plan=None, strength=1.0, check):
    """Run the original PDD denoiser with fixed gains or measured gate settings."""
    step_count = len(PDD_SIGMAS) - 1
    if type(strength) not in (float, int) or not 0 <= strength <= 1:
        raise ValueError("nominal H3 Image strength must be finite in [0,1]")
    if gate_plan is None:
        if not isinstance(step_gains, (list, tuple)) or len(step_gains) != step_count:
            raise ValueError("H3 Image requires one gain per native PDD head")
        if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
            raise ValueError("H3 Image step gains must be finite and nonnegative")
        strengths = [float(strength * gain) for gain in step_gains]
        if any(not 0 <= value <= 1 for value in strengths):
            raise ValueError("effective H3 Image strengths must be finite in [0,1]")
    else:
        if step_gains is not None or not isinstance(gate_plan, ImageGatePlan):
            raise ValueError("Choose fixed H3 Image step gains or one calibrated ImageGatePlan")
        if strength == 0:
            raise ValueError("Gated H3 Image strength must be positive; use fixed strength 0 for OFF")
        sigmas = list(PDD_SIGMAS)
        gate = gate_plan._new_gate()

    head_indices, native_timesteps = [], []
    applied, decisions = [], []
    scope = ExitStack()

    def before(_model, _args, request):
        index = request.get("pdd_step_index")
        if type(index) is not int or index != len(head_indices) or index >= step_count:
            raise RuntimeError("H3 Image native PDD head order differs from the requested schedule")
        time = request["timestep"]
        if time.numel() != 1:
            raise RuntimeError("H3 Image requires its native single-image PDD timestep")
        timestep = float(time.item())
        if gate_plan is None:
            value = strengths[index]
        else:
            if timestep != 1.0 - sigmas[index]:
                raise RuntimeError("H3 Image native timestep differs from its fixed PDD6 grid")
            readout = packed_image_readout(request.get("hidden_states"))
            score = gate_plan._score(readout, strength)
            decision = gate.step(score)
            value = float(strength) if decision.fired else 0.0
            decisions.append({
                "interval": index, "projection": score, "fired": decision.fired,
                "probability": decision.probability, "effective_strength": value,
            })
        scope.enter_context(scale_adapter(value))
        head_indices.append(index)
        native_timesteps.append(timestep)
        applied.append(value)

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as request:
        request.callback(transformer.register_forward_pre_hook(before, with_kwargs=True).remove)
        request.callback(transformer.register_forward_hook(after, always_call=True).remove)
        request.callback(scope.close)
        result = denoise(transformer, conditioning, noise, check=check)
    if len(head_indices) != step_count:
        raise RuntimeError("H3 Image finished before applying every scheduled PDD strength")
    receipt = {
        "pdd_step_indices": head_indices, "native_timesteps": native_timesteps,
        "effective_strengths": applied,
    }
    if gate_plan is not None:
        dose = fsum(value * (start - end) for value, (start, end)
                    in zip(applied, pairwise(sigmas), strict=True))
        receipt.update(
            activation_gate=gate_plan._receipt(), decisions=decisions, sigmas=sigmas,
            nominal_strength=float(strength), sigma_integrated_dose=dose,
            fired=gate.usage.fired, eligible=gate.usage.eligible, gate_rate=gate.usage.rate,
        )
    return result, receipt
