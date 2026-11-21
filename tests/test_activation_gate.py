"""Exercise the native packed-image call contract with synthetic velocities."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from os import environ
from pathlib import Path
from runpy import run_path
from threading import Barrier
from zipfile import ZipFile

import pytest
import torch

from minimax_h3_image_sliders.scheduled_native import (
    CORE_MEMBER,
    CORE_REVISION,
    ImageGatePlan,
    PDD_SIGMAS,
    denoise_with_gains,
    gate_core,
    packed_image_readout,
)
from minimax_h3_image_sliders.contract import source_lock


SIGMAS = source_lock()["sampling"]["sigmas"]
DIRECTION = [1.0] + [0.0] * 95
STRENGTH = 0.5
CALIBRATION = "a" * 64


@pytest.fixture(scope="module")
def core_wheel():
    supplied = environ.get("H3_IMAGE_STEERING_CORE_WHEEL")
    if not supplied:
        pytest.skip("set H3_IMAGE_STEERING_CORE_WHEEL to the reviewed core #269 wheel")
    return Path(supplied)


@pytest.fixture
def native():
    class Transformer(torch.nn.Module):
        strength = 0.125
        fault = None

        def forward(self, *, hidden_states, timestep, pdd_step_index, **_layout):
            if self.fault == "forward":
                raise RuntimeError("fixture forward failed")
            return (torch.full_like(hidden_states, 1.0 + self.strength),)

        @contextmanager
        def scaled(self, value):
            previous, self.strength = self.strength, value
            try:
                yield
            finally:
                self.strength = previous

    def denoise(transformer, conditioning, noise, *, check):
        result = noise
        for index, (start, end) in enumerate(zip(SIGMAS, SIGMAS[1:])):
            check(index)
            head = index
            time = 1.0 - start
            if index == 1:
                if transformer.fault == "repeat":
                    head = 0
                if transformer.fault == "clock":
                    time = 0.0
                if transformer.fault == "unfinished":
                    break
            velocity, = transformer(
                hidden_states=result, encoder_hidden_states=conditioning,
                timestep=torch.tensor([time]), pdd_step_index=head, return_dict=False,
            )
            result = result + (start - end) * velocity
        return result

    transformer = Transformer()
    return transformer, (denoise, transformer, torch.zeros((1, 1, 5120)), torch.zeros((1, 4, 96)))


def plan(core_wheel, method, parameter, **kwargs):
    return ImageGatePlan(core_wheel=core_wheel, method=method, parameter=parameter,
                         calibration_sha256=CALIBRATION, **kwargs)


def run(native, **kwargs):
    transformer, args = native
    return denoise_with_gains(*args, check=int, scale_adapter=transformer.scaled,
                              strength=STRENGTH, **kwargs)


def test_cfp_observes_live_native_latents_and_replays_exactly(core_wheel, native):
    gate_plan = plan(core_wheel, "cfp", 0.2, direction=DIRECTION)
    mixed, trace = run(native, gate_plan=gate_plan)
    replay, fixed = run(native, step_gains=[value / STRENGTH for value in trace["effective_strengths"]])
    assert torch.equal(mixed, replay)
    assert [item["interval"] for item in trace["decisions"]] == list(range(6))
    assert trace["native_timesteps"] == [1.0 - value for value in SIGMAS[:-1]]
    assert trace["sigmas"] == SIGMAS
    assert trace["effective_strengths"] == fixed["effective_strengths"] == [0.5, 0.5, 0.5, 0, 0, 0]
    # Intervening at head zero changes the live score at head one.
    assert trace["decisions"][1]["projection"] == pytest.approx((SIGMAS[0] - SIGMAS[1]) * 1.5)
    assert trace["fired"] == 3 and trace["eligible"] == 6 and trace["gate_rate"] == 0.5
    assert trace["sigma_integrated_dose"] == pytest.approx(STRENGTH * (SIGMAS[0] - SIGMAS[3]))
    assert trace["activation_gate"]["core_revision"] == CORE_REVISION
    assert trace["activation_gate"]["calibration_sha256"] == CALIBRATION
    assert trace["activation_gate"]["direction"] == DIRECTION


@pytest.mark.parametrize("rate", [0, 1])
def test_random_endpoints_match_original_constant_or_off(core_wheel, native, rate):
    transformer, (denoise, model, conditioning, noise) = native
    with transformer.scaled(STRENGTH * rate):
        expected = denoise(model, conditioning, noise, check=int)
    actual, trace = run(native, gate_plan=plan(core_wheel, "random", rate, seed=100))
    assert torch.equal(actual, expected)
    assert trace["fired"] == rate * 6
    assert all(item["projection"] is None and item["probability"] == rate for item in trace["decisions"])
    assert trace["sigma_integrated_dose"] == STRENGTH * rate
    assert transformer.strength == 0.125


def test_clabs_seed_is_request_local(core_wheel, native):
    clabs = dict(mean=0.2, std=1, epsilon=1e-6, setpoint=0.5,
                 proportional_gain=0.1, integral_gain=0.25, beta=0.3)
    gate_plan = plan(core_wheel, "clabs", 0.5, direction=DIRECTION, seed=37, clabs=clabs)
    first, trace = run(native, gate_plan=gate_plan)
    second, second_trace = run(native, gate_plan=gate_plan)
    assert torch.equal(first, second)
    assert trace == second_trace
    gate = gate_core(core_wheel).ActivationGate(
        "clabs", 0.5, seed=37, clabs=gate_core(core_wheel).CLABSParameters(**clabs),
    )
    for item in trace["decisions"]:
        expected = gate.step(item["projection"])
        assert item["fired"] == expected.fired
        assert item["probability"] == expected.probability


@pytest.mark.parametrize(("fault", "message"), [
    ("repeat", "PDD head order"),
    ("clock", "fixed PDD6 grid"),
    ("unfinished", "before applying every"),
    ("forward", "fixture forward failed"),
])
def test_native_fault_removes_hooks_and_restores_prior_strength(core_wheel, native, fault, message):
    transformer, _ = native
    transformer.fault = fault
    with pytest.raises(RuntimeError, match=message):
        run(native, gate_plan=plan(core_wheel, "random", 1, seed=1))
    assert transformer.strength == 0.125
    assert not transformer._forward_pre_hooks and not transformer._forward_hooks


def test_requires_explicit_compatible_calibration_and_exact_core(core_wheel, native, tmp_path):
    gate_plan = plan(core_wheel, "cfp", 0.2, direction=DIRECTION)
    with pytest.raises(ValueError, match="Choose fixed"):
        run(native, gate_plan=gate_plan, step_gains=[1] * 6)
    with pytest.raises(ValueError, match="unit vector"):
        plan(core_wheel, "cfp", 0.2, direction=[1.0, 0.0])
    with pytest.raises(ValueError, match="calibration record"):
        ImageGatePlan(core_wheel=core_wheel, method="cfp", parameter=0.2, calibration_sha256="")
    with pytest.raises(ValueError, match="stochastic gates require"):
        plan(core_wheel, "random", 0.5)
    tampered = tmp_path / "tampered.whl"
    with ZipFile(tampered, "w") as archive:
        archive.writestr(CORE_MEMBER, b"raise AssertionError('unverified code executed')")
    with pytest.raises(ValueError, match="exact adaptive steering source"):
        gate_core(tampered)


def test_readout_is_fp32_mean_over_native_image_rows():
    packed = torch.arange(192, dtype=torch.bfloat16).reshape(1, 2, 96)
    assert torch.equal(packed_image_readout(packed), torch.arange(48, 144, dtype=torch.float32))
    with pytest.raises(ValueError, match="96 channels"):
        packed_image_readout(torch.zeros((1, 4, 24)))


def test_staged_owner_loads_helper_by_filename(core_wheel, native):
    transformer, args = native
    standalone = run_path(denoise_with_gains.__code__.co_filename)
    gate_plan = standalone["ImageGatePlan"](
        core_wheel=core_wheel, method="cfp", parameter=0.2,
        calibration_sha256=CALIBRATION, direction=DIRECTION,
    )
    actual, trace = standalone["denoise_with_gains"](
        *args, check=int, scale_adapter=transformer.scaled, strength=STRENGTH, gate_plan=gate_plan,
    )
    package, expected = run(native, gate_plan=plan(core_wheel, "cfp", 0.2, direction=DIRECTION))
    assert torch.equal(actual, package)
    assert trace == expected
    assert list(PDD_SIGMAS) == SIGMAS


def test_independent_filename_loads_have_private_core_modules(core_wheel):
    first_helper = run_path(denoise_with_gains.__code__.co_filename)
    second_helper = run_path(denoise_with_gains.__code__.co_filename)
    ready = Barrier(2, timeout=10)

    def load(helper):
        ready.wait()
        return helper["gate_core"](core_wheel)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(load, first_helper)
        second = executor.submit(load, second_helper)
        independent = first.result(), second.result()
    assert independent[0].__name__ != independent[1].__name__
    assert all(module.ActivationGate("random", 1.0, seed=1).step().fired for module in independent)
