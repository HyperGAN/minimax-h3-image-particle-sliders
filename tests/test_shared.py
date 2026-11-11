from dataclasses import replace
from importlib.metadata import entry_points
import pytest
import torch
from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation
from minimax_h3_image_sliders.contract import OPERATIONS, NativeUnavailable
from minimax_h3_image_sliders.lifecycle import BACKEND, validate, prepare
from minimax_h3_image_sliders.synthetic import ProjectionFixture, check


def test_installed_registration():
    ep = next(e for e in entry_points(group="particle_sliders.image_backends") if e.name == "minimax-h3-image")
    assert ep.load() is BACKEND


@pytest.mark.parametrize("operation", OPERATIONS)
def test_native_refuses_before_validation_pg_or_preparation(monkeypatch, operation):
    def bad(*args, **kwargs):
        pytest.fail("unsupported operation progressed beyond capability gate")
    monkeypatch.setattr(type(BACKEND), "_check_pg", bad)
    guarded = replace(BACKEND, validate=bad, prepare=bad)
    with pytest.raises(UnsupportedOperation, match="not implemented or qualified"):
        guarded.open(SessionConfig("minimax-h3-image","source-only",ema_decay=None),operation=operation)


@pytest.mark.parametrize("fn",[validate,prepare])
def test_direct_native_hooks_stay_unavailable(fn):
    with pytest.raises(NativeUnavailable):
        fn({"allow_hub":True,"authorized":True,"device":"cuda"})


def test_shared_fixture_probe_preserves_rng():
    state=torch.get_rng_state().clone()
    result=check()
    assert torch.equal(state,torch.get_rng_state())
    assert result["status"] == "passed"
    assert result["bank_shape"] == [128,4]
    assert result["optimizer_steps"] == 0
    assert result["native_qualified"] is False


def test_bank_owned_once_frozen_host_and_cleanup():
    with ProjectionFixture() as f:
        assert sum(p is f.bank for p in f.parameters()) == 1
        assert not any(p.requires_grad for p in f.host.parameters())
        assert "forward" in f.host.__dict__
    assert "forward" not in f.host.__dict__
    f.close()
    with pytest.raises(RuntimeError):
        f(torch.zeros(1,8))


def test_nested_strength_failure_restores():
    with ProjectionFixture() as f:
        with f.at_strength(.4):
            with pytest.raises(RuntimeError,match="probe"):
                with f.at_strength(.9):
                    raise RuntimeError("probe")
            assert f.projection.multiplier == .4
        assert f.projection.multiplier == 0


@pytest.mark.parametrize("strength",[True,-1,2,float("nan"),float("inf")])
def test_synthetic_strength_validation(strength):
    with ProjectionFixture() as f, pytest.raises(ValueError):
        with f.at_strength(strength):
            pass
