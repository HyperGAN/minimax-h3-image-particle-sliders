"""Software-only check of the installed shared projection on unrelated CPU data.

No H3 model, image sampler, GAN objective, optimizer or checkpoint is constructed.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from particle_sliders.routed_projection import TinyParticleProjection


class ProjectionFixture(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(19).get_state())
            self.host = nn.Linear(8, 8, dtype=torch.float32).requires_grad_(False)
            self.bank = nn.Parameter(torch.randn(128, 4, dtype=torch.float32))
            self.original = self.host.forward
            self.projection = TinyParticleProjection("fixture", self.host, self.bank, 4, 4.0).float()
        self.closed = False

    @contextmanager
    def at_strength(self, strength):
        if self.closed:
            raise RuntimeError("fixture is closed")
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be finite and in [0, 1]")
        old = self.projection.multiplier
        self.projection.multiplier = float(strength)
        try:
            yield self
        finally:
            self.projection.multiplier = old

    def forward(self, x):
        if self.closed:
            raise RuntimeError("fixture is closed")
        return self.host(x)

    def close(self):
        if not self.closed:
            # host had no instance-level forward before shared attachment.
            del self.host.forward
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def check():
    with ProjectionFixture() as fixture:
        x = torch.linspace(-1, 1, 48, device="cpu", dtype=torch.float32).reshape(2, 3, 8)
        base = fixture.original(x)
        with fixture.at_strength(1):
            initialized = fixture(x)
        if not torch.equal(base, initialized):
            raise RuntimeError("zero-initialized projection changed the base")
        # An explicit synthetic probe opens the up gate; this is not training.
        with torch.no_grad():
            fixture.projection.lora_up.weight.fill_(0.03)
        with fixture.at_strength(1):
            full = fixture(x)
            full.square().mean().backward()
        with fixture.at_strength(0.5):
            half = fixture(x)
        with fixture.at_strength(0):
            off = fixture(x)
        gradient = fixture.bank.grad
        if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum().item() > 0:
            raise RuntimeError("actual shared bank did not receive finite nonzero gradients")
        if not torch.equal(base, off) or not torch.allclose(half - base, (full.detach() - base) / 2, atol=1e-7):
            raise RuntimeError("slider scaling failed")
        return dict(status="passed", fixture="unrelated random 8x8 Linear", device="cpu",
                    shared_projection=type(fixture.projection).__module__ + "." + type(fixture.projection).__name__,
                    bank_shape=list(fixture.bank.shape), base_identity=True, fractional_strength=True,
                    finite_nonzero_bank_gradients=True, optimizer_steps=0,
                    native_model_loaded=False, native_qualified=False, weights_saved=False)
