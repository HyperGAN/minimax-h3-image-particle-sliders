from contextlib import contextmanager

import pytest
import torch

from minimax_h3_image_sliders.scheduled_native import denoise_with_gains


def test_each_native_pdd_head_uses_its_gain_and_restores_adapter_state():
    class Transformer(torch.nn.Module):
        strength = 0.0

        def forward(self, state, *, timestep, pdd_step_index):
            return state + self.strength * (pdd_step_index + 1)

        @contextmanager
        def scaled(self, value):
            previous, self.strength = self.strength, value
            try:
                yield
            finally:
                self.strength = previous

    def denoise(transformer, conditioning, noise, *, check):
        result = conditioning + noise
        for index in range(6):
            check(index)
            result = transformer(result, timestep=torch.tensor(index / 8), pdd_step_index=index)
        return result

    model = Transformer()
    conditioning, noise = torch.tensor(0.0), torch.tensor(1.0)
    with model.scaled(0.5):
        expected = denoise(model, conditioning, noise, check=int)
    inputs = dict(strength=0.5, check=int, scale_adapter=model.scaled)
    constant, trace = denoise_with_gains(
        denoise, model, conditioning, noise, step_gains=[1] * 6, **inputs,
    )
    assert torch.equal(constant, expected)
    assert trace["native_timesteps"] == [index / 8 for index in range(6)]
    assert trace["pdd_step_indices"] == list(range(6))
    gain_vector = [1.5, 1.5, 1.5, 0.5, 0.5, 0.5]
    varied, trace = denoise_with_gains(
        denoise, model, conditioning, noise, step_gains=gain_vector, **inputs,
    )
    assert not torch.equal(varied, expected)
    assert trace["effective_strengths"] == [gain * 0.5 for gain in gain_vector]
    assert model.strength == 0.0

    def out_of_order(transformer, conditioning, noise, *, check):
        transformer(noise, timestep=torch.tensor(0.25), pdd_step_index=2)

    with pytest.raises(RuntimeError, match="PDD head order"):
        denoise_with_gains(out_of_order, model, conditioning, noise, step_gains=[1] * 6, **inputs)
