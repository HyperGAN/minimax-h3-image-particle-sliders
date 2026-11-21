"""Opt-in gain control for a caller-owned native FP4 H3 Image denoiser."""

from contextlib import ExitStack
from sys import float_info


def denoise_with_gains(denoise, transformer, conditioning, noise, *, scale_adapter,
                       step_gains, strength=1.0, check):
    """Run the original PDD denoiser with a separate strength for each head."""
    if type(strength) not in (float, int) or not 0 <= strength <= 1:
        raise ValueError("nominal H3 Image strength must be finite in [0,1]")
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != 6:
        raise ValueError("H3 Image requires one gain per native PDD head")
    if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("H3 Image step gains must be finite and nonnegative")
    strengths = [float(strength * gain) for gain in step_gains]
    if any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("effective H3 Image strengths must be finite in [0,1]")

    head_indices, native_timesteps = [], []
    scope = ExitStack()

    def before(_model, _args, request):
        index = request.get("pdd_step_index")
        if type(index) is not int or index != len(head_indices) or index >= len(strengths):
            raise RuntimeError("H3 Image native PDD head order differs from the requested schedule")
        time = request["timestep"]
        if time.numel() != 1:
            raise RuntimeError("H3 Image requires its native single-image PDD timestep")
        scope.enter_context(scale_adapter(strengths[index]))
        head_indices.append(index)
        native_timesteps.append(float(time.item()))

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as request:
        request.callback(transformer.register_forward_pre_hook(before, with_kwargs=True).remove)
        request.callback(transformer.register_forward_hook(after, always_call=True).remove)
        request.callback(scope.close)
        result = denoise(transformer, conditioning, noise, check=check)
    if len(head_indices) != len(strengths):
        raise RuntimeError("H3 Image finished before applying every scheduled PDD strength")
    return result, {
        "pdd_step_indices": head_indices, "native_timesteps": native_timesteps,
        "effective_strengths": strengths,
    }
