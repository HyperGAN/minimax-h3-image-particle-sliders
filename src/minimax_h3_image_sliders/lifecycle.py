"""Shared lifecycle entry point: explicit unsupported capability before allocation."""
from particle_sliders.lifecycle import Backend, Capabilities
from .contract import OPERATIONS, REFUSAL, refuse_native

def validate(options):
    refuse_native("validate")

def prepare(options, **staged):
    refuse_native("prepare")

BACKEND = Backend(
    name="minimax-h3-image",
    capabilities=Capabilities(
        generator_arm="routed_particle_gan", render=False,
        operations=(), base_precisions=(), render_outputs=(),
        qualification="Source planning and unrelated synthetic CPU fixture only; no native qualification",
        unavailable_reasons=tuple((op, REFUSAL) for op in OPERATIONS),
    ),
    prepare=prepare, validate=validate,
)
