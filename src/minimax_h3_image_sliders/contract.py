"""Pure-Python request metadata, not a native model or an authorization gate."""
from dataclasses import asdict, dataclass
from importlib.resources import files
import json
import math

MODEL_ID = "Comfy-Org/MiniMax-H3-Image"
REFUSAL = (
    "MiniMax-H3-Image native operations are not implemented or qualified. "
    "Model license authorization must also be established separately. "
    "This source-only package does not download, load, train, render or publish weights."
)
OPERATIONS = ("train", "resume", "evaluate", "export", "response", "render")

class NativeUnavailable(RuntimeError):
    """No native implementation is present in this source cohort."""

def refuse_native(operation):
    raise NativeUnavailable(f"{operation}: {REFUSAL}")

def source_lock():
    """Return a fresh copy of the packaged, pinned inspection record."""
    return json.loads(files(__package__).joinpath("source-lock.json").read_text())

@dataclass(frozen=True)
class RequestPlan:
    prompt: str
    width: int = 4096
    height: int = 4096
    references: tuple[str, ...] = ()
    seed: int = 7
    strength: float = 1.0

    def __post_init__(self):
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be nonempty text")
        for key in ("width", "height"):
            value = getattr(self, key)
            if type(value) is not int or not 32 <= value <= 8192 or value % 32:
                raise ValueError(f"{key} must be a multiple of 32 in [32, 8192]")
        # This is a planner budget, not a native quality or memory-fit claim.
        if self.width * self.height > 17_000_000:
            raise ValueError("canvas exceeds this planner's 17-million-pixel budget")
        if not isinstance(self.references, (tuple, list)):
            raise ValueError("references must be an ordered list or tuple of path labels")
        references = tuple(self.references)
        if len(references) > 16 or any(not isinstance(p, str) or not p.strip() for p in references):
            raise ValueError("provide at most sixteen nonempty reference path labels")
        object.__setattr__(self, "references", references)
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(self.strength) not in (float, int) or not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("planned strength must be finite and in [0, 1]")

    def as_dict(self):
        lock = source_lock()
        return {
            "status": "plan-only; not executable or model-qualified",
            "model": lock["model"],
            "shared_core": lock["shared_core"],
            "request": asdict(self),
            "task": "image-edit" if self.references else "text-to-image",
            "native_boundary": {
                "batch_size": 1,
                "latent_shape": [1, 24, 1, self.height // 16, self.width // 16],
                "target_image_tokens": (self.height // 32) * (self.width // 32),
                "reference_order": "image_1 is the edit target; preserve all reference order",
                "max_documented_references": 16,
                "audio_stream": False,
                "text_encoder": "MiniMax-H3 Qwen3-VL; 50th-layer hidden states",
                "sampling": {**lock["sampling"], "cfg": False, "negative_prompt": False,
                             "head_bank": "checkpoint-specific PDD heads; keep with its sigma grid"},
                "weights": "raw-only; no EMA or checkpoint averaging",
            },
            "caveats": [
                "Official card recommends approximately 16 MP; pinned ComfyUI node defaults to 2048 square.",
                "Planned dimensions and strengths are not trained-slider or renderer qualification.",
                "Reference labels are not opened. Expanded prompts must be frozen for paired teacher collection.",
                REFUSAL,
            ],
        }
