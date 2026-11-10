import json
import subprocess
import sys
import pytest
from minimax_h3_image_sliders.contract import RequestPlan, source_lock


def test_default_image_geometry_and_schedule():
    p = RequestPlan("A ceramic cup by a window").as_dict()
    assert p["model"]["id"] == "Comfy-Org/MiniMax-H3-Image"
    assert p["native_boundary"]["latent_shape"] == [1, 24, 1, 256, 256]
    assert p["native_boundary"]["target_image_tokens"] == 16384
    assert p["native_boundary"]["audio_stream"] is False
    assert p["native_boundary"]["sampling"]["num_inference_steps"] == 6
    assert len(p["native_boundary"]["sampling"]["sigmas"]) == 7
    assert p["native_boundary"]["sampling"]["sigmas"] == [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
    assert not p["native_boundary"]["sampling"]["cfg"]
    assert p["task"] == "text-to-image"
    json.dumps(p, allow_nan=False)


@pytest.mark.parametrize("change", [
    {"prompt":" "}, {"prompt":None}, {"width":True}, {"width":4097}, {"height":0},
    {"height":8193}, {"width":8192,"height":8192}, {"seed":-1}, {"seed":2**63},
    {"seed":True}, {"strength":float("nan")}, {"strength":float("inf")},
    {"strength":True}, {"strength":-0.1}, {"strength":1.1}, {"references":"x.png"},
    {"references":[""]}, {"references":["x"]*17}, {"references":{"x":"y"}},
])
def test_invalid_plan(change):
    with pytest.raises(ValueError):
        RequestPlan(**{"prompt":"cup", **change})


def test_edit_labels_not_read_and_order_preserved():
    refs = ["absent-target.png", "absent-ref.png"]
    plan = RequestPlan("Edit <Picture 1>", width=6272,height=2688,references=refs)
    refs.reverse()
    assert plan.references == ("absent-target.png", "absent-ref.png")
    p = plan.as_dict()
    assert p["task"] == "image-edit"
    assert p["native_boundary"]["latent_shape"] == [1,24,1,168,392]


def test_comfy_canvas_plannable_not_claimed_native():
    p = RequestPlan("cup",width=2048,height=2048).as_dict()
    assert p["native_boundary"]["target_image_tokens"] == 4096
    assert p["status"].startswith("plan-only")


def test_lock_is_independent_and_immutable_to_caller():
    lock=source_lock(); lock["sampling"]["sigmas"][0]=0
    assert source_lock()["sampling"]["sigmas"][0] == 1
    assert len(source_lock()["shared_core"]["revision"]) == 40
    assert source_lock()["shared_core"]["particlegan"] == "0.9.5"
    assert not any(source_lock()["runtime_policy"].values())


def test_plan_has_no_native_or_torch_import():
    code = "from minimax_h3_image_sliders.contract import RequestPlan; RequestPlan('cup').as_dict(); import sys; assert not any(x in sys.modules for x in ['torch','comfy','diffusers','transformers','huggingface_hub'])"
    subprocess.run([sys.executable,"-c",code],check=True)
