import json
import subprocess
import sys
import pytest
from minimax_h3_image_sliders.contract import OPERATIONS


def run(*args):
    return subprocess.run([sys.executable,"-m","minimax_h3_image_sliders.cli",*args],capture_output=True,text=True)


def test_plan_cli():
    p=run("plan","--prompt","cup","--reference","absent.png","--strength","0.5")
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["request"]["references"] == ["absent.png"]


def test_provenance_cli():
    p=run("provenance")
    assert p.returncode == 0
    assert json.loads(p.stdout)["schema"] == "minimax-h3-image-source-plan-v1"


@pytest.mark.parametrize("operation",OPERATIONS)
def test_native_cli_never_executes(operation):
    p=run(operation)
    assert p.returncode == 2
    assert "not implemented or qualified" in p.stderr
    assert not p.stdout


def test_invalid_cli():
    p=run("plan","--prompt","cup","--strength","nan")
    assert p.returncode == 2
    assert "finite" in p.stderr
