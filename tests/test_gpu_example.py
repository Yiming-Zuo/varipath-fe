from pathlib import Path

import openmm as mm
import pytest
import torch

from varipath.endpoint_md import MetropolizedEndpointMD, require_platform
from varipath.physics import MolecularSystem
from varipath.workflow import doctor, load_config, run_md


def test_two_examples_select_cpu_and_cuda_paths():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root/"configs").glob("*.json"))
    assert len(files) == 2
    cpu = load_config(root/"configs/aldp-example.json")
    gpu = load_config(root/"configs/ala4-example.json")
    assert cpu.get("runtime", {}).get("device", "cpu") == "cpu"
    assert gpu["endpoint_md"]["platform"] == "CUDA"
    assert gpu["molecule"] == "ALA4"


def test_unavailable_cuda_fails_before_loading_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(mm.Platform, "getNumPlatforms", lambda: 0)
    config = dict(molecule="ALA4", endpoint_md={"platform":"CUDA"})
    with pytest.raises(RuntimeError, match="CUDA 平台不可用"):
        doctor(config)
    with pytest.raises(RuntimeError, match="CUDA 平台不可用"):
        run_md(config, tmp_path/"absent", 0, tmp_path/"output")
    assert not (tmp_path/"output").exists()


@pytest.mark.skipif("CUDA" not in [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())], reason="OpenMM CUDA platform unavailable")
def test_native_cuda_proposal_and_checkpoint_replay():
    system = MolecularSystem("ALA4")
    kernel = MetropolizedEndpointMD(system, 1, 17, platform="CUDA", steps_per_segment=1)
    try:
        assert kernel.auxiliary.getPlatform().getPropertyValue(kernel.auxiliary, "Precision") == "double"
        kernel.step()
        saved = kernel.state_dict()
        first = kernel.step()
        positions = kernel.positions.copy()
        kernel.load_state_dict(saved)
        second = kernel.step()
        assert first == second
        torch.testing.assert_close(torch.as_tensor(kernel.positions), torch.as_tensor(positions), rtol=0, atol=0)
    finally:
        kernel.close()
