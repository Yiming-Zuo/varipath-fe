import copy

import numpy as np
import pytest
import torch

from varipath.coordinates import BodyCoordinates
from varipath.endpoint_md import MetropolizedEndpointMD, run_endpoint_md
from varipath.physics import MolecularSystem


@pytest.fixture(scope="module")
def system():
    torch.set_num_threads(1)
    return MolecularSystem("ALA4")


@pytest.mark.parametrize("endpoint", [0, 1])
def test_native_52_atom_md_resume_replays_both_contexts_and_rng(system, endpoint, tmp_path):
    kernel = MetropolizedEndpointMD(system, endpoint, 1024, steps_per_segment=2, platform="Reference")
    restored = None
    try:
        validation = kernel.reversibility_check()
        assert validation["passed"]
        assert validation["position_error_nm"] < 1e-8
        kernel.step()
        path = tmp_path/f"endpoint-{endpoint}.pt"
        kernel.save(path)
        expected = kernel.step()
        expected_positions = kernel.positions.copy()
        expected_auxiliary = kernel.auxiliary_positions.copy()
        restored = MetropolizedEndpointMD.load(system, path)
        replay = restored.step()
        assert replay == expected
        assert np.array_equal(restored.positions, expected_positions)
        assert np.array_equal(restored.auxiliary_positions, expected_auxiliary)
        assert kernel._constraint_error(kernel.positions) < 1e-10
        assert np.linalg.norm((kernel.positions*kernel.bridge.masses[:, None]).sum(0)) < 1e-12
    finally:
        kernel.close()
        if restored is not None:
            restored.close()


def test_strict_constraint_and_all_l_rejection_restore_both_native_contexts(system):
    kernel = MetropolizedEndpointMD(system, 1, 1024, steps_per_segment=1, platform="CPU")
    try:
        original = kernel.state_dict()
        malformed = kernel.positions.copy()
        malformed[kernel.bridge.leaves[0], 0] += 1e-3
        with pytest.raises(ValueError, match="不能静默投影"):
            kernel.set_positions(malformed)
        kernel.set_positions(malformed, project_initialization=True)
        assert kernel.initialization["constraint_projection"]
        assert not kernel.initialization["projected_initialization_is_equilibrium_data"]
        assert kernel._constraint_error(kernel.positions) < 1e-10
        original = kernel.state_dict()
        mirror = kernel.positions.copy()
        mirror[:, 0] *= -1
        with pytest.raises(ValueError, match="全 L"):
            kernel.set_positions(mirror)
        kernel._in_domain = lambda positions: False
        result = kernel.step()
        assert result["domain_rejection"] and not result["accepted"]
        restored = kernel.state_dict()
        assert np.array_equal(restored["positions"], original["positions"])
        assert np.array_equal(restored["auxiliary_positions"], original["auxiliary_positions"])
        assert restored["native_checkpoint"] == original["native_checkpoint"]
        assert restored["auxiliary_checkpoint"] == original["auxiliary_checkpoint"]
        assert restored["metric"] == original["metric"]
    finally:
        kernel.close()


def test_bridge_mass_correction_matches_independent_full_rotational_gram(system):
    kernel = MetropolizedEndpointMD(system, 1, 1024, steps_per_segment=1, platform="Reference")
    try:
        metadata = copy.deepcopy(system.metadata)
        for atom in metadata["atoms"]:
            atom["mass_dalton"] = kernel.bridge.auxiliary_mass
        lengths = {leaf: length for _, _, leaf, _, length in kernel.bridge.rows}
        for constraint in metadata["constraints"]:
            constraint["length_nm"] = lengths[constraint["hydrogen_index"]]
        auxiliary_chart = BodyCoordinates(metadata)
        constants = []
        for positions in system.initial_positions[:2].numpy():
            auxiliary = kernel.bridge.to_auxiliary(positions)
            assert np.max(np.abs(kernel.bridge.to_native(auxiliary)-positions)) < 1e-15
            native_z = system.chart.encode(torch.tensor(positions))
            auxiliary_z = auxiliary_chart.encode(torch.tensor(auxiliary))
            native_volume = system.chart.dense_log_volume(native_z)
            auxiliary_volume = auxiliary_chart.dense_log_volume(auxiliary_z)
            constants.append(float(native_volume-auxiliary_volume)-float(kernel.bridge.log_metric_ratio(positions)))
        assert np.ptp(constants) < 1e-9
    finally:
        kernel.close()


def config():
    return {"molecule": "ALA4", "mode": "development", "seed": 1024,
            "budget_seconds": 120., "endpoint_md": {"platform": "Reference",
            "steps_per_segment": 1, "timestep_fs": .5, "ladders": 4,
            "warmup": 0, "endpoint_counts": [1, 1], "allocation": [1, 1],
            "checkpoint_every": 1}}


def test_complete_native_two_endpoint_run_and_interrupted_resume(system, tmp_path):
    expected_dir, resumed_dir = tmp_path/"expected", tmp_path/"resumed"
    expected = run_endpoint_md(config(), expected_dir, preparation={"system": system})
    calls = 0
    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_endpoint_md(config(), resumed_dir, preparation={"system": system}, check_budget=interrupt)
    result = run_endpoint_md(config(), resumed_dir, resume=True, preparation={"system": system})
    assert result["primary_complete"] and not result["eligible_for_formal"]
    assert result["delta_free_energy"] is None
    assert result["native"]["preference"] == expected["native"]["preference"]
    assert result["native"]["native_zero_hit"]
    assert result["native"]["native_preference"] is None
    assert np.isfinite(result["native"]["preference"])
    assert result["fixed_endpoint_counts_per_ladder"] == [1, 1]
    for endpoint in (0, 1):
        for ladder in range(4):
            filename = f"endpoint-{endpoint}-ladder-{ladder}.pt"
            a = torch.load(expected_dir/filename, weights_only=False)
            b = torch.load(resumed_dir/filename, weights_only=False)
            assert np.array_equal(np.stack(a["positions"]), np.stack(b["positions"]))


def test_timeout_discards_whole_primary_and_uses_only_paid_same_budget_fallback(tmp_path):
    fallback = {"preference": .2, "reporting_floor": 1e-4, "native_defined": True}
    def expired():
        raise TimeoutError("test budget exhausted")
    result = run_endpoint_md(config(), tmp_path/"paid", preparation={"fallback": fallback,
                 "fallback_qualified": True, "fallback_within_same_budget": True,
                 "preparation_seconds": .1}, check_budget=expired)
    assert result["status"] == "paid_identity_fallback"
    assert result["output"] == fallback and not result["primary_complete"]
    invalid = run_endpoint_md(config(), tmp_path/"unpaid", preparation={"fallback": fallback}, check_budget=expired)
    assert invalid["status"] == "no_legal_output" and invalid["output"] is None
