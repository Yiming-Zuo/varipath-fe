import numpy as np
import pytest
import torch

from varipath.diagnostics import chain_diagnostics, qualify_bank


def test_rank_split_folded_rhat_and_bulk_tail_ess_on_independent_gaussians():
    chains = np.random.default_rng(1024).normal(size=(4, 2048))
    diagnostic = chain_diagnostics(chains)
    assert diagnostic["status"] == "passed"
    assert diagnostic["rhat"] < 1.01
    assert diagnostic["bulk_ess"] > 400
    assert diagnostic["tail_ess"] > 400


def test_constant_sparse_and_slow_chains_cannot_gain_qualification():
    constant = chain_diagnostics(np.ones((4, 1024)))
    assert constant["status"] == "unresolved" and constant["rhat"] is None
    sparse = np.zeros((4, 1024))
    sparse[0, 0] = 1.
    assert chain_diagnostics(sparse, binary=True)["status"] == "unresolved"
    rng = np.random.default_rng(1024)
    slow = np.zeros((4, 4096))
    slow[:, 0] = rng.normal(size=4) * 10
    for step in range(1, slow.shape[1]):
        slow[:, step] = .999 * slow[:, step-1] + rng.normal(size=4) * .02
    diagnosed = chain_diagnostics(slow)
    assert diagnosed["status"] == "failed"
    assert diagnosed["rhat"] > 1.01 or diagnosed["bulk_ess"] < 400
    assert chain_diagnostics(rng.normal(size=(3, 4096)))["status"] == "unresolved"


class ExampleSystem:
    name = "ALDP"
    dim = 3

    def state(self, endpoint):
        return (550., 300.)[endpoint]

    def regions(self, points):
        return {"a": points[:, 0] < -.2, "b": points[:, 0] > .2,
                "remainder": points[:, 0].abs() <= .2,
                "valid": torch.isfinite(points).all(-1),
                "angles": points[:, 1:] * 50}

    def in_domain(self, points):
        return torch.isfinite(points).all(-1)


def example_bank():
    positions = np.random.default_rng(1024).normal(size=(4, 2048, 2, 3))
    energy = .5 * np.square(positions).sum(-1)
    cross = np.stack((energy, energy * 300/550), axis=2)
    return {"positions": positions, "u_kn": cross, "states": [300., 550.],
            "source_unit_ids": [f"ladder-{i}" for i in range(4)], "mode": "formal",
            "hmc_acceptance": [[.8, .8]] * 4, "exchange_acceptance": [[.5]] * 4}


def protocol():
    return {"protocol": {"kernel": "metropolis_hmc_replica_exchange", "warmup": 1000,
                         "step_size": .05, "leapfrog_steps": 8,
                         "warmup_selection": "frozen_on_independent_D",
                         "initialization_strategy": "four_separately_initialized_ladders",
                         "numerical_validation_passed": True}}


def test_bank_uses_ladders_and_actual_aldp_endpoint_order():
    bank = example_bank()
    report = qualify_bank(ExampleSystem(), bank, protocol())
    assert report["endpoint_support"]["0"]["state_index"] == 1
    assert report["endpoint_support"]["1"]["state_index"] == 0
    assert report["eligible_for_formal"]
    assert report["ladders"] == 4
    assert report["identity_weight_summary"]["importance_ess"] < 4 * 2048
    bank["mode"] = "smoke"
    assert not qualify_bank(ExampleSystem(), bank, protocol())["eligible_for_formal"]
    bank["mode"] = "formal"
    assert not qualify_bank(ExampleSystem(), bank, {})["eligible_for_formal"]


def test_duplicate_ladder_ids_and_copied_trajectories_rejected():
    bank = example_bank()
    bank["source_unit_ids"][1] = bank["source_unit_ids"][0]
    with pytest.raises(ValueError, match="唯一独立"):
        qualify_bank(ExampleSystem(), bank, protocol())
    bank = example_bank()
    bank["positions"][1] = bank["positions"][0]
    with pytest.raises(ValueError, match="完全相同"):
        qualify_bank(ExampleSystem(), bank, protocol())


def test_zero_native_region_hits_remain_visible_and_unresolved():
    bank = example_bank()
    bank["positions"][:, :, :, 0] = np.abs(bank["positions"][:, :, :, 0]) + .3
    report = qualify_bank(ExampleSystem(), bank, protocol())
    assert report["endpoint_support"]["0"]["a"]["zero_hit"]
    assert report["endpoint_support"]["0"]["a"]["total"] == 0
    assert not report["eligible_for_formal"]
    assert report["diagnostics"]["endpoint/0/indicator/a"]["status"] == "unresolved"
