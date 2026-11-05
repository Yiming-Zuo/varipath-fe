import numpy as np
import pytest
import torch

from varipath.physical_analysis import freeze_energy_bins, pii_beta_preference, risk_partition


def test_empirical_quota_shape_identity_and_empty_support():
    loga = torch.tensor([-.3, 1., .2, -.1, .4], dtype=torch.float64)
    result = risk_partition(loga, [0, 0, 1, 1, 1], 3, reference_probabilities=[.2, .6, .2])
    assert result["identity_residual"] < 1e-14
    assert result["status"] == "unestimable"
    assert result["unsupported_reference_cells"] == [2]
    assert result["quota"] >= 0 and result["shape"] >= -1e-14


def test_frozen_energy_bins_and_pii_beta_from_same_joint_cells():
    edges = freeze_energy_bins(np.arange(40), 16)
    assert edges[0] == -np.inf and edges[-1] == np.inf and np.all(np.diff(edges) > 0)
    source = np.ones(25)/25
    target = source.reshape(5, 5).copy()
    target[2] *= 2
    target /= target.sum()
    result = pii_beta_preference(source, target.ravel())
    assert result["native_defined"] and result["log_odds_difference"] > 0
    assert not pii_beta_preference(np.zeros(25), np.zeros(25))["native_defined"]
