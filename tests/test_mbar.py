import math

import numpy as np
import pytest
from scipy.special import ndtr
import torch

from varipath.baselines.mbar import EnergyBias, OverlapError, PeriodicCoreBias, fit_mbar, window_potentials


def test_mbar_recovers_ratio_and_hard_regions_including_biased_windows():
    rng = np.random.default_rng(571)


    means = np.array([-0.5, 0.2, 0.8])
    springs = np.array([1.0, 1.7, 0.6])
    shifts = np.array([0.4, -1.3, 0.9])
    count = 10000
    x = np.concatenate([rng.normal(m, 1 / math.sqrt(k), count) for m, k in zip(means, springs)])
    u = 0.5 * springs[:, None] * (x[None] - means[:, None]) ** 2 + shifts[:, None]
    result = fit_mbar(u, [count] * 3, endpoint_u=u[[0, -1]], region_flags=np.column_stack([x > 0, x < -0.8]))
    exact = shifts[-1] - shifts[0] + 0.5 * math.log(springs[-1] / springs[0])
    assert result["endpoint_delta_f"] == pytest.approx(exact, abs=0.02)
    probabilities = np.asarray(result["endpoint_probabilities"])
    target = np.array([[ndtr(means[i] * math.sqrt(springs[i])), ndtr((-0.8 - means[i]) * math.sqrt(springs[i]))] for i in (0, 2)])
    np.testing.assert_allclose(probabilities, target, atol=0.015)
    assert result["fixed_point_residual"] < 1e-8


def test_mbar_rejects_disconnected_network_and_duplicate_records():
    u = np.array([[0, 0, 1e6, 1e6], [1e6, 1e6, 0, 0]], dtype=float)
    with pytest.raises(OverlapError):
        fit_mbar(u, [2, 2])
    with pytest.raises(ValueError, match="counted twice"):
        fit_mbar(np.zeros((2, 4)), [2, 2], sample_ids=["a", "b", "a", "d"])
    with pytest.raises(ValueError, match="counts"):
        fit_mbar(np.zeros((2, 4)), [2, 1])


def test_known_bias_cross_energies_and_gradients():
    x = torch.tensor([[-0.7, 0.1], [0.4, -0.2]], dtype=torch.float64)
    energy_bias = EnergyBias(lambda z: z.square().sum(), 0.3, 1.4)
    core = PeriodicCoreBias(lambda z: z, [[-0.2, 0.4], [0.7, -0.3]], 0.6, 3.0)
    potential = lambda z, state: 0.5 * z.square().sum() + state * z[0]
    result = window_potentials(x, [0.0, 0.5], potential, [energy_bias, core])
    expected = torch.stack([torch.stack([potential(z, 0.0) + energy_bias(z) for z in x]), torch.stack([potential(z, 0.5) + core(z) for z in x])])
    torch.testing.assert_close(result, expected)
    z = x[0].clone().requires_grad_()
    grad = torch.autograd.grad(core(z), z)[0]
    eps = 1e-6
    finite = torch.stack([(core(z.detach() + eps * e) - core(z.detach() - eps * e)) / (2 * eps) for e in torch.eye(2, dtype=z.dtype)])
    torch.testing.assert_close(grad, finite, rtol=1e-6, atol=1e-8)
