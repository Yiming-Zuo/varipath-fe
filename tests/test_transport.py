import math

import numpy as np
import pytest
import torch
from scipy.interpolate import BSpline

from varipath.protection import RadialProtection
from varipath.transport import Transport


torch.set_num_threads(1)


def make_model(anchors=False):
    guard = RadialProtection(torch.zeros(2), 4.0, 8.0)
    kw = dict(centers=[[0.2, -0.1]], bandwidths=[1.0], amplitudes=[0.4]) if anchors else {}
    return Transport(2, guard, spline_count=4, **kw)


def test_identity_initialization_exact_map_and_weight():
    model = make_model(True)
    x = torch.tensor([[0.3, -0.1], [1.0, 0.5]], dtype=torch.float64)
    y, ld = model(x)
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    torch.testing.assert_close(ld, torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
    u0 = lambda z: z.square().sum() / 2
    u1 = lambda z: (z - 0.3).square().sum() / 2 + 1.2
    loga, _ = model.log_weights(x, u0, u1)
    torch.testing.assert_close(loga, torch.vmap(u0)(x) - torch.vmap(u1)(x))


def test_exact_composed_logdet_inverse_and_gaussian_gradient():
    model = make_model(True)
    with torch.no_grad():
        model.C.copy_(0.08 * torch.cos(torch.arange(model.C.numel()).reshape_as(model.C)))
        model.alpha.fill_(0.3)
    x = torch.tensor([[0.3, -0.1], [1.0, 0.5]], dtype=torch.float64)
    y, ld = model(x)
    for row, actual in zip(x, ld):
        matrix = torch.func.jacrev(lambda z: model(z[None], check=False)[0][0])(row)
        expected = torch.linalg.slogdet(matrix)[1]
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    back, inverse_ld = model.inverse(y.detach())
    torch.testing.assert_close(back, x, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(ld + inverse_ld, torch.zeros_like(ld), atol=2e-11, rtol=2e-11)
    u0 = lambda z: z.square().sum() / 2
    u1 = lambda z: (z - 0.3).square().sum() / 2
    loss = model.objective(x, u0, u1)
    derivative = torch.autograd.grad(loss, model.alpha)[0][0]
    values = []
    original = model.alpha.detach().clone()
    for sign in (-1, 1):
        with torch.no_grad():
            model.alpha.copy_(original + sign * 1e-5)
        values.append(float(model.objective(x, u0, u1).detach()))
    with torch.no_grad():
        model.alpha.copy_(original)
    assert math.isclose(float(derivative), (values[1] - values[0]) / 2e-5, abs_tol=2e-8, rel_tol=2e-5)


def test_certified_bound_contains_transition_and_field_derivatives():
    model = make_model(True)
    assert model.global_lipschitz / model.n_steps <= 0.25
    with torch.no_grad():
        model.C.fill_(1)
        model.alpha.fill_(1)
    points = torch.tensor([[0, 0], [4.5, 0], [6, 0], [7.5, 0], [9, 0]], dtype=torch.float64)
    for k in (0, model.n_steps // 2, model.n_steps - 1):
        matrices = torch.vmap(torch.func.jacrev(lambda z: model.single_velocity(z, k)))(points)
        assert float(torch.linalg.matrix_norm(matrices, ord=2).max().detach()) <= model.global_lipschitz
    outside = points[-1:]
    y, ld = model(outside)
    torch.testing.assert_close(y, outside, atol=0, rtol=0)
    torch.testing.assert_close(ld, torch.zeros_like(ld), atol=0, rtol=0)


def test_invalid_step_parameters_and_inverse_failure_are_explicit():
    guard = RadialProtection(torch.zeros(2), 4, 8)
    with pytest.raises(ValueError, match="至少需要"):
        Transport(2, guard, steps=1)
    model = make_model()
    with torch.no_grad():
        model.C[0, 0] = 1.01
    with pytest.raises(ValueError, match="参数"):
        model(torch.zeros(1, 2, dtype=torch.float64))
    model = make_model()
    model.inverse_iterations = 1
    with torch.no_grad():
        model.C.fill_(0.5)
    with pytest.raises(RuntimeError, match="残差"):
        model.inverse(torch.ones(1, 2, dtype=torch.float64))
    with pytest.raises(FloatingPointError, match="非有限"):
        model(torch.full((1, 2), float("nan"), dtype=torch.float64))


def test_frozen_time_centers_and_spectral_bounds():
    model = make_model(True)
    moving = model.centers[None].expand(model.n_steps, -1, -1).clone()
    moved = Transport(2, model.protection, spline_count=4, steps=model.n_steps,
                      centers=model.centers, bandwidths=model.bandwidths,
                      amplitudes=model.amplitudes, time_centers=moving)
    with torch.no_grad():
        moved.raw_U.fill_(2)
        moved.raw_V.fill_(-2)
        moved.raw_W.fill_(2)
    for matrix in (moved.U, moved.V, moved.W):
        assert float(torch.linalg.matrix_norm(matrix, ord=2).detach()) <= 2 + 1e-12


def test_local_spline_basis_matches_clamped_scipy_and_large_grid_is_lazy():
    model = Transport(2, RadialProtection([0, 0], 4, 8), steps=10000001)
    with torch.no_grad():
        model.C.copy_(torch.cos(torch.arange(model.C.numel()).reshape_as(model.C)))
    knots = np.r_[np.zeros(4), np.linspace(0, 1, 48)[1:-1], np.ones(4)]
    for k in (0, 1, 81321, model.n_steps // 2, model.n_steps - 1):
        reference = BSpline(knots, model.C.detach().numpy(), 3)(k / model.n_steps)
        torch.testing.assert_close(model.spline_coefficients(k), torch.from_numpy(reference), atol=1e-14, rtol=1e-14)
    assert sum(value.numel() for value in model.buffers()) < 100


def test_nontrivial_finite_map_recovers_known_partition_ratio():
    model = Transport(1, RadialProtection([0], 4, 8), spline_count=4,
                      centers=[[0.4]], bandwidths=[0.8], amplitudes=[0.3])
    with torch.no_grad():
        model.C.fill_(0.2)
        model.alpha.fill_(0.5)
    points, weights = np.polynomial.hermite.hermgauss(80)
    x = torch.from_numpy((math.sqrt(2) * points)[:, None])
    u0 = lambda z: z.square().sum() / 2
    u1 = lambda z: (z - 0.3).square().sum() / 2 + 0.7
    with torch.no_grad():
        loga, _ = model.log_weights(x, u0, u1)
    estimate = np.dot(weights / math.sqrt(math.pi), loga.exp().numpy())
    assert estimate == pytest.approx(math.exp(-0.7), rel=2e-9, abs=2e-9)
