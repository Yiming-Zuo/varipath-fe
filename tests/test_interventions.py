import math

import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from scipy.integrate import quad

from varipath.interventions import (
    ConjugateRoute, GeometryQualification, IdentityLoop, InterventionError,
    ProtectedTranslation, conditional_hellinger, finite_intervention, geometric_mass_prediction,
    response_samples, window_weights,
)


DTYPE = torch.float64


class Guard:
    lipschitz = 1.875

    def __call__(self, z):
        radius = torch.linalg.vector_norm(z, dim=-1)
        u = (radius - 4).clamp(0, 1)
        return 1 - 10 * u ** 3 + 15 * u ** 4 - 6 * u ** 5


class Base:
    def __init__(self, n_steps=2, scale=0.):
        self.n_steps, self.scale = n_steps, scale

    def single_step(self, z, k):
        return z + self.scale * torch.tanh(z)

    def single_inverse_step(self, y, k):
        x = y
        for _ in range(20):
            x = y - self.scale * torch.tanh(x)
        return x


def make_route(shift, epsilon=0., conjugated=False, scale=0.):
    guard = Guard()
    loop = ProtectedTranslation(guard, torch.tensor([shift], dtype=DTYPE))
    return ConjugateRoute(Base(scale=scale), [IdentityLoop(), loop, IdentityLoop()], guard,
                          torch.zeros(1, dtype=DTYPE), 1., .5,
                          torch.tensor([0., 1.], dtype=DTYPE), epsilon=epsilon,
                          conjugated_kicks=conjugated)


def test_double_well_actual_finite_maps_reverse_mass_and_inverse():
    source = torch.tensor([[-1.2], [-.3], [0.], [.6], [1.4]], dtype=DTYPE)
    off_minus, off_plus = make_route(-.8), make_route(.8)
    assert torch.allclose(off_minus.forward(source)[0], off_plus.forward(source)[0], atol=1e-12)
    normalizer = quad(lambda x: math.exp(-(x*x-1)**2), -5, 5, epsabs=1e-12)[0]
    changes = []
    for shift in (-.8, .8):
        route = make_route(shift, .1)
        y, logdet = route.forward(source)
        recovered, inverse_ld = route.inverse(y)
        assert torch.allclose(recovered, source, atol=2e-12)
        assert torch.allclose(logdet + inverse_ld, torch.zeros_like(logdet), atol=2e-12)
        root = brentq(lambda x: float(route.single_value(torch.tensor([x], dtype=DTYPE))), -.2, .2, xtol=1e-13)
        changes.append(quad(lambda x: math.exp(-(x*x-1)**2), root, 0)[0] / normalizer)
    assert changes[0] > .002
    assert changes[1] < -.002
    assert changes[0] == pytest.approx(-changes[1], abs=1e-12)


def test_conjugated_kicks_recover_same_corrected_endpoint_and_logweight():
    source = torch.linspace(-2, 2, 11, dtype=DTYPE)[:, None]
    routes = [make_route(shift, .15, conjugated=True, scale=.04) for shift in (-.8, .8)]
    first, second = (route.forward(source) for route in routes)
    assert torch.allclose(first[0], second[0], atol=2e-12)
    assert torch.allclose(first[1], second[1], atol=2e-12)
    absolute_first = make_route(-.8, .15, scale=.04).forward(source)[0]
    absolute_second = make_route(.8, .15, scale=.04).forward(source)[0]
    assert float((absolute_first - absolute_second).abs().max()) > .05


def test_score_predicts_hard_region_mass_without_indicator_derivative():
    nodes, weights = np.polynomial.legendre.leggauss(32)
    positive = 1.5 * (nodes + 1)
    x = np.concatenate((-positive[::-1], positive))
    quadrature = np.concatenate((1.5 * weights[::-1], 1.5 * weights))
    source_density = np.exp(-(x*x-1)**2)
    probability = quadrature * source_density
    probability /= probability.sum()
    z = torch.tensor(x[:, None], dtype=DTYPE)
    u0 = lambda v: (v[:, 0].square() - 1).square()
    u1 = lambda v: u0(v) - .05 * torch.tanh(v[:, 0])
    route = make_route(-.8)
    result = response_samples(route, z, u0, u1)
    score = result["score"].detach().numpy()
    dotlog = result["dot_logweight"].detach().numpy()
    loga = result["logweight"].detach().numpy()
    normalization = quad(lambda a: math.exp(-(a*a-1)**2), -5, 5)[0]
    expected_mass_derivative = .5 * .8 * math.exp(-.8**2/2) * math.exp(-1) / normalization
    assert np.sum(probability[x > 0] * score[x > 0]) == pytest.approx(expected_mass_derivative, abs=2e-8)
    assert abs(np.sum(probability * score)) < 2e-8
    assert abs(np.sum(probability * np.exp(loga) * dotlog)) < 2e-8
    assert abs(np.sum(probability * np.exp(2*loga) * (2*dotlog + score))) < 2e-8
    epsilon = 1e-4
    moments = []
    for dose in (-epsilon, epsilon):
        trial = make_route(-.8, dose)
        y, logdet = trial.forward(z)
        actual_logweight = u0(z) - u1(y) + logdet
        moments.append(float(torch.log((torch.tensor(probability) * torch.exp(2*actual_logweight)).sum())))
    derivative = (moments[1] - moments[0]) / (2*epsilon)
    predicted = 2 * np.sum(probability*np.exp(2*loga)*dotlog) / np.sum(probability*np.exp(2*loga))
    assert derivative == pytest.approx(predicted, abs=1e-8)


class Shift:
    def __init__(self, shift):
        self.shift = shift

    def forward(self, z):
        return z + self.shift, z.new_zeros(len(z))

    def inverse(self, z):
        return z - self.shift, z.new_zeros(len(z))


def test_inverse_mixture_affinity_matches_gaussians():
    source = torch.tensor(ndtri((np.arange(1000) + .5)/1000)[:, None], dtype=DTYPE)
    u0 = lambda z: .5*z[:, 0].square()
    classify = lambda z: (z[:, 0] > 0).long()
    same = conditional_hellinger(Shift(0.), Shift(0.), source, u0, classify, 2)
    assert abs(float(same["total_h2"])) < 1e-14
    changed = conditional_hellinger(Shift(0.), Shift(.7), source, u0, classify, 2)
    assert float(changed["total_h2"]) == pytest.approx(1-math.exp(-.7**2/8), abs=3e-4)
    assert float(changed["rhoe"][1]) == pytest.approx(ndtr(.7), abs=.001)


def test_window_equal_dose_and_construction_failures():
    times = torch.linspace(0, 1, 10, dtype=DTYPE)
    deltas = torch.full_like(times, 1/10)
    for window in range(3):
        assert float(window_weights(times, deltas, window).sum()) == pytest.approx(1.)
    with pytest.raises(InterventionError, match="收缩界"):
        make_route(.8, 1.)
    with pytest.raises(InterventionError, match="事前上限"):
        ProtectedTranslation(Guard(), torch.tensor([10.]), max_microsteps=1)
    qualification = GeometryQualification(True, 0., 0., 1., True)
    assert not qualification.qualified
    with pytest.raises(InterventionError, match="资格认证"):
        geometric_mass_prediction(torch.zeros(2, 3, 1), torch.zeros(2, 1), torch.ones(2, 1),
                                  torch.zeros(1), 1., .5, torch.ones(2)/2, .1, Guard(), qualification)


def test_full_half_branches_use_same_initial_block_and_actual_weights():
    samples = torch.tensor([[-1.], [-.005], [.1], [.7]], dtype=DTYPE)
    u0 = lambda z: (z[:, 0].square() - 1).square()
    u1 = lambda z: u0(z) - .1 * torch.tanh(z[:, 0])
    classify = lambda z: (z[:, 0] > 0).long()
    result = finite_intervention(make_route(-.8, .1), samples, u0, u1, classify, 2, include_shape=False)
    assert float(result["mass_change"][1]) == pytest.approx(.25)
    assert float(result["mass_change"].sum()) == pytest.approx(0.)
    expected = result["full"]["log_second_moment"] - result["baseline"]["log_second_moment"]
    assert float(result["log_risk_change"]) == pytest.approx(float(expected))
    assert not torch.allclose(result["full"]["endpoint"], result["half"]["endpoint"])
