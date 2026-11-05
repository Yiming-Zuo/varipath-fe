import math
import pytest
import torch

from varipath.catalog import certify_stationary, dimer_search
from varipath.geometry import fit_protection, MolecularProtection
from varipath.physics import AffineSystem, MolecularSystem


def test_geometry_safe_plateau_global_derivative_and_outside_identity():
    raw = MolecularSystem("ALA4")
    pool = torch.stack([raw.initial_coordinates(i) for i in range(4)])
    system = AffineSystem(raw, pool.mean(0), pool.std(0).clamp_min(.01))
    pool = (pool-system.location)/system.scale
    guard = fit_protection(system, pool)
    assert guard.compact_support and torch.all(guard(pool) == 1)
    assert torch.all(torch.vmap(torch.func.grad(guard))(pool).norm(dim=-1) <= guard.lipschitz)
    outside = pool[0].clone()
    outside[0] = guard.upper[0]+1
    assert guard(outside) == 0
    assert torch.equal(torch.func.grad(guard)(outside), torch.zeros_like(outside))
    assert guard.clearance(pool[0]) > 0
    assert guard.normalized_margins(pool[0]).min() >= guard.clearance(pool[0])
    settings = guard.specification()
    settings.pop("lipschitz")
    restored = MolecularProtection(system, **settings)
    torch.testing.assert_close(restored(pool), guard(pool))


def test_dimer_uses_physical_saddle_and_keeps_failed_search_cost():
    potential = lambda z: (z[0]**2-1)**2+.5*z[1]**2
    metric = lambda z: torch.eye(2, dtype=z.dtype)
    zero = torch.zeros(2, dtype=torch.float64)
    cert = certify_stationary(potential, zero, metric)
    assert cert["certified"] and cert["negative_modes"] == 1
    assert not certify_stationary(potential, torch.tensor([1., 0.], dtype=torch.float64), metric)["certified"]
    with pytest.raises(ValueError):
        certify_stationary(potential, zero, metric, gradient_tolerance=math.inf)
    cost = {}
    with pytest.raises(RuntimeError, match="force-call"):
        dimer_search(potential, torch.tensor([.2, .1], dtype=torch.float64), 1024, force_limit=1, cost=cost)
    assert cost["search_force_calls"] == 1
