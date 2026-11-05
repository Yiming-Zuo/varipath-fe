import numpy as np
import pytest
import torch

from varipath.physics import AffineSystem, MolecularSystem


@pytest.mark.parametrize("name,dimension,constraints",[("ALDP",48,12),("ALA4",123,27)])
def test_native_physics_coordinates_and_measure(name,dimension,constraints):
    system=MolecularSystem(name)
    assert system.dim==dimension
    assert len(system.chart.hydrogens)==constraints
    z=system.initial_coordinates()
    x=system.positions(z)
    assert system.in_domain(z)
    assert system.chart.constraint_error(x)<1e-14
    torch.testing.assert_close(system.encode(x),z,atol=2e-14,rtol=1e-13)
    assert ((x*system.chart.masses[:,None]).sum(0).abs()<1e-13).all()
    oracle=system.oracle()
    try:
        for s in ([1.] if name=="ALDP" else [0.,.5,1.]):
            native,force=oracle.evaluate(x.numpy(),s)
            energy=system.energy(x,s)
            gradient=torch.func.grad(lambda p:system.energy(p,s))(x)
            assert abs(float(energy)-native)<2e-9
            torch.testing.assert_close(gradient,-torch.from_numpy(force),atol=2e-8,rtol=2e-10)
    finally:
        oracle.close()
    moved=z+.0002*torch.sin(torch.arange(dimension,dtype=torch.float64))
    constants=[system.chart.dense_log_volume(q)-system.log_volume(q) for q in (z,moved)]
    assert abs(float(constants[1]-constants[0]))<1e-10
    dense_gradient=torch.func.grad(system.chart.dense_log_volume)(z)
    fast_gradient=torch.func.grad(system.log_volume)(z)
    torch.testing.assert_close(dense_gradient,fast_gradient,atol=2e-9,rtol=2e-9)


def test_polar_energy_is_linear_in_s_and_temperature_direction():
    system=MolecularSystem("ALA4")
    x=system.positions(system.initial_coordinates())
    torch.testing.assert_close(system.energy(x,.5),(system.energy(x,0)+system.energy(x,1))/2,atol=2e-12,rtol=0)
    aldp=MolecularSystem("ALDP")
    assert aldp.state(0)==550.
    assert aldp.state(1)==300.


def test_encoder_rejects_constraint_violation_and_domain_reflection():
    system=MolecularSystem("ALA4")
    x=system.positions(system.initial_coordinates())
    invalid=x.clone()
    invalid[system.chart.hydrogens[0],0]+=.001
    with pytest.raises(ValueError,match="投影"):
        system.encode(invalid)
    reflected=x.clone()
    reflected[:,0]*=-1
    z=system.encode(reflected)
    assert not system.in_domain(z)
    assert torch.isposinf(system.effective_u(z,0))


def test_affine_change_of_coordinates_and_fine_partition():
    raw=MolecularSystem("ALA4")
    center=raw.initial_coordinates()
    scale=torch.linspace(.5,1.5,raw.dim,dtype=torch.float64)
    system=AffineSystem(raw,center,scale)
    z=torch.zeros(raw.dim,dtype=torch.float64)
    torch.testing.assert_close(system.positions(z),raw.positions(center))
    torch.testing.assert_close(system.effective_u(z,1),raw.effective_u(center,1)-scale.log().sum())
    torch.testing.assert_close(system.log_volume(z),raw.log_volume(center)+scale.log().sum())
    region=system.regions(z)
    assert sum(bool(region[k]) for k in ("a","b","remainder"))==1
    assert 0<=int(region["fine"])<25
    assert region["valid"]


def test_degenerate_angles_are_uncertain_not_remainder():
    system=MolecularSystem("ALDP")
    x=system.positions(system.initial_coordinates())
    x[system.quads[0,0]]=x[system.quads[0,1]]
    region=system.region_positions(x)
    assert not region["valid"]
    assert int(region["fine"])==-1
    assert not region["remainder"]
