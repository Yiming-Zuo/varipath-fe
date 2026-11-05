import pytest
import torch

from varipath.ingress import IngressError,transport_configurations
from varipath.physics import AffineSystem,MolecularSystem,R_KJ
from varipath.protection import RadialProtection
from varipath.sampling import haar_rotation
from varipath.transport import Transport


def actual_fixture(name="ALDP"):
    raw=MolecularSystem(name)
    z=raw.initial_coordinates()
    system=AffineSystem(raw,z,torch.ones_like(z))

    model=Transport(raw.dim,RadialProtection(torch.zeros(raw.dim),4.,8.),spline_count=4)
    return raw,system,model


@pytest.mark.parametrize("name",["ALDP","ALA4"])
def test_certified_zero_field_uses_physical_identity_without_log_volume(monkeypatch,name):
    raw,system,_=actual_fixture(name)
    x=raw.positions(raw.initial_coordinates())
    guard=RadialProtection(torch.ones(raw.dim)*100,1.,2.)
    model=Transport(raw.dim,guard,spline_count=4)
    def no_volume(*args,**kwargs):
        raise AssertionError("identity must not evaluate chart volume")
    monkeypatch.setattr(system,"log_volume",no_volume)
    monkeypatch.setattr(raw,"log_volume",no_volume)
    result=transport_configurations(system,model,x)
    assert result["identity"].all()
    assert result["identity_reason"]==["zero_field"]
    torch.testing.assert_close(result["positions"][0],x,atol=0,rtol=0)
    physical=raw.energy(x,1.)/R_KJ*(1/550.-1/300.) if name=="ALDP" else (raw.energy(x,0.)-raw.energy(x,1.))/(R_KJ*300)
    torch.testing.assert_close(result["log_weights"][0],physical,atol=2e-14,rtol=2e-14)
    assert result["log_mass_jacobian"][0]==0


@pytest.mark.parametrize("name",["ALDP","ALA4"])
def test_exact_methyl_south_pole_uses_known_zero_measure_identity(monkeypatch,name):
    raw,system,model=actual_fixture(name)
    x=raw.positions(raw.initial_coordinates()).clone()
    hydrogen=0
    index=raw.chart.hydrogens.index(hydrogen)
    parent=raw.chart.parents[index]
    assert hydrogen not in raw.chiral_quads.flatten().tolist()
    x[hydrogen]=x[parent]+x.new_tensor([0.,0.,-float(raw.chart.lengths[index])])
    x=x-(x*raw.chart.masses[:,None]).sum(0)/raw.chart.masses.sum()
    assert raw.chart.constraint_error(x)<1e-14
    assert raw.in_domain_positions(x)
    assert raw.region_positions(x)["valid"]
    oracle=raw.oracle()
    try:
        native,_=oracle.evaluate(x.numpy())
        assert float(raw.energy(x))==pytest.approx(native,abs=2e-8,rel=2e-12)
    finally:
        oracle.close()
    def forbidden(*args,**kwargs):
        raise AssertionError("south-pole extension must not encode or evaluate volume")
    monkeypatch.setattr(system,"encode",forbidden)
    monkeypatch.setattr(system,"log_volume",forbidden)
    result=transport_configurations(system,model,x)
    assert result["identity_reason"]==["sphere_chart_south_pole"]
    torch.testing.assert_close(result["positions"][0],x,atol=0,rtol=0)
    assert torch.isfinite(result["log_weights"]).all()


def test_active_actual_map_carries_rotation_and_matches_effective_weight():
    raw,system,model=actual_fixture()
    with torch.no_grad():
        model.C.fill_(1e-4)
    rotation=haar_rotation(torch.Generator().manual_seed(17))
    shift=torch.tensor([.25,-.4,.2],dtype=torch.float64)
    x=raw.positions(raw.initial_coordinates())@rotation+shift
    shape=system.encode(x)
    with torch.no_grad():
        moved,ordinary=model(shape[None])
        expected_log,expected_y=model.log_weights(shape[None],lambda z:system.effective_u(z,0),lambda z:system.effective_u(z,1))
    result=transport_configurations(system,model,x)
    assert not result["identity"].any()
    assert result["identity_reason"]==[None]
    torch.testing.assert_close(result["rotations"][0],rotation,atol=2e-15,rtol=0)
    torch.testing.assert_close(result["positions"],system.positions(expected_y)@rotation+shift,atol=2e-15,rtol=0)
    torch.testing.assert_close(result["log_weights"],expected_log,atol=3e-11,rtol=3e-11)
    torch.testing.assert_close(result["ordinary_logdet"],ordinary,atol=0,rtol=0)
    assert result["inverse_residual"].max()<1e-12


def test_arbitrary_encoding_or_inverse_failure_cannot_become_identity(monkeypatch):
    raw,system,model=actual_fixture()
    x=raw.positions(raw.initial_coordinates())
    original=system.encode
    def bad_encode(*args,**kwargs):
        raise ValueError("unresolved chart problem")
    monkeypatch.setattr(system,"encode",bad_encode)
    with pytest.raises(IngressError,match="形状图"):
        transport_configurations(system,model,x)
    monkeypatch.setattr(system,"encode",original)
    def bad_inverse(*args,**kwargs):
        raise RuntimeError("configured inverse failure")
    monkeypatch.setattr(model,"inverse",bad_inverse)
    with pytest.raises(IngressError,match="逆解"):
        transport_configurations(system,model,x)


def test_invalid_constraints_chirality_labels_and_energy_fail_whole_batch(monkeypatch):
    raw,system,model=actual_fixture()
    x=raw.positions(raw.initial_coordinates())
    invalid=x.clone()
    invalid[raw.chart.hydrogens[0],0]+=.001
    with pytest.raises(IngressError,match="约束"):
        transport_configurations(system,model,torch.stack([x,invalid]))
    reflected=x.clone()
    reflected[:,0]*=-1
    with pytest.raises(IngressError,match="全 L"):
        transport_configurations(system,model,torch.stack([x,reflected]))
    degenerate=x.clone()
    degenerate[raw.quads[0,0]]=degenerate[raw.quads[0,1]]
    with pytest.raises(IngressError,match="标签"):
        transport_configurations(system,model,torch.stack([x,degenerate]))
    monkeypatch.setattr(raw.energy,"forward",lambda positions,s:positions.new_tensor(float("nan")))
    with pytest.raises(IngressError,match="能量"):
        transport_configurations(system,model,x)
