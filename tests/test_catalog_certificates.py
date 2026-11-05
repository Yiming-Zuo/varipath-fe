from types import SimpleNamespace

import pytest
import torch

from varipath.catalog import certify_stationary,physical_regularity,search_catalog
from varipath.geometry import fit_protection
from varipath.physics import AffineSystem,MolecularSystem


def test_covariant_residual_envelope_rejects_spurious_coordinate_index():
    epsilon,coefficient=.0005,-4000.
    def potential(z):
        x=z[0]+.5*coefficient*z[0]**2
        return .5*x*x+epsilon*x
    metric=lambda z:((1+coefficient*z[0])**2).reshape(1,1)
    certificate=certify_stationary(potential,torch.zeros(1,dtype=torch.float64),metric)
    assert certificate["gradient_mass_norm"]<certificate["gradient_tolerance"]
    assert certificate["eigenvalues"][0]==pytest.approx(1.)
    assert certificate["residual_connection_bound"]==pytest.approx(2.)
    assert certificate["unresolved_modes"]==1
    assert not certificate["certified"]


def test_physical_local_regular_piece_and_branch_boundary_rejection():
    raw=MolecularSystem("ALA4")
    pool=torch.stack([raw.initial_coordinates(i) for i in range(4)])
    system=AffineSystem(raw,pool.mean(0),pool.std(0).clamp_min(.01))
    pool=(pool-system.location)/system.scale
    protection=fit_protection(system,pool)
    result=physical_regularity(system,pool[0],protection)
    assert result["certified"] and result["radius"]>0
    assert set(result["branch_distance_lower_bounds"])=={"screening_zero","absolute_value","max_lower_limit"}
    fake=SimpleNamespace(energy=SimpleNamespace(gb_or=torch.tensor([.1,.2],dtype=torch.float64),gb_sr=torch.tensor([.1,.1],dtype=torch.float64),gb_kind="OBC1"),positions=lambda z:torch.stack([z.new_zeros(3),torch.stack([z[0],z[0]*0,z[0]*0])]))
    guard=SimpleNamespace(normalized_margins=lambda z:z.new_ones(2),position_lipschitz=torch.ones(2,dtype=torch.float64))
    boundary=physical_regularity(fake,torch.tensor([.1],dtype=torch.float64),guard)
    assert not boundary["certified"] and boundary["radius"]==0


def test_catalog_resumes_terminal_starts_without_repeating_search(monkeypatch):
    attempted=[]
    def failed_search(*args,**kwargs):
        attempted.append(1)
        raise ValueError("configured unsuccessful search")
    monkeypatch.setattr("varipath.catalog.dimer_search",failed_search)
    system=SimpleNamespace(dim=2,name="ALDP",metadata={"atoms":[dict(index=0,element="C")]})
    source=torch.zeros(3,2,dtype=torch.float64)
    initial=[dict(surface=1.,start=0,status="failed",reason="completed previously")]
    output=search_catalog(system,source,source,None,dict(seed=1),initial_records=initial)
    assert len(attempted)==2
    assert len(output["records"])==3
    assert output["records"][0]==initial[0]
    with pytest.raises(ValueError,match="duplicated"):
        search_catalog(system,source,source,None,dict(seed=1),initial_records=initial*2)
