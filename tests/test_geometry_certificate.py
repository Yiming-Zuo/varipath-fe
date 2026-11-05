import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from varipath.baselines.mnf import MNFIC
from varipath.coordinates import dihedral_radians
from varipath.geometry import MolecularProtection
from varipath.geometry_certificate import certify_geometry, _clearance_interval, _i, _lo
from varipath.interventions import IdentityLoop
from varipath.physics import MolecularSystem
from varipath.transport import Transport


def molecular_boundary(name,targets=None):
    system=MolecularSystem(name)
    rotors=MNFIC(system)
    x=system.positions(system.initial_coordinates())
    targets=targets if targets is not None else ([-30.,-40.] if name=="ALDP" else [-30.,-40.,-70.,-40.])
    for target,q,ids in zip(targets,rotors.quads,rotors.rotors):
        angle=x.new_tensor(math.radians(target))-dihedral_radians(x[q])
        axis=x[q[2]]-x[q[1]];axis=axis/axis.norm()
        delta=x[ids]-x[q[1]]
        arm=delta-(delta*axis).sum(-1,keepdim=True)*axis
        moved=(angle.cos()-1)*arm+angle.sin()*torch.linalg.cross(axis.expand_as(arm),arm)
        x=x.index_add(0,torch.tensor(ids),moved)
    point=system.encode(x)
    def level(z):
        angles=system.regions(z[None])["angles"][0].reshape(-1,2)
        delta=(angles-z.new_tensor([-70.,-40.])+180)%360-180
        return delta.square().sum(-1)/1600-1
    normal=torch.func.grad(lambda z:level(z)[0])(point)
    normal=normal/normal.norm()

    protection=MolecularProtection(system,point-1,point+1,width=1e-5,temperature=1e-9)
    model=Transport(system.dim,protection)
    node=model.n_steps-1
    route=SimpleNamespace(base=model,n_steps=model.n_steps,protection=protection,
        loops={node:IdentityLoop()},center=point)
    return system,point,normal,route,node,level


@pytest.fixture(autouse=True)
def cpu_single_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name",["ALDP","ALA4"])
def test_real_molecular_identity_map_has_small_but_nonzero_qualified_neighborhood(name):
    system,p,n,route,k,level=molecular_boundary(name)
    assert system.in_domain(p)
    assert abs(float(level(p)[0]))<1e-12
    qualification,certificate=certify_geometry(route,system,{k:p},{k:n},1e-7,0,{k:1.})
    assert qualification.qualified,certificate["reasons"]
    assert certificate["certified"] and certificate["qualified"]
    row=certificate["nodes"][0]
    assert row["full_anchor_neighborhood_radius"]==2e-7
    assert row["normal_derivative_lower"]>0
    assert row["endpoint_level_hessian_upper"]>0
    hessian=torch.func.hessian(lambda z:level(z)[0])(p)
    assert float(torch.linalg.matrix_norm(hessian,ord=2))<row["endpoint_level_hessian_upper"]
    assert row["plane_graph_exists_and_unique"]
    assert row["other_patch_excluded"]
    assert row["maximum_normal_pair_angle_upper"]<.1
    assert row["maximum_vertical_departure_upper"]<1e-8


def test_actual_nonlinear_transport_uses_interval_gradient_and_chain_bounds():
    system,p,n,route,k,level=molecular_boundary("ALDP")
    with torch.no_grad():
        route.base.C.copy_(1e-4*torch.sin(torch.arange(route.base.C.numel()).reshape_as(route.base.C)))
    native=lambda z:level(route.base.single_step(z,k))[0]
    gradient=torch.func.grad(native)(p);n=gradient/gradient.norm()
    qualification,certificate=certify_geometry(route,system,{k:p},{k:n},1e-7,0,{k:1.})
    assert qualification.qualified,certificate["reasons"]
    row=certificate["nodes"][0]
    assert row["remaining_map_hessian_upper"]>0
    assert row["remaining_map_lipschitz_upper"]>1
    actual=float(torch.linalg.matrix_norm(torch.func.hessian(native)(p),ord=2).detach())
    assert actual<row["pulled_level_hessian_upper"]


def test_large_real_molecular_neighborhood_returns_numerical_failure_reason():
    system,p,n,route,k,_=molecular_boundary("ALDP")
    qualification,certificate=certify_geometry(route,system,{k:p},{k:n},.2,0,{k:1.})
    assert not qualification.qualified
    assert not certificate["qualified"]
    assert certificate["reasons"] and "平台" in certificate["reasons"][0]
    assert "lower=" in certificate["reasons"][0]


def test_directed_clearance_encloses_actual_value_and_no_fake_oracle():
    system,p,n,route,k,_=molecular_boundary("ALDP")
    lower=_lo(_clearance_interval(route.protection,[_i(v) for v in p.tolist()]))
    actual=float(route.protection.clearance(p))
    assert lower<=actual+1e-14
    assert actual-lower<1e-12
    fake=SimpleNamespace(base=SimpleNamespace(n_steps=1),n_steps=1,protection=route.protection)
    q,certificate=certify_geometry(fake,system,{0:p},{0:n},.001,0,{0:1.})
    assert not q.qualified and "手填" in certificate["reasons"][0]


def test_affine_shape_chart_uses_its_actual_full_metric_state():
    from varipath.physics import AffineSystem
    raw,p,_,_,_,_=molecular_boundary("ALDP")
    system=AffineSystem(raw,p,torch.full_like(p,.3))
    z=torch.zeros_like(p)
    protection=MolecularProtection(system,z-1,z+1,width=1e-5,temperature=1e-9)
    model=Transport(system.dim,protection)
    node=model.n_steps-1
    def level(q):
        angles=system.regions(q[None])["angles"][0]
        return (((angles-q.new_tensor([-70.,-40.])+180)%360-180).square().sum()/1600-1)
    n=torch.func.grad(level)(z);n=n/n.norm()
    route=SimpleNamespace(base=model,n_steps=model.n_steps,protection=protection,loops={node:IdentityLoop()},center=z)
    q,certificate=certify_geometry(route,system,{node:z},{node:n},1e-7,0,{node:1.})
    assert q.qualified,certificate["reasons"]
    foreign=AffineSystem(raw,p,torch.full_like(p,.31))
    q,certificate=certify_geometry(route,foreign,{node:z},{node:n},1e-7,0,{node:1.})
    assert not q.qualified and "不匹配" in certificate["reasons"][0]


def test_ala4_intersection_of_two_real_basin_faces_is_not_one_qualified_patch():
    system,p,n,route,k,_=molecular_boundary("ALA4",[-30.,-40.,-30.,-40.])
    q,certificate=certify_geometry(route,system,{k:p},{k:n},1e-7,0,{k:1.})
    assert not q.qualified
    assert any("另一圆盘片" in reason for reason in certificate["reasons"])
