import math

import pytest
import torch

from varipath.interventions import (ConjugateRoute, IdentityLoop, InterventionError,
    ProtectedTranslation, reference_response_samples_whole, response_samples)
from varipath.protection import RadialProtection
from varipath.transport import Transport


@pytest.fixture(autouse=True)
def single_thread_double():
    dtype,threads=torch.get_default_dtype(),torch.get_num_threads()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    yield
    torch.set_default_dtype(dtype);torch.set_num_threads(threads)


class NonlinearBase:
    def __init__(self,steps=3):
        self.n_steps=steps
        self.matrix=torch.tensor([[.7,-.2],[.1,.6]])

    def single_step(self,z,k):
        return z+.08*torch.tanh(self.matrix@z)/self.n_steps

    def single_inverse_step(self,y,k):
        z=y
        for _ in range(20):z=y-.08*torch.tanh(self.matrix@z)/self.n_steps
        return z


def route(conjugated=False,steps=3):
    guard=RadialProtection([0.,0.],.5,2.)
    loops=[IdentityLoop()]
    for k in range(1,steps):
        loops.append(ProtectedTranslation(guard,torch.tensor([.18*math.sin(k),.12*math.cos(k)]),tolerance=1e-14))
    loops.append(IdentityLoop())
    return ConjugateRoute(NonlinearBase(steps),loops,guard,torch.tensor([.1,-.1]),.9,.3,
        torch.ones(steps)/steps,epsilon=.02,conjugated_kicks=conjugated,tolerance=1e-13)


def u0(z):
    return .5*z.square().sum(-1)-.1*torch.log1p(z[...,1].square())


def u1(z):
    return .5*((z-z.new_tensor([.25,-.2]))*z.new_tensor([1.2,.8])).square().sum(-1)-.1*torch.log1p(z[...,1].square())


@pytest.mark.parametrize("conjugated",[False,True])
def test_all_streamed_response_fields_match_whole_nested_ad(conjugated):
    model=route(conjugated)
    source=torch.tensor([[.2,-.3],[.9,.1],[-1.4,.2]])
    measured=response_samples(model,source,u0,u1)
    expected=reference_response_samples_whole(model,source,u0,u1)
    assert measured.keys()==expected.keys()
    for key in measured:
        torch.testing.assert_close(measured[key],expected[key],atol=2e-10,rtol=2e-10,msg=lambda msg:f"{key}: {msg}")
        assert not measured[key].requires_grad and measured[key].grad_fn is None


@pytest.mark.parametrize("conjugated",[False,True])
def test_accumulated_local_logdets_and_inverse_equal_whole_ad(conjugated):
    model=route(conjugated)
    source=torch.tensor([[.2,-.3],[.9,.1],[-1.4,.2]])
    endpoint,logdet=model.forward(source)
    expected=torch.stack([model.single_value(z) for z in source])
    whole=torch.stack([model.reference_whole_logdet(z) for z in source])
    torch.testing.assert_close(endpoint,expected,atol=1e-13,rtol=1e-13)
    torch.testing.assert_close(logdet,whole,atol=2e-12,rtol=2e-12)
    inverse,ild=model.inverse(endpoint)
    torch.testing.assert_close(inverse,source,atol=2e-12,rtol=1e-12)
    torch.testing.assert_close(ild,-logdet,atol=2e-12,rtol=1e-12)


def test_actual_nonzero_spline_transport_has_identical_response():
    guard=RadialProtection([0.,0.],10.,100.)
    base=Transport(2,guard,seed=804)
    with torch.no_grad():
        base.C.copy_(.02*torch.sin(torch.arange(base.C.numel()).reshape_as(base.C)))
    for p in base.parameters():p.requires_grad_(False)
    loops=[IdentityLoop() for _ in range(base.n_steps+1)]
    loops[base.n_steps//2]=ProtectedTranslation(guard,torch.tensor([.15,-.1]))
    model=ConjugateRoute(base,loops,guard,torch.tensor([.1,-.1]),1.,.3,
        torch.ones(base.n_steps)/base.n_steps,epsilon=.03)
    source=torch.tensor([[.3,-.2]])
    measured=response_samples(model,source,u0,u1)
    expected=reference_response_samples_whole(model,source,u0,u1)
    for key in measured:
        torch.testing.assert_close(measured[key],expected[key],atol=2e-10,rtol=2e-10)


def test_production_does_not_call_or_retain_whole_map_graph():
    model=route(steps=24)
    def prohibited(*args,**kwargs):raise AssertionError("whole-map AD used by production")
    model.single_value=prohibited
    model.reference_whole_logdet=prohibited
    source=torch.tensor([[.3,-.2]],requires_grad=True)
    endpoint,logdet=model.forward(source)
    inverse,ild=model.inverse(endpoint)
    measured=response_samples(model,source,u0,u1)
    assert source.grad is None
    assert all(value.grad_fn is None and not value.requires_grad for value in (*measured.values(),endpoint,logdet,inverse,ild))
    torch.testing.assert_close(inverse,source.detach(),atol=2e-12,rtol=1e-12)


def test_ill_conditioned_prefix_is_failure_without_pseudoinverse():
    class IllConditioned:
        n_steps=1
        def single_step(self,z,k):return z*z.new_tensor([1e7,1e-7])
    guard=RadialProtection([0.,0.],10.,100.)
    model=ConjugateRoute(IllConditioned(),[IdentityLoop(),IdentityLoop()],guard,
        torch.zeros(2),1.,.3,torch.ones(1))
    with pytest.raises(InterventionError,match="病态.*伪逆"):
        response_samples(model,torch.tensor([[.2,-.1]]),u0,u1)


def test_streamed_budget_abort_propagates_before_whole_output():
    model=route()
    count=0
    def budget():
        nonlocal count
        count+=1
        if count==2:raise TimeoutError("test budget")
    with pytest.raises(TimeoutError):
        model.forward(torch.tensor([[.3,-.2]]),check_budget=budget)
    count=0
    with pytest.raises(TimeoutError):
        response_samples(model,torch.tensor([[.3,-.2]]),u0,u1,check_budget=budget)


def test_streamed_dose_proxy_and_geometric_exposure_match_complete_small_trace(monkeypatch):
    import varipath.mechanism as mechanism
    from varipath.interventions import GeometryQualification,geometric_mass_prediction
    model=route()
    x=torch.linspace(-.45,.45,37)
    source=torch.stack([x,.3*torch.sin(3*x)],1)
    baseline=mechanism._trace(model,source,0.)
    perturbed=mechanism._trace(model,source,model.epsilon)
    displacement=(perturbed-baseline).norm(dim=-1).amax(-1)
    boundary=torch.zeros(model.n_steps,2)
    normals=torch.zeros_like(boundary);normals[:,0]=1.
    relative=baseline[:,1::2]-boundary
    nu=(relative*normals).sum(-1)
    tangent=relative-nu[...,None]*normals
    expected_proxy=(((nu.abs()<.1*model.sigma)*torch.exp(-tangent.square().sum(-1)/(2*model.sigma**2)))*model.eta[None]).sum(-1).mean()
    geometry=GeometryQualification(True,0.,0.,model.sigma,True,certified=True,reason="Analytical helper control only, not molecular qualification")
    cell=dict(route=model,boundary=boundary,normals=normals,sigma=model.sigma,eta=model.eta,
        epsilon=model.epsilon,amplitude=model.amplitude,anchor_center=model.center,geometry=geometry)
    nodes=baseline[:,:-1:2].transpose(0,1)
    expected_geom=geometric_mass_prediction(nodes,boundary,normals,model.center,model.sigma,model.amplitude,model.eta,model.epsilon,model.protection,geometry)
    expected_half=geometric_mass_prediction(nodes,boundary,normals,model.center,model.sigma,model.amplitude,model.eta,model.epsilon,model.protection,geometry,bandwidth_factor=.05)
    def prohibited(*args,**kwargs):raise AssertionError("formal helper retained complete sample trajectories")
    monkeypatch.setattr(mechanism,"_trace",prohibited)
    passed,diagnostic=mechanism._dose_check(model,source,model.epsilon,model.sigma,source_batch_size=7)
    assert passed
    assert diagnostic["p95_maximum_path_displacement"]==pytest.approx(float(torch.quantile(displacement,.95)),abs=2e-14)
    assert diagnostic["maximum_path_displacement"]==pytest.approx(float(displacement.max()),abs=2e-14)
    proxy=mechanism._primary_proxy_stream(model,source,boundary,normals,model.sigma,model.eta,source_batch_size=7)
    torch.testing.assert_close(proxy,expected_proxy,atol=1e-14,rtol=1e-13)
    measured,slab=mechanism._geometric_prediction_stream(cell,source,model.protection,source_batch_size=7)
    torch.testing.assert_close(measured,expected_geom,atol=1e-14,rtol=1e-13)
    assert slab["half_slab_prediction"]==pytest.approx(float(expected_half),abs=1e-14)
    second,_=mechanism._geometric_prediction_stream(cell,source,model.protection,source_batch_size=11)
    torch.testing.assert_close(measured,second,atol=1e-14,rtol=1e-13)


def test_panel_memory_preflight_keeps_fixed_slots_and_does_not_reduce_grid():
    from test_mechanism import DoubleWell,IdentityBase
    from varipath.mechanism import prepare_panel,_panel_memory_preflight
    source=torch.linspace(-2,2,17)[:,None]
    anchor={"status":"certified","certificate":{"certified":True},"z":[0.],"sigma":.5,"amplitude":.5,
        "minima":[{"a":False},{"a":True}]}
    guard=RadialProtection([0.],8.,10.)
    base=IdentityBase()
    result=prepare_panel(base,DoubleWell(),guard,[anchor],source,{"panel_memory_limit_bytes":1})
    assert len(result.cells)==48 and result.base.n_steps==base.n_steps
    assert all(cell["status"]=="construction_failed" and cell["failure_type"]=="resource_limit" for cell in result.cells)
    assert all("不是几何或物理不可构造" in cell["reason"] for cell in result.cells)
    estimate=_panel_memory_preflight(1_000_000,123,4,{"panel_memory_limit_bytes":128*1024**3})
    assert estimate["planned_ready_tensor_lower_bound_bytes"]>60*1024**3
    assert estimate["status"]=="planned_lower_bound_within_limit"
