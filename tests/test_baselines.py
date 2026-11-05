import math

import numpy as np
import pytest
from scipy.special import i0, ndtr
import torch

from varipath.baselines.bridge import bar, bridge_readout, nebar
from varipath.baselines.feat import FEAT, FEATConfig
from varipath.baselines.mnf import MNFConfig, MNFIC, PeriodicMap, distance_loss
from varipath.physics import MolecularSystem


class GaussianEndpoints:
    dim = 2
    name = "gaussian"

    def effective_u(self, z, endpoint):
        return .5*z.square().sum() if endpoint==0 else .5*((z-.3)/1.1).square().sum()+.2

    def regions(self, z):
        return {"a":z[...,0]>0,"b":z[...,0]<=0}


@pytest.fixture(autouse=True)
def double_single_thread():
    previous = torch.get_default_dtype()
    threads = torch.get_num_threads()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous); torch.set_num_threads(threads)


def test_nebar_matches_pymbar_unequal_counts_and_support():
    import pymbar
    rng = np.random.default_rng(28)
    source, target = rng.normal(size=731), rng.normal(.4,1.2,size=907)
    u0=lambda x:.5*x*x
    u1=lambda x:.5*((x-.4)/1.2)**2+.17
    wf,wr = u1(source)-u0(source), u0(target)-u1(target)
    expected = pymbar.bar(wf,wr,compute_uncertainty=False)["Delta_f"]
    result = bar(wf,wr)
    assert result["free_energy"] == pytest.approx(expected,abs=2e-8)
    assert result["solver_log_equation_residual"] < 1e-10
    supported = nebar(np.array([.2,-np.inf,.1]), np.array([-.1,-np.inf,-.2]))
    assert math.isfinite(supported["log_partition_ratio"])
    with pytest.raises(ValueError,match="重叠"):
        nebar([-np.inf],[-np.inf])


def test_feat_actual_gaussian_path_ratio_and_known_readout():
    model=FEAT(GaussianEndpoints(),FEATConfig(iterations=2,batch_size=4,path_steps=8))
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
    generator=torch.Generator().manual_seed(1122)
    source=torch.randn((10000,2),generator=generator)
    target=.3+1.1*torch.randn((10000,2),generator=generator)
    result=model.evaluate(source,target,generator,store_paths=True)
    expected=2*math.log(1.1)-.2
    assert result["native_nebar"]["log_partition_ratio"] == pytest.approx(expected,abs=.025)
    probabilities=np.array(result["readout"]["endpoint_region_probabilities"])
    assert probabilities[0,0] == pytest.approx(.5,abs=.015)
    assert probabilities[1,0] == pytest.approx(ndtr(.3/1.1),abs=.015)
    fp=result["forward_path"]

    expected_ratio=-torch.vmap(lambda z: model.system.effective_u(z,1))(fp[:,-1])+torch.vmap(lambda z:model.system.effective_u(z,0))(fp[:,0])
    torch.testing.assert_close(result["forward_log_ratio"],expected_ratio,rtol=1e-10,atol=1e-10)
    rp=result["reverse_path"]
    torch.testing.assert_close(model.replay_ratio(rp,False),result["reverse_log_ratio"])


def test_feat_real_losses_and_exact_resume(tmp_path):
    system=GaussianEndpoints()
    generator=torch.Generator().manual_seed(8)
    source=torch.randn((12,2),generator=generator)
    target=.3+1.1*torch.randn((12,2),generator=generator)
    config=FEATConfig(iterations=2,batch_size=4,path_steps=2,checkpoint_every=1)
    straight=FEAT(system,config)
    straight.fit(source,target)
    interrupted=FEAT(system,config)
    checkpoint=tmp_path/"feat.pt"
    assert not interrupted.fit(source,target,checkpoint,stop_after=1)["complete"]
    restored=FEAT(system,config)
    assert restored.fit(source,target,checkpoint,resume=True)["complete"]
    for actual, expected in zip(restored.parameters(),straight.parameters()):
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    with pytest.raises(ValueError,match="训练池"):
        restored.fit(source+.01,target,checkpoint,resume=True)


def test_periodic_map_monotone_inverse_and_full_jacobian():
    flow=PeriodicMap(2)
    with torch.no_grad():
        flow.coeff.copy_(.2*torch.sin(torch.arange(flow.coeff.numel()).reshape_as(flow.coeff)))
    theta=torch.tensor([[1.2,-.3],[-3.1,2.4],[math.pi-1e-6,0.]])
    y,ld=flow.map(theta)
    back,ild=flow.map(y,True)
    difference=torch.atan2((back-theta).sin(),(back-theta).cos())
    torch.testing.assert_close(difference,torch.zeros_like(theta),atol=1e-12,rtol=0)
    torch.testing.assert_close(ld,-ild,atol=1e-12,rtol=0)
    matrices=torch.vmap(torch.func.jacrev(lambda z:flow.map(z)[0]))(theta)
    sign,exact=torch.linalg.slogdet(matrices)
    assert (sign>0).all()
    torch.testing.assert_close(ld,exact,atol=2e-12,rtol=1e-12)
    scalar=PeriodicMap(1)
    torch.testing.assert_close(scalar.map(torch.tensor([[.2]]))[0],torch.tensor([[.2]]))


def test_native_bd_hd_formulas_and_gradients():
    wf=torch.tensor([-2.,.1,1.3,4.5,7.],requires_grad=True)
    wr=torch.tensor([.2,-3.,4.1,-.4,1.5],requires_grad=True)
    free=.37
    for kind in ("bd","hd"):
        if kind=="bd":
            expected=-.5*(torch.exp(.5*(free-wf)).mean().log()+torch.exp(.5*(-wr-free)).mean().log())
        else:
            expected=-.5*((2*torch.sigmoid(free-wf)).mean().log()+(2*torch.sigmoid(-wr-free)).mean().log())
        observed=distance_loss(kind,wf,wr,free)
        torch.testing.assert_close(observed,expected,atol=1e-14,rtol=0)
        a=torch.autograd.grad(observed,(wf,wr),retain_graph=True)
        b=torch.autograd.grad(expected,(wf,wr),retain_graph=True)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,atol=1e-14,rtol=0)


def test_periodic_native_bar_and_region_bridge_known_normalizers():
    rng=np.random.default_rng(777)
    source=torch.tensor(rng.vonmises(0,.4,(20000,2)))
    target=torch.tensor(rng.vonmises([.7,-.4],1.3,(20000,2)))
    flow=PeriodicMap(2)
    with torch.no_grad():flow.coeff.copy_(.03*torch.sin(torch.arange(flow.coeff.numel()).reshape_as(flow.coeff)))
    forward,ld=flow.map(source);reverse,ild=flow.map(target,True)
    u0=lambda x:-.4*x.cos().sum(1)
    u1=lambda x:-1.3*(x-x.new_tensor([.7,-.4])).cos().sum(1)+.3
    wf=u1(forward)-u0(source)-ld
    wr=u0(reverse)-u1(target)-ild
    solved=bar(wf.detach().numpy(),wr.detach().numpy())
    expected=2*math.log(i0(1.3)/i0(.4))-.3
    assert solved["log_partition_ratio"]==pytest.approx(expected,abs=.025)
    classify=lambda z:np.stack([np.cos(z[:,0])>.8,np.cos(z[:,0])<-.8],1)
    readout=bridge_readout(torch.stack([source,forward],1).detach().numpy(),torch.stack([target,reverse],1).detach().numpy(),-wf.detach().numpy(),-wr.detach().numpy(),solved["log_partition_ratio"],classify)
    from scipy.integrate import quad
    expected_p=quad(lambda x:math.exp(1.3*math.cos(x-.7))/(2*math.pi*i0(1.3)),-math.acos(.8),math.acos(.8))[0]
    assert readout["endpoint_region_probabilities"][1][0]==pytest.approx(expected_p,abs=.012)


@pytest.mark.parametrize("name",["ALDP","ALA4"])
def test_mnf_molecular_full_chart_volume_inverse_and_loss_gradient(name):
    system=MolecularSystem(name)
    flow=MNFIC(system,MNFConfig(iterations=2,batch_size=1))
    source=system.initial_coordinates()[None]
    with torch.no_grad():flow.coeff.copy_(.03*torch.sin(torch.arange(flow.coeff.numel()).reshape_as(flow.coeff)))
    target,ld=flow.map(source)
    back,ild=flow.map(target,True)
    torch.testing.assert_close(back,source,atol=2e-12,rtol=0)
    torch.testing.assert_close(ld,-ild,atol=2e-11,rtol=0)
    assert system.chart.constraint_error(system.positions(target))<1e-12

    old_angles=dihedral(system,source)
    _,circle_ld=flow.circle(old_angles)
    physical_ld=ld+system.log_volume(target)-system.log_volume(source)
    torch.testing.assert_close(physical_ld,circle_ld,atol=5e-10,rtol=1e-10)
    wf,wr,*_=flow.works(source,source)
    loss=distance_loss("hd",wf,wr,0.)
    gradient=torch.autograd.grad(loss,flow.coeff)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0


def dihedral(system,z):
    from varipath.coordinates import dihedral_radians
    return dihedral_radians(system.positions(z)[:,torch.tensor(system.torsion_quads)])


def test_periodic_mnf_complete_bd_bar_hd_resume(tmp_path):
    from varipath.baselines.mnf import PeriodicMNF

    class Torus:
        dim=2
        name="periodic"
        def effective_u(self,z,endpoint):
            return -.4*z.cos().sum() if endpoint==0 else -1.3*(z-z.new_tensor([.7,-.4])).cos().sum()+.3
        def regions(self,z):
            return {"a":z[...,0].cos()>.8,"b":z[...,0].cos()<-.8}

    rng=np.random.default_rng(101)
    source=torch.tensor(rng.vonmises(0,.4,(24,2)))
    target=torch.tensor(rng.vonmises([.7,-.4],1.3,(24,2)))
    config=MNFConfig(iterations=4,batch_size=4,checkpoint_every=1)
    straight=PeriodicMNF(Torus(),config)
    complete=straight.fit(source,target)
    assert complete["complete"] and {row["phase"] for row in complete["history"]}=={"bd","hd"}
    checkpoint=tmp_path/"mnf.pt"
    PeriodicMNF(Torus(),config).fit(source,target,checkpoint,stop_after=1)
    resumed=PeriodicMNF(Torus(),config)
    resumed.fit(source,target,checkpoint,resume=True)
    torch.testing.assert_close(straight.coeff,resumed.coeff,atol=0,rtol=0)
    assert resumed.evaluate(source,target)["native_bar"]["solver_log_equation_residual"]<1e-10
    with pytest.raises(ValueError,match="训练池"):
        resumed.fit(source+.001,target,checkpoint,resume=True)


def test_feat_zero_endpoint_support_is_retained_not_projected():
    class PositiveHalfLine:
        dim=1
        name="halfline"
        def positions(self,z):return z
        def in_domain(self,z):return z[...,0]>0
        def effective_u(self,z,endpoint):
            return torch.where(z[0]>0,.5*z.square().sum(),z.new_tensor(float("inf")))

    model=FEAT(PositiveHalfLine(),FEATConfig(iterations=1,batch_size=4,path_steps=2))
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
    path,ratio=model.sample_paths(torch.full((128,1),.01),generator=torch.Generator().manual_seed(14))
    outside=path[:,-1,0]<=0
    assert outside.any() and (~outside).any()
    assert torch.isneginf(ratio[outside]).all()
    assert torch.isfinite(ratio[~outside]).all()
    assert (path[outside,-1,0]<=0).all()


def test_native_budget_callbacks_propagate_before_partial_output(tmp_path):
    from varipath.baselines.mnf import PeriodicMNF

    class Deadline(RuntimeError):pass
    def expired():raise Deadline("budget exhausted")
    source=torch.zeros((4,2))
    feat=FEAT(GaussianEndpoints(),FEATConfig(iterations=1,batch_size=4,path_steps=2))
    with pytest.raises(Deadline):
        feat.fit(source,source,tmp_path/"expired-feat.pt",check_budget=expired)
    assert not (tmp_path/"expired-feat.pt").exists()
    with pytest.raises(Deadline):
        feat.sample_paths(source,check_budget=expired)
    class Torus:
        dim=2
        name="periodic"
        def effective_u(self,z,endpoint):return -.4*z.cos().sum()
        def regions(self,z):return {"a":z[...,0]>0,"b":z[...,0]<=0}
    mnf=PeriodicMNF(Torus(),MNFConfig(iterations=2,batch_size=4))
    with pytest.raises(Deadline):
        mnf.fit(source,source,tmp_path/"expired-mnf.pt",check_budget=expired)
    assert not (tmp_path/"expired-mnf.pt").exists()
    with pytest.raises(Deadline):
        mnf.evaluate(source,source,check_budget=expired)


def test_chunked_native_estimators_solve_one_global_bridge():
    from varipath.baselines.mnf import PeriodicMNF
    from varipath.benchmarks import PeriodicSystem
    periodic=PeriodicSystem()
    mnf=PeriodicMNF(periodic,MNFConfig(iterations=2,batch_size=4))
    with torch.no_grad():mnf.coeff.copy_(.03*torch.sin(torch.arange(mnf.coeff.numel()).reshape_as(mnf.coeff)))
    source=periodic.sample(71,0,17);target=periodic.sample(83,1,18)
    full=mnf.works(source,target)
    chunk=mnf.batched_works(source,target,batch_size=7)
    for actual,expected in zip(chunk,full):
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=2e-12)
    result=mnf.evaluate(source,target,batch_size=7)
    expected=bar(full[0].detach().numpy(),full[1].detach().numpy())
    assert result["native_bar"]["free_energy"]==pytest.approx(expected["free_energy"],abs=1e-12)
    feat=FEAT(GaussianEndpoints(),FEATConfig(iterations=1,batch_size=4,path_steps=3))
    with torch.no_grad():
        for parameter in feat.parameters():parameter.zero_()
    generator=torch.Generator().manual_seed(224)
    source=torch.randn((37,2),generator=generator)
    target=.3+1.1*torch.randn((41,2),generator=generator)
    persisted=[]
    result=feat.evaluate(source,target,generator,batch_size=7,path_sink=lambda forward,start,path,ratio:persisted.append((forward,start,path.shape[1])))
    assert result["path_storage"]=="endpoints_only_after_actual_kernel_replay"
    assert result["forward_path"].shape==(37,2,2)
    assert result["reverse_path"].shape==(41,2,2)
    assert len(persisted)==12 and all(nodes==4 for _,_,nodes in persisted)
    expected=nebar(result["forward_log_ratio"].numpy(),result["reverse_log_ratio"].numpy())
    assert result["native_nebar"]["log_partition_ratio"]==pytest.approx(expected["log_partition_ratio"],abs=1e-14)
