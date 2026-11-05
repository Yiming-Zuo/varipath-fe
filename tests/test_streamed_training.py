import torch
import pytest

from varipath.protection import RadialProtection
from varipath.transport import Transport
from varipath.training import streamed_objective,TrainingConfig,train_transport


def fixture():
    model=Transport(2,RadialProtection([0.,0.],2.,8.),spline_count=4,steps=80,
                    centers=[[.3,-.2]],bandwidths=[.8],amplitudes=[.4])
    with torch.no_grad():
        model.C.copy_(.08*torch.sin(torch.arange(model.C.numel()).reshape_as(model.C)))
        model.alpha.fill_(.25)
    points=torch.tensor([[.2,.4],[3.,.2],[.4,2.7],[-.5,.2],[4.,-.7]],dtype=torch.float64)
    u0=lambda z:.5*z.square().sum()
    u1=lambda z:.5*((z-.2)/1.1).square().sum()+.3
    return model,points,u0,u1


def test_checkpoint_step_exact_map_logdet_and_all_parameter_gradients():
    model,points,u0,u1=fixture()
    ordinary=model.log_weights(points,u0,u1)
    value=model.objective(points,u0,u1)
    derivatives=torch.autograd.grad(value,list(model.parameters()))
    recomputed=model.log_weights(points,u0,u1,checkpoint_steps=13)
    chunked=model.objective(points,u0,u1,checkpoint_steps=13)
    chunked_derivatives=torch.autograd.grad(chunked,list(model.parameters()))
    for actual,expected in zip(recomputed,ordinary):
        torch.testing.assert_close(actual,expected,atol=2e-13,rtol=2e-13)
    torch.testing.assert_close(chunked,value,atol=2e-13,rtol=2e-13)
    for actual,expected in zip(chunked_derivatives,derivatives):
        torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-12)
    for step in (0,1,model.n_steps//2,model.n_steps-1):
        direct=model.step(points,step)
        analytical=model._checkpoint_step(points,step,(model.U,model.V,model.W))
        for actual,expected in zip(analytical,direct):
            torch.testing.assert_close(actual,expected,atol=3e-14,rtol=3e-14)


def test_two_pass_objective_uses_global_softmax_across_all_uneven_batches():
    model,points,u0,u1=fixture()
    expected=model.objective(points,u0,u1)
    gradient=torch.cat([g.flatten() for g in torch.autograd.grad(expected,list(model.parameters()))])
    value,actual=streamed_objective(model,points,u0,u1,batch_size=2,checkpoint_steps=17,gradient=True)
    assert value==pytest.approx(float(expected.detach()),abs=3e-13,rel=3e-13)
    torch.testing.assert_close(actual,gradient,atol=2e-12,rtol=2e-12)
    value_only=streamed_objective(model,points,u0,u1,batch_size=3,checkpoint_steps=17)
    assert value_only==pytest.approx(value,abs=3e-13,rel=3e-13)


def test_budget_checks_run_during_recomputed_step_chunks():
    model,points,u0,u1=fixture()
    checks=[]
    result=model.objective(points,u0,u1,checkpoint_steps=13,check_budget=lambda:checks.append(1))
    forward_count=len(checks)
    torch.autograd.grad(result,list(model.parameters()))
    assert forward_count>=7
    assert len(checks)>forward_count
    with pytest.raises(ValueError,match="全部节点"):
        model(points,checkpoint_steps=16,return_nodes=True)


def test_streamed_training_resume_preserves_whole_pool_path(tmp_path):
    model,points,u0,u1=fixture()
    path=tmp_path/"training.pt"
    cfg=dict(batch_size=2,checkpoint_steps=17)
    train_transport(model,points,points+.05,u0,u1,TrainingConfig(max_steps=1,**cfg),path)
    resumed,*_=fixture()
    actual=train_transport(resumed,points,points+.05,u0,u1,TrainingConfig(max_steps=2,**cfg),path,resume=True)
    direct,*_=fixture()
    expected=train_transport(direct,points,points+.05,u0,u1,TrainingConfig(max_steps=2,**cfg))
    assert actual["history"]==expected["history"]
    for a,b in zip(resumed.parameters(),direct.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_real_molecular_guard_transition_input_and_parameter_gradients():
    from varipath.physics import MolecularSystem,AffineSystem
    from varipath.geometry import fit_protection
    raw=MolecularSystem("ALA4")
    pool=torch.stack([raw.initial_coordinates(i) for i in range(4)])
    system=AffineSystem(raw,pool.mean(0),pool.std(0).clamp_min(.01))
    pool=(pool-system.location)/system.scale
    guard=fit_protection(system,pool)
    outside=pool[0].clone()
    outside[0]=guard.upper[0]+1
    lower,upper=0.,1.
    for _ in range(55):
        mid=(lower+upper)/2
        q=pool[0]+mid*(outside-pool[0])
        if guard(q)>.5:
            lower=mid
        else:
            upper=mid
    point=(pool[0]+.5*(lower+upper)*(outside-pool[0]))[None].detach().requires_grad_(True)
    assert .49<float(guard(point[0]).detach())<.51
    model=Transport(system.dim,guard,spline_count=4)
    with torch.no_grad():
        model.C.fill_(.04)
    params=list(model.parameters())
    direct=model.step(point,model.n_steps//2)
    exact=torch.autograd.grad(direct[0].square().sum()+direct[1].sum(),params+[point],allow_unused=True)
    alternate=model._checkpoint_step(point,model.n_steps//2,(model.U,model.V,model.W))
    actual=torch.autograd.grad(alternate[0].square().sum()+alternate[1].sum(),params+[point],allow_unused=True)
    for a,b in zip(actual,exact):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a,b,atol=2e-11,rtol=2e-9)
