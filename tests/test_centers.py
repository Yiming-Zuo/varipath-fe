import json
import shutil

import numpy as np
import pytest
import torch

from varipath.centers import DiskCenters,freeze_centers
from varipath.protection import RadialProtection
from varipath.transport import Transport
from varipath.training import TrainingConfig,train_transport


def fixture():
    pretrain=Transport(2,RadialProtection([0.,0.],2.,10.),spline_count=4,steps=64)
    with torch.no_grad():
        pretrain.C.copy_(.04*torch.sin(torch.arange(pretrain.C.numel()).reshape_as(pretrain.C)))
    initial=torch.tensor([[.2,.1],[-.4,.3]],dtype=torch.float64)
    return pretrain,initial


def moving_model(pretrain,initial,trajectory):
    model=Transport(2,pretrain.protection,spline_count=4,steps=64,centers=initial,
                    bandwidths=[1.,1.],amplitudes=[.4,.4],time_centers=trajectory)
    with torch.no_grad():
        model.C.copy_(pretrain.C)
        model.alpha.fill_(.3)
    return model


def test_lazy_centers_match_tensor_maps_all_gradients_and_higher_response(tmp_path):
    pretrain,initial=fixture()
    provider=freeze_centers(pretrain,initial,tmp_path/"moving",run_root=tmp_path,chunk_steps=8)
    reference=[]
    point=initial.clone()
    with torch.no_grad():
        for k in range(pretrain.n_steps):
            reference.append(point.clone())
            point=pretrain.step(point,k)[0]
    timeline=torch.stack(reference)
    loaded=DiskCenters.from_descriptor(provider.descriptor(),tmp_path,expected_source_state=provider.source_state(),cache_steps=3)
    for k in (0,1,7,8,31,63):
        torch.testing.assert_close(loaded.at(k),timeline[k],atol=0,rtol=0)
        assert len(loaded._cache)<=3
    direct=moving_model(pretrain,initial,timeline)
    lazy=moving_model(pretrain,initial,loaded)
    assert lazy.time_centers is None
    assert "time_centers" not in lazy.state_dict()
    points=torch.tensor([[.2,.3],[3.,.1],[-.7,.4]],dtype=torch.float64)
    u0=lambda z:.5*z.square().sum()
    u1=lambda z:.5*(z-.2).square().sum()
    direct_loss=direct.objective(points,u0,u1)
    lazy_loss=lazy.objective(points,u0,u1,checkpoint_steps=8)
    torch.testing.assert_close(lazy_loss,direct_loss,atol=2e-13,rtol=2e-13)
    direct_grad=torch.autograd.grad(direct_loss,list(direct.parameters()))
    lazy_grad=torch.autograd.grad(lazy_loss,list(lazy.parameters()))
    for actual,expected in zip(lazy_grad,direct_grad):
        torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-12)
    target=points[0]
    direct_jac=torch.func.jacrev(lambda z:direct(z[None],check=False)[0][0])(target)
    lazy_jac=torch.func.jacrev(lambda z:lazy(z[None],check=False)[0][0])(target)
    torch.testing.assert_close(lazy_jac,direct_jac,atol=0,rtol=0)
    with pytest.raises(TypeError,match="descriptor"):
        torch.save(provider,tmp_path/"not-a-model.pt")


def test_generation_recovers_without_repeating_committed_prefix(tmp_path,monkeypatch):
    pretrain,initial=fixture()
    calls=[]
    original=pretrain.step
    def record(points,k,**kwargs):
        calls.append(k)
        return original(points,k,**kwargs)
    monkeypatch.setattr(pretrain,"step",record)
    def stop():
        if len(calls)>=8:
            raise TimeoutError("stop after first committed center chunk")
    folder=tmp_path/"moving"
    with pytest.raises(TimeoutError):
        freeze_centers(pretrain,initial,folder,chunk_steps=8,check_budget=stop)
    assert calls==list(range(8))
    with pytest.raises(ValueError,match="完整生成"):
        DiskCenters(folder)
    provider=freeze_centers(pretrain,initial,folder,chunk_steps=8,resume=True)
    assert calls==list(range(64))
    freeze_centers(pretrain,initial,folder,chunk_steps=8,resume=True)
    assert calls==list(range(64))
    with torch.no_grad():
        pretrain.C[0,0]+=.001
    with pytest.raises(ValueError,match="源预训练模型"):
        freeze_centers(pretrain,initial,folder,chunk_steps=8,resume=True)
    assert provider.shape==(64,2,2)


def test_relative_descriptor_and_training_resume_survive_run_move(tmp_path):
    pretrain,initial=fixture()
    run=tmp_path/"original-run"
    run.mkdir()
    provider=freeze_centers(pretrain,initial,run/"moving",run_root=run,chunk_steps=8)
    descriptor=provider.descriptor()
    assert descriptor["directory"]=="moving"
    assert str(tmp_path) not in json.dumps(descriptor)
    model=moving_model(pretrain,initial,provider)
    points=torch.tensor([[.2,.4],[.6,-.2],[-.4,.1]],dtype=torch.float64)
    u0=lambda z:.5*z.square().sum()
    u1=lambda z:.5*(z-.2).square().sum()
    cfg=dict(batch_size=2,checkpoint_steps=16)
    train_transport(model,points,points+.03,u0,u1,TrainingConfig(max_steps=1,**cfg),run/"training.pt")
    saved=torch.load(run/"training.pt",weights_only=True)
    assert saved["center_provider"]==descriptor
    destination=tmp_path/"moved-run"
    shutil.move(run,destination)
    moved=DiskCenters.from_descriptor(saved["center_provider"],destination,expected_source_state=saved["center_provider_source_state"])
    resumed=moving_model(pretrain,initial,moved)
    report=train_transport(resumed,points,points+.03,u0,u1,TrainingConfig(max_steps=2,**cfg),destination/"training.pt",resume=True)
    uninterrupted=moving_model(pretrain,initial,moved)
    expected=train_transport(uninterrupted,points,points+.03,u0,u1,TrainingConfig(max_steps=2,**cfg))
    assert report["history"]==expected["history"]
    for actual,wanted in zip(resumed.parameters(),uninterrupted.parameters()):
        torch.testing.assert_close(actual,wanted,atol=0,rtol=0)


def test_missing_changed_metadata_shape_and_source_fail_explicitly(tmp_path):
    pretrain,initial=fixture()
    provider=freeze_centers(pretrain,initial,tmp_path/"moving",run_root=tmp_path,chunk_steps=8)
    descriptor=provider.descriptor()
    changed=dict(descriptor,metadata=dict(descriptor["metadata"],spline_count=50))
    with pytest.raises(ValueError,match="元数据"):
        DiskCenters.from_descriptor(changed,tmp_path)
    expected_state=provider.source_state()
    progress=torch.load(provider.folder/"progress.pt",weights_only=True)
    progress["source_state"]["C"][0,0]+=.001
    torch.save(progress,provider.folder/"progress.pt")
    with pytest.raises(ValueError,match="源预训练模型参数"):
        DiskCenters.from_descriptor(descriptor,tmp_path,expected_source_state=expected_state)
    with pytest.raises(ValueError,match="源预训练模型"):
        provider.validate()
    np.save(provider.path,np.zeros((1,2,2),dtype=np.float64))
    with pytest.raises(ValueError,match="shape"):
        DiskCenters.from_descriptor(descriptor,tmp_path)
    provider.path.unlink()
    with pytest.raises(FileNotFoundError):
        DiskCenters.from_descriptor(descriptor,tmp_path)
