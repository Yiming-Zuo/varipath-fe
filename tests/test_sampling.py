import pytest
import torch

from varipath.physics import MolecularSystem
from varipath.sampling import HMCSampler, ReplicaExchangeSampler, run_sampling


class _HalfGaussian:
    name="ALDP"
    dim=1

    def initial_coordinates(self):
        return torch.ones(1,dtype=torch.float64)

    def unrestricted_u_state(self,z,state):
        return .5*z.square().sum()

    def in_domain(self,z):
        return (z>0).all()


def test_hmc_samples_known_hard_domain_measure():
    sampler=HMCSampler(_HalfGaussian(),1.,19,step_size=.2,leapfrog_steps=7,coordinate_scale=[1.])
    values=sampler.sample(4096,warmup=100).flatten()
    assert values.min()>0
    assert abs(float(values.mean())-(2/torch.pi)**.5)<.05
    assert abs(float(values.var())-(1-2/torch.pi))<.05
    assert sampler.domain_rejections>0
    assert sampler.numerical_rejections==0


def test_hmc_resume_replays_whole_batch(tmp_path):
    system=MolecularSystem("ALDP")
    sampler=HMCSampler(system,550.,19,leapfrog_steps=2)
    sampler.sample(2)
    sampler.save(tmp_path/"hmc.pt")
    continued=sampler.sample(3)
    restored=HMCSampler.load(system,tmp_path/"hmc.pt")
    torch.testing.assert_close(restored.sample(3),continued,atol=0,rtol=0)
    assert restored.accepted==sampler.accepted
    assert system.in_domain(continued).all()
    assert restored.numerical_rejections==0


def test_exchange_cross_energies_and_resume(tmp_path):
    system=MolecularSystem("ALA4")
    sampler=ReplicaExchangeSampler(system,[0.,.5,1.],123,leapfrog_steps=1)
    sampler.step()
    sampler.save(tmp_path/"exchange.pt")
    expected=sampler.step()
    restored=ReplicaExchangeSampler.load(system,tmp_path/"exchange.pt")
    actual=restored.step()
    torch.testing.assert_close(actual["positions"],expected["positions"],atol=0,rtol=0)
    torch.testing.assert_close(actual["rotations"],expected["rotations"],atol=0,rtol=0)
    torch.testing.assert_close(actual["u_kn"],expected["u_kn"],atol=0,rtol=0)
    for i,state in enumerate(restored.states):
        for j,z in enumerate(actual["positions"]):
            torch.testing.assert_close(actual["u_kn"][i,j],system.effective_u_state(z,state))
    assert sorted(actual["walkers"])==[0,1,2]


def test_failed_inputs_and_changed_checkpoint_are_rejected(tmp_path):
    system=MolecularSystem("ALDP")
    with pytest.raises(ValueError):
        HMCSampler(system,-1.,123)
    with pytest.raises(ValueError):
        ReplicaExchangeSampler(system,[550.,300.],123)
    bad=system.initial_coordinates()*float("nan")
    with pytest.raises(ValueError):
        HMCSampler(system,550.,123,initial=bad)
    sampler=HMCSampler(system,550.,123)
    data=sampler.state_dict()
    data["state"]=300.
    with pytest.raises(ValueError,match="恢复配置"):
        sampler.load_state_dict(data)


def test_whole_ladder_bank_and_resume_keeps_smoke_unqualified(tmp_path):
    config=dict(count=2,warmup=1,ladders=2,seed=17,states=[300.,550.],mode="smoke",role="D",leapfrog_steps=1,checkpoint_every=1)
    system=MolecularSystem("ALDP")
    first=run_sampling(system,config,tmp_path)
    assert first["positions"].shape==(2,2,2,48)
    assert first["rotations"].shape==(2,2,2,3,3)
    rotation=first["rotations"]
    torch.testing.assert_close(rotation@rotation.transpose(-1,-2),torch.eye(3,dtype=torch.float64).expand_as(rotation),atol=2e-15,rtol=0)
    torch.testing.assert_close(torch.linalg.det(rotation),torch.ones(rotation.shape[:-2],dtype=torch.float64),atol=2e-15,rtol=0)
    assert first["eligible_for_formal"] is False
    second=run_sampling(system,config,tmp_path,resume=True)
    torch.testing.assert_close(second["positions"],first["positions"],atol=0,rtol=0)
    torch.testing.assert_close(second["rotations"],first["rotations"],atol=0,rtol=0)
    assert first["source_unit_ids"][0]!=first["source_unit_ids"][1]
    assert first["analysis_block_ids"][0][0]!=first["analysis_block_ids"][1][0]
    with pytest.raises(ValueError,match="原采样"):
        run_sampling(system,dict(config,count=3),tmp_path,resume=True)


def test_R_extension_keeps_exact_prefix_rng_and_single_warmup(tmp_path):
    config=dict(count=2,warmup=2,ladders=2,seed=117,states=[300.,550.],mode="development",role="R",leapfrog_steps=1,checkpoint_every=1)
    system=MolecularSystem("ALDP")
    first=run_sampling(system,config,tmp_path/"sequential")
    extended=run_sampling(system,dict(config,count=4),tmp_path/"sequential",resume=True,extend=True)
    direct=run_sampling(system,dict(config,count=4),tmp_path/"direct")
    for key in ("positions","u_kn","rotations"):
        torch.testing.assert_close(extended[key][:,:2],first[key],atol=0,rtol=0)
        torch.testing.assert_close(extended[key],direct[key],atol=0,rtol=0)
    assert (extended["walkers"]==direct["walkers"]).all()
    assert extended["source_unit_ids"]==first["source_unit_ids"]
    assert extended["elapsed_seconds"]>=first["elapsed_seconds"]
    assert extended["reference_base_count"]==2
    for ladder in range(2):
        actual=torch.load(tmp_path/"sequential"/f"ladder-{ladder}.pt",weights_only=False)["sampler"]
        expected=torch.load(tmp_path/"direct"/f"ladder-{ladder}.pt",weights_only=False)["sampler"]
        assert actual["iteration"]==config["warmup"]+4
        assert actual["numpy_rng"]==expected["numpy_rng"]
        for left,right in zip(actual["kernels"],expected["kernels"]):
            assert left["attempted"]==right["attempted"]
            assert left["numpy_rng"]==right["numpy_rng"]
            torch.testing.assert_close(left["torch_rng"],right["torch_rng"],atol=0,rtol=0)


def test_R_extension_resumes_partial_growth_without_new_initialization(tmp_path):
    config=dict(count=2,warmup=1,ladders=2,seed=118,states=[300.,550.],mode="development",role="R",leapfrog_steps=1,checkpoint_every=1)
    system=MolecularSystem("ALDP")
    first=run_sampling(system,config,tmp_path/"sequential")
    calls=0

    def interrupt():
        nonlocal calls
        calls+=1
        if calls==2:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_sampling(system,dict(config,count=4),tmp_path/"sequential",resume=True,extend=True,check_budget=interrupt)
    retained=torch.load(tmp_path/"sequential/bank.pt",weights_only=False)
    torch.testing.assert_close(retained["positions"],first["positions"],atol=0,rtol=0)
    resumed=run_sampling(system,dict(config,count=4),tmp_path/"sequential",resume=True)
    direct=run_sampling(system,dict(config,count=4),tmp_path/"direct")
    for key in ("positions","u_kn","rotations"):
        torch.testing.assert_close(resumed[key],direct[key],atol=0,rtol=0)


def test_extension_rejects_changed_kernel_other_roles_and_final_reference(tmp_path):
    import json
    config=dict(count=2,warmup=1,ladders=2,seed=119,states=[300.,550.],mode="development",role="R",leapfrog_steps=1,checkpoint_every=1)
    system=MolecularSystem("ALDP")
    run_sampling(system,config,tmp_path/"R")
    with pytest.raises(ValueError,match="仅允许增加"):
        run_sampling(system,dict(config,count=4,warmup=2),tmp_path/"R",resume=True,extend=True)
    (tmp_path/"R/reference-status.json").write_text(json.dumps({"final":True,"stopped":True}))
    with pytest.raises(ValueError,match="停止|final"):
        run_sampling(system,dict(config,count=4),tmp_path/"R",resume=True,extend=True)
    assert json.loads((tmp_path/"R/sampling-config.json").read_text())["count"]==2
    foreign=dict(config,role="F-production")
    run_sampling(system,foreign,tmp_path/"F")
    with pytest.raises(ValueError,match="R"):
        run_sampling(system,dict(foreign,count=4),tmp_path/"F",resume=True,extend=True)


def test_same_seed_and_role_have_disjoint_development_and_formal_streams(tmp_path):
    config=dict(count=2,warmup=1,ladders=2,seed=120,states=[300.,550.],role="F-production",leapfrog_steps=1,checkpoint_every=1)
    system=MolecularSystem("ALDP")
    development=run_sampling(system,dict(config,mode="development"),tmp_path/"D")
    formal=run_sampling(system,dict(config,mode="formal"),tmp_path/"F")
    assert development["random_stream_version"]==formal["random_stream_version"]==2
    assert development["role_seed"]!=formal["role_seed"]
    assert not set(development["source_unit_ids"])&set(formal["source_unit_ids"])
    assert not torch.equal(development["positions"],formal["positions"])
    assert not torch.equal(development["rotations"],formal["rotations"])


@pytest.mark.parametrize("key,value",[("count",3.7),("count",True),("warmup",1.2),("warmup",False),("ladders",2.5),("ladders",True)])
def test_fixed_batch_counts_never_truncate_floats_or_accept_bools(tmp_path,key,value):
    config=dict(count=2,warmup=1,ladders=2,seed=121,states=[300.,550.],mode="development",role="R",leapfrog_steps=1)
    config[key]=value
    with pytest.raises(ValueError,match="整数"):
        run_sampling(MolecularSystem("ALDP"),config,tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()


def test_old_stream_contract_is_rejected_without_restart_or_checkpoint_mutation(tmp_path):
    import json
    config=dict(count=2,warmup=1,ladders=2,seed=122,states=[300.,550.],mode="development",role="R",leapfrog_steps=1,checkpoint_every=1)
    system=MolecularSystem("ALDP")
    run_sampling(system,config,tmp_path)
    checkpoint=(tmp_path/"ladder-0.pt").read_bytes()
    settings=json.loads((tmp_path/"sampling-config.json").read_text())
    settings.pop("random_stream_version")
    (tmp_path/"sampling-config.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError,match="随机流版本不匹配"):
        run_sampling(system,config,tmp_path,resume=True)
    assert (tmp_path/"ladder-0.pt").read_bytes()==checkpoint
