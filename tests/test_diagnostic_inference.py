import math
from pathlib import Path

import numpy as np
import pytest
import torch

from varipath.diagnostic_inference import DIAGNOSTIC_EVENTS, _reference_moments, summarize_e2
from varipath.inference import RoleSeal
from varipath.physics import MolecularSystem
from varipath.readouts import R_KJ
from varipath.records import write_json


class TwoRegionControl:
    name, dim = "ALA4", 1
    def __init__(self):self.calls=0
    def state(self, endpoint):return float(endpoint)
    def in_domain(self, z):return torch.isfinite(z).all(-1)
    def regions(self, z):
        return dict(a=z[...,0]>0,b=z[...,0]<=0,valid=torch.isfinite(z).all(-1))
    def polar_energy(self, z):
        self.calls+=1
        return (z[...,0]<=0).double()*(R_KJ*300*math.log(2))


def bank(count=60, target_counts=None):
    rng=np.random.default_rng(1024)
    target_counts=target_counts or [int(2*count/3)+i for i in (-3,-1,1,3)]
    positions=[]
    for ladder in range(4):
        states=[]
        for number in (count//2, int(.6*count),target_counts[ladder]):
            points=np.r_[1+rng.random(number),-1-rng.random(count-number)]
            rng.shuffle(points);states.append(points[:,None])
        positions.append(np.stack(states,1))
    points=torch.tensor(np.asarray(positions),dtype=torch.float64)
    cross=(points[...,0]<=0).double()[:,:,None,:]*torch.tensor([0.,.5,1.])[None,None,:,None]*math.log(2)
    return dict(role="R",mode="formal",source_unit_ids=[f"ALA4:R:control:ladder-{i}" for i in range(4)],
        positions=points,states=[0.,.5,1.],u_kn=cross,eligible_for_formal=True,qualification_result={"eligible_for_formal":True})


def inputs(tmp_path, repetitions=128, reference=None):
    reference=bank() if reference is None else reference
    report_paths=[];f_ids=[]
    for method in ("V","V0"):
        for replica in range(repetitions):
            source=tmp_path/"runs"/method/str(replica)
            report_path=tmp_path/"reports"/method/str(replica)/"diagnostic.json"
            ids={role:[f"F:{method}:{replica}:{role}"] for role in ("fallback","fit","select","production")}
            f_ids.extend(unit for values in ids.values() for unit in values)
            record=dict(status="completed",molecule="ALA4",method=method,replica=replica,
                eligible_for_formal=True,data_unit_ids=ids,native={"sample_count":100})
            write_json(source/"result.json",record)
            delta=.002*math.sin(replica)
            proposal=[.55+delta,.25-delta,.2] if method=="V" else [.2+delta,.4-delta,.4]
            report=dict(status="diagnostic",molecule="ALA4",method=method,replica=replica,
                reused_production=str(source),data_unit_ids=ids,main_partition={"proposal":proposal})
            if method=="V":
                dv,dv0=-.05+.001*math.sin(replica),.001*math.cos(replica)
                report["frozen_2x2"]=dict(status="diagnostic",d_V=dv,d_V0=dv0,interaction=dv-dv0,
                    moments={"V_on":dv,"V_off":0.,"V0_on":dv0,"V0_off":0.})
            write_json(report_path,report);report_paths.append(report_path)
    seal=RoleSeal(tmp_path/"roles.json")
    for role,ids in (("P",["P-control"]),("I",["I-control"]),("F",f_ids)):
        seal.open(role,ids)
        artifact=tmp_path/f"{role}.json";write_json(artifact,{"scope":"test fixture"})
        seal.seal(role,[str(artifact)])
    seal.open("R",reference["source_unit_ids"])
    return report_paths,reference,seal


def test_full_proposal_reference_projection_closure_and_paired_cross(tmp_path):
    reports,reference,seal=inputs(tmp_path)
    system=TwoRegionControl()
    result=summarize_e2(reports,reference,system,seal,{"batch_size":32})
    assert result["diagnostic_interval_events"]==DIAGNOSTIC_EVENTS==31
    assert not result["independent_confirmation"] and result["formal_error_budget_unchanged"]
    assert result["mismatch"]["V"]["independent_deployments"]==128


    assert result["mismatch"]["V"]["mean_deployment_mismatch"]["full_deployment_gate"]=="unresolved"
    assert result["mismatch"]["V0"]["mismatch_of_mean_proposal"]["lower"]>.005
    assert result["frozen_2x2"]["valid_paired_deployments"]==128
    assert result["frozen_2x2"]["beneficial_and_route_specific"]=="supported"
    closure=result["polar_closure"]
    assert closure["integral"]["point"]==pytest.approx(math.log(2),abs=1e-14)
    assert closure["endpoint_preference"]["point"]==pytest.approx(math.log(2),abs=1e-14)
    assert abs(closure["finite_grid_difference"]["point"])<1e-14
    assert closure["finite_grid_difference"]["interval"]["lower"]<=0<=closure["finite_grid_difference"]["interval"]["upper"]
    assert closure["continuous_difference_interval"]["status"]=="undefined"
    assert system.calls==4*3*math.ceil(reference["positions"].shape[1]/32)


def test_single_deployment_and_missing_four_maps_do_not_make_a_128_repeat_ci(tmp_path):
    reports,reference,seal=inputs(tmp_path,repetitions=1)
    result=summarize_e2(reports,reference,TwoRegionControl(),seal)
    summary=result["mismatch"]["V"]
    assert summary["independent_deployments"]==1 and len(summary["missing_replicas"])==127
    assert summary["mean_deployment_mismatch"]["interval"]["status"]=="undefined"
    assert summary["mean_deployment_mismatch"]["full_deployment_gate"]=="unresolved"
    assert result["frozen_2x2"]["joint_intervals"]["interaction"]["status"]=="undefined"


def test_global_raw_conditional_moments_and_odds_are_not_averaged_ladder_ratios(tmp_path):
    reference=bank(count=10,target_counts=[6,7,8,9])
    reference["positions"][0,:,1,0]=-reference["positions"][0,:,1,0].abs()
    reference["u_kn"]=(reference["positions"][...,0]<=0).double()[:,:,None,:]*torch.tensor([0.,.5,1.])[None,None,:,None]*math.log(2)
    reports,reference,seal=inputs(tmp_path,repetitions=2,reference=reference)
    result=summarize_e2(reports,reference,TwoRegionControl(),seal)
    closure=result["polar_closure"]
    assert closure["status"]=="diagnostic"
    assert closure["endpoint_preference"]["point"]==pytest.approx(math.log(3),abs=1e-14)
    assert closure["integral"]["point"]==pytest.approx(math.log(2),abs=1e-14)
    assert closure["finite_grid_difference"]["point"]==pytest.approx(math.log(2/3),abs=1e-14)
    old=np.mean([math.log(p/(1-p)) for p in (.6,.7,.8,.9)])
    assert abs(old-closure["endpoint_preference"]["point"])>.1


def test_R_reveal_independence_duplicates_and_zero_global_core(tmp_path):
    with pytest.raises(ValueError,match="揭盲"):
        summarize_e2([],tmp_path/"unreadable-R.pt",TwoRegionControl(),None)
    reports,reference,seal=inputs(tmp_path,repetitions=2)
    with pytest.raises(ValueError,match="重复读取"):
        summarize_e2([reports[0],reports[0]],reference,TwoRegionControl(),seal)
    copied=dict(reference,source_unit_ids=[seal.state["roles"]["F"]["unit_ids"][0],*reference["source_unit_ids"][1:]])
    with pytest.raises(ValueError,match="泄漏"):
        summarize_e2(reports,copied,TwoRegionControl(),seal)
    reference["positions"][:,:,2,0]=-reference["positions"][:,:,2,0].abs()
    reference["u_kn"]=(reference["positions"][...,0]<=0).double()[:,:,None,:]*torch.tensor([0.,.5,1.])[None,None,:,None]*math.log(2)
    result=summarize_e2(reports,reference,TwoRegionControl(),seal)
    assert result["polar_closure"]["status"]=="unestimable"
    assert "地板" in result["polar_closure"]["reason"]


def test_actual_molecular_polar_operator_runs_in_bounded_batches():
    system=MolecularSystem("ALA4")
    initial=torch.stack([system.initial_coordinates(i) for i in range(4)])
    points=initial[:,None,None,:].expand(-1,2,2,-1).clone()
    reference=dict(positions=points,source_unit_ids=[f"R-native-{i}" for i in range(4)],states=[0.,1.],
        u_kn=torch.zeros(4,2,2,2))
    probabilities,conditional,states,endpoints=_reference_moments(reference,system,{"batch_size":1})
    assert probabilities.shape==(4,2,3) and conditional.shape==(4,2,4)
    assert np.isfinite(conditional).all() and endpoints==[0,1]
