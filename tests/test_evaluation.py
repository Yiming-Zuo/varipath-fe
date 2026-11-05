import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from varipath.evaluation import UNITS, aggregate_results, reference_summary
from varipath.inference import RoleSeal
from varipath.readouts import R_KJ
from varipath.records import METHODS


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def registry(tmp_path,f_ids,r_ids):
    seal=RoleSeal(tmp_path/"roles.json")
    for role,ids in (("P",["P-test"]),("I",["I-test"]),("F",f_ids)):
        seal.open(role,ids)
        artifact=tmp_path/f"{role}.json"
        write(artifact,{"status":"complete_test_fixture"})
        seal.seal(role,[str(artifact)])
    seal.open("R",r_ids)
    return seal


class AnalyticalReferenceSystem:
    name="ALDP"
    dim=1
    def state(self,endpoint):return 550. if endpoint==0 else 300.
    def in_domain(self,z):return torch.isfinite(z).all(-1)
    def regions(self,z):
        return {"a":z[...,0]>0,"b":z[...,0]<=0,"valid":torch.ones(z.shape[:-1],dtype=torch.bool)}
    def u(self,z,state):
        return .5*z[...,0]**2 if state==550 else .5*((z[...,0]-.2)/1.1)**2+.1


def reference_bank(count=96):
    system=AnalyticalReferenceSystem()
    rng=np.random.default_rng(775)
    positions=torch.tensor(np.stack([np.stack([.2+1.1*rng.normal(size=(count,1)),rng.normal(size=(count,1))],1) for _ in range(4)]),dtype=torch.float64)
    states=[300.,550.]
    cross=torch.stack([torch.stack([system.u(positions,state) for state in states],2)],0)[0]

    return dict(role="R",mode="formal",source_unit_ids=[f"R-real-fixture-{i}" for i in range(4)],
        positions=positions,u_kn=cross,states=states,eligible_for_formal=True,
        qualification_result={"eligible_for_formal":True,"protocol":{"protocol":{"kernel":"metropolis_hmc_replica_exchange","step_size":.05,"leapfrog_steps":8}}})


def numerical_certificate(tmp_path,bank):
    check=tmp_path/"independent-numerical-check.json"
    write(check,{"status":"passed","molecule":"ALDP","scope":"synthetic_io_contract_test_only", "source_unit_ids":["independent-bias-validation"],"measured_errors":{"delta_f":[0.],"preference":[0.],"target_a":[0.]}})
    certificate=tmp_path/"bias.json"
    write(certificate,dict(kind="reference_numerical_bias",status="passed",population_bound=True,
        molecule="ALDP",units=UNITS,bounds={"delta_f":1e-6,"preference":2e-6,"target_a":1e-7},
        source_unit_ids=["independent-bias-validation"],
        protocol=dict(states=bank["states"],kernel="metropolis_hmc_replica_exchange",step_size=.05,leapfrog_steps=8),
        derivation="Synthetic certificate exercises schema only; not a physical bias certification.",checks=[{"path":str(check)}]))
    return str(certificate)


def test_reference_requires_frozen_roles_and_real_bias_evidence(tmp_path):
    bank=reference_bank()

    unopened=reference_summary({"source_unit_ids":["opaque-R"]},AnalyticalReferenceSystem(),{})
    assert unopened["status"]=="unresolved" and not unopened["centers"]
    incomplete=RoleSeal(tmp_path/"incomplete.json")
    with pytest.raises(ValueError,match="先封存"):
        reference_summary(bank,AnalyticalReferenceSystem(),{},registry=incomplete)
    seal=registry(tmp_path,["F-test"],bank["source_unit_ids"])
    config={"reference":{"count":96}}
    missing=reference_summary(bank,AnalyticalReferenceSystem(),config,registry=seal)
    assert missing["status"]=="unresolved"
    assert missing["centers"]["delta_f"] is not None
    assert missing["intervals"]["delta_f"]["status"]=="undefined"
    config["reference"]["numerical_bias_certificate"]=numerical_certificate(tmp_path,bank)
    valid=reference_summary(bank,AnalyticalReferenceSystem(),config,registry=seal)
    assert valid["status"]=="ready" and valid["eligible_for_formal"]
    assert valid["interval_events"]==24
    assert len(valid["observations"])==4 and set(valid["centers"])=={"delta_f","preference","target_a"}
    assert valid["intervals"]["delta_f"]["lower"]<=.1-math.log(1.1)<=valid["intervals"]["delta_f"]["upper"]

    certificate=json.loads(Path(config["reference"]["numerical_bias_certificate"]).read_text())
    certificate["checks"]=[];certificate["verified"]=True
    write(Path(config["reference"]["numerical_bias_certificate"]),certificate)
    assert not reference_summary(bank,AnalyticalReferenceSystem(),config,registry=seal)["eligible_for_formal"]


def test_reference_zero_hit_is_undefined_without_floor(tmp_path):
    bank=reference_bank()
    bank["positions"]=bank["positions"].abs()+1.
    system=AnalyticalReferenceSystem()
    bank["u_kn"]=torch.stack([system.u(bank["positions"],state) for state in bank["states"]],2)
    seal=registry(tmp_path,["F-test"],bank["source_unit_ids"])
    config={"reference":{"count":96,"numerical_bias_certificate":numerical_certificate(tmp_path,bank)}}
    result=reference_summary(bank,system,config,registry=seal)
    assert result["status"]=="unresolved"
    assert result["centers"]["preference"] is None
    assert result["intervals"]["preference"]["status"]=="undefined"
    assert len(result["observations"])==4
    assert all(row[1] is None for row in result["observations"])
    assert any("零命中" in failure["reason"] for failure in result["ladder_failures"])


@pytest.fixture
def complete_runs(tmp_path):
    root=tmp_path/"runs"
    f_ids=[]
    for molecule in ("ALA4","ALDP"):
        for method in METHODS:
            for replica in range(128):
                error=.015*math.sin(replica) if method=="V" else .25+.02*math.sin(replica)
                units="kJ/mol" if molecule=="ALA4" else "dimensionless"
                scale=R_KJ*300 if molecule=="ALA4" else 1.
                readout=dict(delta_f=.5+error,delta_free_energy=scale*(.5+error),preference=scale*(.2+error),native_preference=scale*(.2+error),units=units,native_defined=True,floor_triggered=False)
                ids={role:([f"{molecule}-{method}-{replica}-{role}"] if role in ("fallback","production") or method not in ("E-MBAR","EB-MBAR") else []) for role in ("fallback","fit","select","production")}
                f_ids.extend(unit for role in ids.values() for unit in role)
                result=dict(method=method,replica=replica,molecule=molecule,mode="formal",status="completed",
                    eligible_for_formal=True,fallback_used=False,deployed=readout,native=readout,data_unit_ids=ids,
                    budget_seconds=100.,budget_ratio=1.,execution_unit={"cpu":"same-test-model","threads":1},
                    cost={"phases":[{"name":"paid-identity-fallback","status":"complete","wall_seconds":1.}]})
                write(root/molecule/method/str(replica)/"result.json",result)
    refs={name:dict(molecule=name,role="R",reference_id=f"reference-{name}",units=UNITS,source_unit_ids=[f"R-{name}"],
        eligible_for_formal=True,centers={"delta_f":.5,"preference":.2,"target_a":.4},
        intervals={key:{"lower":value-.001,"upper":value+.001,"status":"valid"} for key,value in (("delta_f",.5),("preference",.2),("target_a",.4))}) for name in ("ALA4","ALDP")}
    seal=registry(tmp_path,f_ids,["R-ALA4","R-ALDP"])
    config={"budget_seconds":100.,"budget_ratio":1.,"experiment":{"replicas":128,"methods":list(METHODS)}}
    return root,refs,config,seal


def test_aggregate_complete_2048_and_reference_once(complete_runs):
    root,refs,config,seal=complete_runs
    result=aggregate_results(root,refs,config,seal)
    assert result["observed_deployments"]==2048
    assert result["expected_deployments"]==2048
    assert result["comparison_count"]==28 and len(result["comparisons"])==28
    assert result["absolute_precision_count"]==4 and len(result["absolute_precision"])==4
    assert result["moment_interval_events"]==64
    assert result["status"]=="supported"
    baseline=result["absolute_precision"]["ALA4/preference/V"]["interval"]["upper"]
    refs["ALA4"]["intervals"]["preference"]={"lower":.05,"upper":.35,"status":"valid"}
    widened=aggregate_results(root,refs,config,seal)
    assert widened["absolute_precision"]["ALA4/preference/V"]["interval"]["upper"]>baseline+.01
    assert widened["absolute_precision"]["ALA4/preference/V"]["status"]=="unresolved"


def test_aggregate_missing_and_no_legal_output_not_deleted(complete_runs):
    root,refs,config,seal=complete_runs
    (root/"ALDP"/"V0"/"127"/"result.json").unlink()
    path=root/"ALA4"/"V"/"0"/"result.json"
    record=json.loads(path.read_text());record.update(status="no_legal_output",eligible_for_formal=False,deployed=None,native=None)
    write(path,record)
    result=aggregate_results(root,refs,config,seal)
    assert len(result["missing_deployments"])==1
    assert result["comparisons"]["ALDP/delta_f/V-vs-V0"]["status"]=="unresolved"
    assert result["absolute_precision"]["ALA4/delta_f/V"]["status"]=="unresolved"
    assert result["deployment_diagnostics"]["ALA4/delta_f"]["V"]["native_conditionally_defined_count"]==127
    assert result["deployment_diagnostics"]["ALA4/delta_f"]["V"]["deployed_point_mse"] is None


def test_aggregate_reference_leakage_and_units_rejected(complete_runs):
    root,refs,config,seal=complete_runs
    refs["ALA4"]["units"]={"delta_f":"kJ/mol","preference":"kJ/mol","target_a":"probability"}
    with pytest.raises(ValueError,match="单位"):
        aggregate_results(root,refs,config,seal)
    refs["ALA4"]["units"]=UNITS
    path=root/"ALA4"/"V"/"0"/"result.json"
    record=json.loads(path.read_text());record["data_unit_ids"]["production"]=["R-ALA4"]
    write(path,record)
    with pytest.raises(ValueError,match="泄漏"):
        aggregate_results(root,refs,config,seal)


def test_declared_endpoint_md_diagnostics_leave_e3_families_unchanged(complete_runs,tmp_path):
    root,refs,config,seal=complete_runs
    config["experiment"]["direct_md_replicas"]=128
    f_ids=list(seal.state["roles"]["F"]["unit_ids"])
    for replica in range(128):
        record=json.loads((root/"ALA4/V"/str(replica)/"result.json").read_text())
        record["method"]="endpoint-MD"
        record["data_unit_ids"]={"fallback":[f"MD-{replica}-fallback"],"production":[f"MD-{replica}-production"]}
        f_ids.extend(unit for units in record["data_unit_ids"].values() for unit in units)
        write(root/"ALA4/endpoint-MD"/str(replica)/"result.json",record)
    complete_seal=registry(tmp_path/"full-md-registry",f_ids,["R-ALA4","R-ALDP"])
    result=aggregate_results(root,refs,config,complete_seal)
    assert result["observed_deployments"]==2048
    assert result["direct_md_diagnostic_records"]==128
    assert result["comparison_count"]==28 and result["absolute_precision_count"]==4
    assert result["moment_interval_events"]==64
    record["method"]="unknown-method"
    write(root/"unknown/result.json",record)
    with pytest.raises(ValueError,match="八法"):
        aggregate_results(root,refs,config,complete_seal)
