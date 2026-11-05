import json
import math

import numpy as np
import pytest
from scipy.stats import t as student_t

from varipath.evaluation import UNITS
from varipath.inference import RoleSeal
from varipath.md_evaluation import endpoint_md_summary
from varipath.readouts import R_KJ


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def md_panel(tmp_path):
    root=tmp_path/"runs"
    ids=[];paths=[];vectors={}
    for method in ("V","endpoint-MD"):
        vectors[method]=[]
        for replica in range(128):
            error=.01*math.cos(replica) if method=="V" else .2+.02*math.sin(replica)
            normalized=.2+error
            vectors[method].append(normalized)
            readout=dict(preference=normalized*R_KJ*300,native_preference=normalized*R_KJ*300,
                         native_defined=True,units="kJ/mol",floor_triggered=False)
            roles=("fallback","fit","select","production") if method=="V" else ("fallback","production")
            data={role:[f"{method}:{replica}:{role}"] for role in roles}
            ids.extend(unit for values in data.values() for unit in values)
            hardware=dict(device="cpu",cpu_threads=1,machine="synthetic-test-machine")
            if method=="endpoint-MD":
                hardware["native_md_platform"]="CPU"
            record=dict(method=method,replica=replica,molecule="ALA4",mode="formal",status="completed",
                        eligible_for_formal=True,deployed=readout,native=readout,fallback_used=False,
                        data_unit_ids=data,budget_seconds=100.,budget_ratio=1.,execution_unit=hardware,
                        cost=dict(wall_seconds=5.,phases=[dict(name="paid-identity-fallback",status="complete")]))
            path=root/method/str(replica)/"result.json"
            write(path,record);paths.append(str(path.resolve()))
    seal=RoleSeal(tmp_path/"roles.json")
    for role in ("P","I"):
        artifact=tmp_path/f"{role}.json"
        write(artifact,{"scope":"synthetic contract fixture only"})
        seal.open(role,[role+"-fixture"]);seal.seal(role,[str(artifact)])
    seal.open("F",ids);seal.seal("F",paths)
    seal.open("R",["independent-R-fixture"])
    ref=dict(molecule="ALA4",role="R",reference_id="same-reference-for-both-methods",units=UNITS,
             source_unit_ids=["independent-R-fixture"],eligible_for_formal=True,family_alpha=.01,
             centers=dict(preference=.2),intervals=dict(preference=dict(lower=.199,upper=.201,status="valid")))
    config=dict(budget_seconds=100.,budget_ratio=1.,experiment=dict(replicas=128,direct_md_replicas=128))
    return root,ref,config,seal,vectors


def test_complete_paired_md_diagnostic_math_and_reference_projection(md_panel):
    root,ref,config,seal,vectors=md_panel
    result=endpoint_md_summary(root,{"ALA4":ref},config,seal)
    assert result["status"]=="diagnostic_complete"
    assert result["observed_deployments"]==256
    assert result["moment_interval_events"]==2 and result["added_E3_events"]==0
    assert result["excluded_from_formal_families"] and not result["eligible_for_formal"]
    v,m=np.asarray(vectors["V"])-.2,np.asarray(vectors["endpoint-MD"])-.2
    mse_v,mse_m=(values@values/128 for values in (v,m))
    assert result["point_mse_ratio_V_over_MD"]==pytest.approx(mse_v/mse_m)
    assert result["point_rmse_ratio_V_over_MD"]==pytest.approx(math.sqrt(mse_v/mse_m))
    observations=np.column_stack((v*v-m*m,v-m))
    half=student_t.ppf(.99,127)*observations.std(0,ddof=1)/math.sqrt(128)
    low,high=observations.mean(0)-half,observations.mean(0)+half
    product=[d*b for d in (-.001,.001) for b in (low[1],high[1])]
    interval=result["mse_difference_interval"]
    assert interval["lower"]==pytest.approx(low[0]-2*max(product))
    assert interval["upper"]==pytest.approx(high[0]-2*min(product))
    ref["intervals"]["preference"].update(lower=.05,upper=.35)
    wider=endpoint_md_summary(root,ref,config,seal)
    assert wider["mse_difference_interval"]["upper"]>interval["upper"]
    ref["eligible_for_formal"]=False
    unqualified=endpoint_md_summary(root,ref,config,seal)
    assert unqualified["status"]=="unresolved"
    assert unqualified["methods"]["V"]["valid_complete_outputs"]==128
    assert unqualified["methods"]["endpoint-MD"]["deployed_point_mse"] is None


def test_native_undefined_floor_fallback_and_missing_keep_128_denominators(md_panel):
    root,ref,config,seal,_=md_panel
    path=root/"endpoint-MD/0/result.json"
    record=json.loads(path.read_text())
    record["native"]["native_preference"]=None
    record["native"]["native_defined"]=False
    record["deployed"]["floor_triggered"]=True
    write(path,record)
    path=root/"endpoint-MD/1/result.json"
    record=json.loads(path.read_text());record.update(status="fallback",fallback_used=True,native=None)
    write(path,record)
    result=endpoint_md_summary(root,ref,config,seal)
    md=result["methods"]["endpoint-MD"]
    assert md["native_conditionally_defined_count"]==126
    assert md["native_undefined_rate"]==2/128
    assert md["floor_trigger_rate"]==1/128
    assert md["paid_fallback_rate"]==1/128
    assert md["native_conditional_rmse"] is not None
    assert md["deployed_point_mse"] is not None


    missing=endpoint_md_summary(root/"V",ref,config,seal)
    assert missing["status"]=="unresolved"
    assert missing["methods"]["endpoint-MD"]["native_undefined_rate"]==1.
    assert missing["methods"]["endpoint-MD"]["deployed_point_mse"] is None
    assert missing["point_mse_ratio_V_over_MD"] is None
    assert missing["mse_difference_interval"]["status"]=="undefined"
    assert missing["missing_deployments"]==[dict(method="endpoint-MD",replica=i) for i in range(128)]


    (root/"endpoint-MD/127/result.json").unlink()
    with pytest.raises(ValueError,match="封存产物已缺失"):
        endpoint_md_summary(root,ref,config,seal)


def test_no_legal_output_units_or_hardware_cannot_produce_finite_comparison(md_panel):
    root,ref,config,seal,_=md_panel
    path=root/"endpoint-MD/0/result.json"
    original=json.loads(path.read_text())
    record=dict(original,status="no_legal_output",eligible_for_formal=False,deployed=None,native=None)
    write(path,record)
    result=endpoint_md_summary(root,ref,config,seal)
    assert result["methods"]["endpoint-MD"]["deployed_point_mse"] is None
    assert result["mse_difference_interval"]["status"]=="undefined"
    record=json.loads(json.dumps(original));record["deployed"]["units"]="dimensionless"
    write(path,record)
    assert endpoint_md_summary(root,ref,config,seal)["status"]=="unresolved"
    record=json.loads(json.dumps(original));record["execution_unit"]["cpu_threads"]=2
    write(path,record)
    assert endpoint_md_summary(root,ref,config,seal)["point_mse_ratio_V_over_MD"] is None


def test_reference_leak_duplicates_and_unsealed_reference_are_rejected(md_panel,tmp_path):
    root,ref,config,seal,_=md_panel
    unopened=endpoint_md_summary(root,tmp_path/"must-not-read.json",config,None)
    assert unopened["status"]=="unresolved"
    path=root/"endpoint-MD/0/result.json"
    record=json.loads(path.read_text());record["data_unit_ids"]["production"]=["independent-R-fixture"]
    write(path,record)
    with pytest.raises(ValueError,match="泄漏"):
        endpoint_md_summary(root,ref,config,seal)
    record["data_unit_ids"]["production"]=["endpoint-MD:0:production"]
    write(path,record)
    write(root/"duplicate/result.json",record)
    with pytest.raises(ValueError,match="多份"):
        endpoint_md_summary(root,ref,config,seal)
