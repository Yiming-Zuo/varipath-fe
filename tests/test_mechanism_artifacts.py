from pathlib import Path

import pytest
import torch

import varipath.mechanism as mechanism
from test_mechanism import panel, source


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    directory=tmp_path_factory.mktemp("mechanism-raw")
    built=panel()
    prediction=mechanism.run_predictions(built,source("P"),directory,{"numerical_error_bound":0.})
    intervention=mechanism.run_interventions(prediction,source("I"),directory)
    return directory,built,prediction,intervention


def first_block(result):
    return next(record["blocks"][0] for record in result["cells"].values() if record["status"]=="computed")


def test_compact_records_keep_all_real_outputs_in_relative_atomic_artifacts(produced):
    directory,_,prediction,intervention=produced
    assert prediction["artifact_storage_schema"]=="artifact_storage_v1"
    assert intervention["artifact_storage_schema"]=="artifact_storage_v1"
    disk=torch.load(directory/"predictions.pt",weights_only=False)
    assert "panels" not in disk and "sources" not in disk
    assert (directory/disk["context_path"]).is_file()
    restored=mechanism.load_mechanism_result(directory/"predictions.pt")
    assert len(restored["panels"])==1 and "ALDP" in restored["sources"]
    p=first_block(prediction)
    assert set(p["response"])=={"score","dot_logweight","logweight"}
    assert not Path(p["raw_path"]).is_absolute()
    raw=mechanism.load_raw_block(directory,p,expected_role="P")
    assert "endpoint" in raw["response"] and "displacement" in raw["response"]
    torch.testing.assert_close(raw["response"]["logweight"],p["response"]["logweight"],rtol=0,atol=0)
    actual_source=torch.load(directory/raw["source_path"],weights_only=False)
    torch.testing.assert_close(actual_source["samples"],source("P").block(p["block"]),rtol=0,atol=0)
    i=first_block(intervention)
    assert all("endpoint" not in i["result"][branch] for branch in ("baseline","full","half"))
    raw=mechanism.load_raw_block(directory,i,expected_role="I")
    assert all("endpoint" in raw["result"][branch] for branch in ("baseline","full","half"))
    for branch in ("baseline","full","half"):
        torch.testing.assert_close(raw["result"][branch]["logweight"],i["result"][branch]["logweight"],rtol=0,atol=0)
    assert prediction["raw_storage"]["bytes"]>0 and intervention["raw_storage"]["bytes"]>0
    assert not list((directory/"raw").rglob("*.pending"))


def test_missing_true_raw_data_cannot_resume_as_completed(produced):
    directory,built,prediction,intervention=produced
    for result,callback in ((prediction,lambda:mechanism.run_predictions(built,source("P"),directory,{"numerical_error_bound":0.},resume=True)),
                            (intervention,lambda:mechanism.run_interventions(prediction,source("I"),directory,resume=True))):
        block=first_block(result)
        path=directory/block["raw_path"]
        original=path.read_bytes();path.unlink()
        try:
            with pytest.raises(ValueError,match="缺失"):
                callback()
        finally:
            path.write_bytes(original)
        context=directory/result["context_path"]
        original=context.read_bytes();context.unlink()
        try:
            with pytest.raises(ValueError,match="缺失"):
                callback()
        finally:
            context.write_bytes(original)


def test_raw_block_resume_reuses_completed_execution_after_mid_cell_interrupt(tmp_path,monkeypatch):
    built=panel()
    original=mechanism.response_samples
    calls=0
    def interrupted(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise KeyboardInterrupt("test interrupt before next raw block")
        return original(*args,**kwargs)
    monkeypatch.setattr(mechanism,"response_samples",interrupted)
    with pytest.raises(KeyboardInterrupt):
        mechanism.run_predictions(built,source("P"),tmp_path,{"numerical_error_bound":0.})
    assert (tmp_path/"predictions-working.pt").is_file()
    assert len(list((tmp_path/"raw"/"P").glob("ALDP-*/block-*.pt")))==1
    remaining=0
    def continued(*args,**kwargs):
        nonlocal remaining
        remaining+=1
        return original(*args,**kwargs)
    monkeypatch.setattr(mechanism,"response_samples",continued)
    result=mechanism.run_predictions(built,source("P"),tmp_path,{"numerical_error_bound":0.},resume=True)
    blocks=sum(len(record["blocks"]) for record in result["cells"].values() if record["status"]=="computed")
    assert remaining==blocks-1
    assert result["sealed"]


def test_legacy_full_records_are_readable_but_not_executable_resume(produced):
    directory,_,prediction,_=produced
    raw=mechanism.load_raw_block(directory,first_block(prediction))
    legacy={"response":raw["response"],"labels":raw["labels"]}
    assert mechanism.load_raw_block(directory,legacy)["response"]["endpoint"].shape[1]==1
    old=dict(prediction);old.pop("artifact_storage_schema")
    with pytest.raises(ValueError,match="旧存储"):
        mechanism._validate_resumed_sources(old,{"ALDP":source("P")},{"numerical_error_bound":0.},directory)
