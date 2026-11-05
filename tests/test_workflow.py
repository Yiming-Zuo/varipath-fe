import copy
import math

import pytest
import torch

from varipath.cli import main
from varipath.workflow import (_WindowSystem, _transport_readout, doctor, prepare,
                              prepare_mechanism, reference, run_deployment, run_md, sample, seal_deployments)
from varipath.physics import MolecularSystem


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    root = tmp_path_factory.mktemp("real-molecular-workflow")
    config = dict(molecule="ALDP", mode="development", seed=1024, budget_seconds=300,
                  sampling=dict(count=4, warmup=2, ladders=2, states=[300., 550.],
                                leapfrog_steps=1, hmc_trials=1, checkpoint_every=1),
                  runtime=dict(max_transport_steps=1, threads=1), training=dict(max_steps=1))
    result = prepare(config, root/"preparation")
    assert result["status"] == "prepared"
    return root, config


def test_doctor_checks_real_molecular_energy_and_prepare_resume(workflow):
    root, config = workflow
    result = doctor(config)
    assert result["energy_error_kjmol"] < 1e-6
    assert result["shape_dimension"] == 48
    resumed = prepare(config, root/"preparation", resume=True)
    assert resumed["status"] == "prepared"
    with pytest.raises(FileExistsError):
        prepare(config, root/"preparation")


def test_actual_mbar_and_budget_rejection_use_separate_paid_banks(workflow):
    root, config = workflow
    result = run_deployment(config, root/"preparation", "E-MBAR", 0, root/"mbar")
    assert result["status"] == "completed"
    assert math.isfinite(result["deployed"]["delta_f"])
    assert not result["eligible_for_formal"]
    assert result["deployed"]["native_defined"] is False
    assert result["deployed"]["floor_triggered"] is True
    fallback = torch.load(root/"mbar/fallback/bank.pt", weights_only=False)
    production = torch.load(root/"mbar/production/bank.pt", weights_only=False)
    assert not set(fallback["source_unit_ids"]) & set(production["source_unit_ids"])
    resumed = run_deployment(config, root/"preparation", "E-MBAR", 0, root/"mbar", resume=True)
    assert resumed == result
    rejected = run_deployment(config, root/"preparation", "V0", 0, root/"v0")
    assert rejected["status"] == "fallback"
    assert rejected["native"] is None
    assert "未下调步数" in rejected["primary_failure"]["reason"]
    assert rejected["fallback_used"]
    assert math.isfinite(rejected["deployed"]["delta_f"])


def test_real_biased_window_sampling_is_not_a_relabelled_energy_bank(workflow):
    root, config = workflow
    config = copy.deepcopy(config)
    config["mbar"] = dict(states=[300., 400., 550.], biased_windows=[{}, {"energy":{"center":-30., "stiffness":.001}}, {}])
    result = run_deployment(config, root/"preparation", "EB-MBAR", 0, root/"eb-mbar")
    assert result["status"] == "completed", result
    assert result["diagnostics"]["actual_biased_windows"][1]["energy"]["stiffness"] > 0
    bank = torch.load(root/"eb-mbar/production/bank.pt", weights_only=False)
    assert bank["states"] == [1., 2., 3.]
    assert bank["u_kn"].shape[-2:] == (3, 3)
    assert math.isfinite(result["deployed"]["delta_f"])


def test_invalid_formal_fallback_has_no_legal_output_and_no_finite_score(workflow):
    root, config = workflow
    config = dict(config, mode="formal")
    result = run_deployment(config, root/"preparation", "E-MBAR", 0, root/"unqualified")
    assert result["status"] == "no_legal_output"
    assert result["deployed"] is None
    assert not result["eligible_for_formal"]


def test_role_R_samples_actual_independent_hamiltonian_bank(workflow):
    root, config = workflow
    result = sample(config, "R", root/"reference")
    assert result["status"] == "completed_sampling"
    assert not result["eligible_for_formal"]
    assert "readout" not in result
    assert result["reference_access"].startswith("sealed")
    bank = torch.load(root/"reference/sampling/bank.pt", weights_only=False)
    assert bank["role"] == "R"
    assert torch.isfinite(bank["u_kn"]).all()
    assert all(":R:" in unit for unit in bank["source_unit_ids"])


def test_cli_doctor_and_invalid_config_are_explicit(tmp_path, capsys):
    import json
    path = tmp_path/"config.json"
    path.write_text(json.dumps({"molecule":"ALDP"}))
    assert main(["doctor", str(path)]) == 0
    assert '"status": "passed"' in capsys.readouterr().out
    path.write_text(json.dumps({"molecule":"old-43-atom-ALA4"}))
    with pytest.raises(SystemExit) as error:
        main(["doctor", str(path)])
    assert error.value.code == 2


def test_incomplete_real_dimer_catalog_fails_instead_of_becoming_V0(workflow):
    root, config = workflow
    config = dict(config, catalog=dict(starts=2, force_limit=1, max_steps=1))
    result = run_deployment(config, root/"preparation", "V", 0, root/"insufficient-catalog")
    assert result["status"] == "fallback"
    assert "目录未取得预定数量" in result["primary_failure"]["reason"]
    assert result["native"] is None


def test_reference_embargo_and_full_F_scope_are_enforced_before_truth_read(workflow):
    root, config = workflow
    with pytest.raises(ValueError, match="读取 R 银行前"):
        reference(config, {"ALDP":root/"does-not-exist.pt", "ALA4":root/"also-missing.pt"},
                  root/"empty-roles.json", root/"blocked-reference")
    with pytest.raises(ValueError, match="非正式运行|尚缺"):
        seal_deployments(config, root/"mbar", root/"unsealed-F.json")
    assert not (root/"unsealed-F.json").exists()


def test_production_chunks_preserve_all_exact_weights_and_exposures():
    from varipath.protection import RadialProtection
    from varipath.transport import Transport

    class Harmonic:
        name = "harmonic"

        def effective_u(self, z, endpoint):
            return (z-.2*endpoint).square().sum()/2

        def regions(self, z):
            a, b = z[:, 0] < 0, z[:, 0] >= 0
            return dict(a=a, b=b, remainder=~(a|b), fine=b.long(), valid=torch.ones(len(z), dtype=torch.bool))

    model = Transport(1, RadialProtection([0], 4, 8), spline_count=4,
                      centers=[[.4]], bandwidths=[.8], amplitudes=[.3])
    with torch.no_grad():
        model.C.fill_(.08)
        model.alpha.fill_(.2)
    source = torch.linspace(-1.2, 1.5, 13, dtype=torch.float64)[:, None]
    split, split_arrays = _transport_readout(model, Harmonic(), source, {"runtime":{"production_batch_size":3}}, lambda:None)
    whole, whole_arrays = _transport_readout(model, Harmonic(), source, {"runtime":{"production_batch_size":100}}, lambda:None)
    for name in ("target", "log_weights", "ordinary_logdet"):
        torch.testing.assert_close(split_arrays[name], whole_arrays[name], atol=2e-14, rtol=2e-14)
    assert split["anchor_exposure"] == pytest.approx(whole["anchor_exposure"], abs=2e-15)
    assert split["protection_fractions"] == pytest.approx(whole["protection_fractions"], abs=2e-15)


@pytest.mark.parametrize("name", ["ALDP", "ALA4"])
def test_line_search_scale_receives_both_physical_reduced_endpoint_potentials(tmp_path, monkeypatch, name):
    import varipath.workflow as workflow_module
    from varipath.readouts import R_KJ

    system = MolecularSystem(name)
    point = system.initial_coordinates()
    positions = point.reshape(1, 1, 1, -1).expand(1, 2, 2, -1).clone()
    bank = dict(positions=positions, states=sorted([system.state(0), system.state(1)]))
    received = []

    def inspect_training(*args, **kwargs):
        potentials = kwargs["scale_potential"]
        assert len(potentials) == 2
        for endpoint, potential in enumerate(potentials):
            coupling = float(endpoint) if name == "ALA4" else 1.
            temperature = 550. if name == "ALDP" and endpoint == 0 else 300.
            expected = system.energy(system.positions(point), coupling)/(R_KJ*temperature)
            actual = potential(point)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            received.append(endpoint)
        return {"status":"checked"}

    monkeypatch.setattr(workflow_module, "train_transport", inspect_training)
    result = workflow_module._fit_transport(None, system, bank, bank, {}, tmp_path/"unused.pt", lambda:None)
    assert result["status"] == "checked"
    assert received == [0, 1]


def test_R_four_planned_lengths_extend_same_real_bank_without_revealing_truth(workflow):
    import json
    root, config = workflow
    output=root/"extending-reference"
    first=sample(config,"R",output,look=1)
    prefix=torch.load(output/"sampling/bank.pt",weights_only=False)
    second=sample(config,"R",output,look=2)
    actual=torch.load(output/"sampling/bank.pt",weights_only=False)
    assert second["look"]==2
    assert second["production_draws_per_ladder"]==2*first["production_draws_per_ladder"]
    assert second["independent_units"]==first["independent_units"]
    assert second["sampling_elapsed_seconds"]>=first["sampling_elapsed_seconds"]
    assert "readout" not in second
    torch.testing.assert_close(actual["positions"][:,:4],prefix["positions"],atol=0,rtol=0)
    with pytest.raises(ValueError,match="不能把较新"):
        sample(config,"R",output,resume=True,look=1)
    for look in (3,4):
        continued=sample(config,"R",output,look=look)
        current=torch.load(output/"sampling/bank.pt",weights_only=False)
        assert continued["look"]==look
        assert continued["production_draws_per_ladder"]==4*2**(look-1)
        assert "readout" not in continued
        torch.testing.assert_close(current["positions"][:,:4],prefix["positions"],atol=0,rtol=0)
    stopped_output=root/"stopped-reference"
    sample(config,"R",stopped_output,look=1)
    (stopped_output/"sampling/reference-status.json").write_text(json.dumps({"look":1,"final":True,"stopped":True}))
    with pytest.raises(ValueError,match="停止|final"):
        sample(config,"R",stopped_output,look=2)


def test_budget_checks_are_frequent_while_cost_persistence_is_throttled(tmp_path,monkeypatch):
    import varipath.workflow as workflow_module
    from varipath.records import Budget
    budget=Budget(10)
    check_calls,write_calls=0,0
    original_check=budget.check
    original_write=workflow_module.write_json

    def checked():
        nonlocal check_calls
        check_calls+=1
        original_check()

    def written(path,value):
        nonlocal write_calls
        write_calls+=1
        original_write(path,value)

    monkeypatch.setattr(budget,"check",checked)
    monkeypatch.setattr(workflow_module,"write_json",written)
    callback=workflow_module._checker(budget,tmp_path)
    for _ in range(100):
        callback()
    assert check_calls==100
    assert write_calls==1
    budget.seconds=0
    with pytest.raises(TimeoutError):
        callback()


def test_formal_E3_models_are_rejected_before_opening_as_D_mechanism_inputs(tmp_path):
    import json
    deployment=tmp_path/"formal-F"
    deployment.mkdir()
    (deployment/"result.json").write_text(json.dumps({"mode":"formal"}))
    with pytest.raises(ValueError,match="专用 D|正式 E3"):
        prepare_mechanism({"molecule":"ALDP"},tmp_path/"absent-preparation",deployment,tmp_path/"panel")
    assert not (tmp_path/"panel").exists()


def test_native_endpoint_MD_wrapper_shares_paid_budget_and_never_reports_deltaF(tmp_path):
    config=dict(molecule="ALA4",mode="development",seed=1024,budget_seconds=300,
                sampling=dict(count=2,warmup=1,ladders=2,states=[0.,1.],leapfrog_steps=1,hmc_trials=1,checkpoint_every=1),
                endpoint_md=dict(platform="Reference",steps_per_segment=1,timestep_fs=.2,ladders=4,warmup=1,
                                 allocation=[1,1],total_saved_per_ladder=4,checkpoint_every=1),
                endpoint_md_qualification=dict(warmup_selection="D_frozen"),runtime=dict(threads=1))
    preparation=tmp_path/"preparation"
    prepare(config,preparation)
    result=run_md(config,preparation,0,tmp_path/"MD")
    assert result["status"]=="completed",result
    assert not result["eligible_for_formal"]
    assert result["method"]=="endpoint-MD"
    assert result["deployed"]["delta_f"] is None
    assert result["deployed"]["delta_free_energy"] is None
    assert math.isfinite(result["deployed"]["preference"])
    assert result["data_unit_ids"]["fallback"]
    assert len(result["data_unit_ids"]["production"])==8
    assert not set(result["data_unit_ids"]["fallback"])&set(result["data_unit_ids"]["production"])
    assert any(row["name"]=="paid-identity-fallback" and row["status"]=="complete" for row in result["cost"]["phases"])
    assert result["cost"]["counts"]["endpoint_md_steps"]>0
    assert run_md(config,preparation,0,tmp_path/"MD",resume=True)==result


def test_F_seal_scope_includes_configured_128_MD_runs_without_adding_E3_comparisons(tmp_path):
    config=dict(molecule="ALA4",experiment=dict(direct_md_replicas=128))
    with pytest.raises(ValueError,match="2176"):
        seal_deployments(config,tmp_path/"empty-runs",tmp_path/"roles.json")


@pytest.mark.parametrize("role,stage",[("R","reference"),("P","prediction"),("I","intervention")])
def test_research_sampling_can_use_explicit_no_cap_without_disabling_cold_budget(tmp_path,role,stage):
    import json
    from varipath.records import Budget
    from varipath.workflow import load_config
    with pytest.raises(ValueError,match="research"):
        Budget(None)
    research=Budget(None,research=True)
    research.previous=1e8
    research.check()
    assert research.record()["budget_seconds"] is None
    assert research.record()["budget_kind"]=="research"
    with pytest.raises(ValueError,match="冷启动预算"):
        load_config({"molecule":"ALDP","budget_seconds":None})
    config=dict(molecule="ALDP",mode="development",budget_seconds=.001,seed=1024,
                sampling=dict(count=2,warmup=1,ladders=2,states=[300.,550.],leapfrog_steps=1,checkpoint_every=1))
    config[stage]={"budget_seconds":None}
    result=sample(config,role,tmp_path/role)
    assert result["status"]=="completed_sampling"
    cost=json.loads((tmp_path/role/"cost.json").read_text())
    assert cost["budget_seconds"] is None
    assert cost["budget_kind"]=="research"
    assert cost["wall_seconds"]>config["budget_seconds"]


def test_uncapped_reference_recipe_never_removes_a_methods_paid_cold_start_cap(workflow):
    root, config = workflow
    config=dict(config,budget_seconds=.001,reference=dict(budget_seconds=None))
    result=run_deployment(config,root/"preparation","E-MBAR",2,root/"expired-cold-budget")
    assert result["status"]=="no_legal_output"
    assert result["deployed"] is None
    assert result["cost"]["budget_seconds"]==.001
    assert result["cost"]["budget_kind"]=="cold_start"


def test_old_geometry_certificate_panels_are_not_silently_reused():
    from types import SimpleNamespace
    from varipath.workflow import _require_panel_schema
    with pytest.raises(ValueError,match="旧 E1 几何证书"):
        _require_panel_schema(SimpleNamespace(development={}))
    _require_panel_schema(SimpleNamespace(development={"geometry_certificate_schema":"interval_heavy_basin_v1"}))


def test_older_workflow_stream_contract_is_preserved_and_explicitly_rejected(tmp_path):
    import json
    from varipath.workflow import _contract
    path=tmp_path/"run-config.json"
    path.write_text(json.dumps({"action":"prepare","config":{"molecule":"ALDP"}}))
    old=path.read_bytes()
    with pytest.raises(ValueError,match="旧工作流或随机流版本"):
        _contract(tmp_path,{"action":"prepare","config":{"molecule":"ALDP"}},True)
    assert path.read_bytes()==old


def test_completed_bank_verifies_Hamiltonian_affine_and_nonbuffer_definition_before_reuse(tmp_path,monkeypatch):
    import varipath.workflow as workflow_module
    from varipath.physics import AffineSystem
    from varipath.sampling import run_sampling
    raw=MolecularSystem("ALDP")
    system=AffineSystem(raw,torch.zeros(raw.dim,dtype=torch.float64),torch.ones(raw.dim,dtype=torch.float64))
    cfg=dict(count=2,warmup=1,ladders=2,seed=200,states=[300.,550.],mode="development",role="F-fit",
             leapfrog_steps=1,checkpoint_every=1,random_stream_version=2)
    original=run_sampling(system,cfg,tmp_path)
    bank_bytes=(tmp_path/"bank.pt").read_bytes()
    checkpoint_bytes=(tmp_path/"ladder-0.pt").read_bytes()

    def never_sample(*args,**kwargs):
        raise AssertionError("完成银行不得重新采样")

    monkeypatch.setattr(workflow_module,"run_sampling",never_sample)
    cached=workflow_module._bank(system,cfg,tmp_path,{"mode":"development"},lambda:None,require_qualified=False)
    torch.testing.assert_close(cached["positions"],original["positions"],atol=0,rtol=0)
    changed_force=copy.deepcopy(system)
    changed_force.system.energy.bond_parameters[0,0]+=.001
    changed_affine=copy.deepcopy(system)
    changed_affine.scale[0]*=1.01
    changed_gb=copy.deepcopy(system)
    changed_gb.system.energy.gb_kind="OBC2"
    changed_chart=copy.deepcopy(system)
    changed_chart.chart.triad=tuple(reversed(changed_chart.chart.triad))
    for current in (changed_force,changed_affine,changed_gb,changed_chart):
        with pytest.raises(ValueError,match="发生变化"):
            workflow_module._bank(current,cfg,tmp_path,{"mode":"development"},lambda:None,require_qualified=False)
        assert (tmp_path/"bank.pt").read_bytes()==bank_bytes
        assert (tmp_path/"ladder-0.pt").read_bytes()==checkpoint_bytes


def test_cached_biased_window_definition_cannot_change_outside_tensor_buffers(tmp_path):
    from varipath.sampling import run_sampling
    from varipath.workflow import _bank
    system=_WindowSystem(MolecularSystem("ALDP"),[300.,400.,550.],[{}, {"energy":{"center":-30.,"stiffness":.001}}, {}])
    cfg=dict(count=2,warmup=1,ladders=2,seed=201,states=system.encoded,mode="development",role="F-production",
             leapfrog_steps=1,checkpoint_every=1,random_stream_version=2)
    run_sampling(system,cfg,tmp_path)
    before=(tmp_path/"bank.pt").read_bytes()
    system.windows[1]["energy"]["stiffness"]*=2
    with pytest.raises(ValueError,match="固定物理定义"):
        _bank(system,cfg,tmp_path,{"mode":"development"},lambda:None,require_qualified=False)
    assert (tmp_path/"bank.pt").read_bytes()==before


def test_legacy_bank_uses_actual_checkpoint_proof_and_refuses_missing_definitions(tmp_path,monkeypatch):
    import varipath.workflow as workflow_module
    from varipath.sampling import run_sampling
    system=MolecularSystem("ALDP")
    cfg=dict(count=2,warmup=1,ladders=2,seed=202,states=[300.,550.],mode="development",role="R",
             leapfrog_steps=1,checkpoint_every=1,random_stream_version=2)
    bank=run_sampling(system,cfg,tmp_path)
    bank.pop("frozen_system")
    torch.save(bank,tmp_path/"bank.pt")
    monkeypatch.setattr(workflow_module,"run_sampling",lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("不得重新采样")))
    recovered=workflow_module._bank(system,cfg,tmp_path,{"mode":"development"},lambda:None,require_qualified=False)
    assert recovered["frozen_system"]["physical_definition"]["schema_version"]==1
    checkpoint=torch.load(tmp_path/"ladder-0.pt",weights_only=False)
    checkpoint["sampler"]["kernels"][0].pop("physics_definition")
    torch.save(checkpoint,tmp_path/"ladder-0.pt")
    before=(tmp_path/"bank.pt").read_bytes()
    with pytest.raises(ValueError,match="非 tensor 物理定义"):
        workflow_module._bank(system,cfg,tmp_path,{"mode":"development"},lambda:None,require_qualified=False)
    assert (tmp_path/"bank.pt").read_bytes()==before


def test_prepare_terminal_cache_checks_original_D_fit_physics_without_loading_old_state(workflow,monkeypatch):
    import varipath.workflow as workflow_module
    root,config=workflow
    prepared=(root/"preparation/prepared.pt").read_bytes()
    bank=(root/"preparation/fit/bank.pt").read_bytes()
    system=MolecularSystem("ALDP")
    system.energy.bond_parameters[0,0]+=.001
    monkeypatch.setattr(workflow_module,"_setup",lambda unused:system)
    with pytest.raises(ValueError,match="Hamiltonian"):
        prepare(config,root/"preparation",resume=True)
    assert (root/"preparation/prepared.pt").read_bytes()==prepared
    assert (root/"preparation/fit/bank.pt").read_bytes()==bank
