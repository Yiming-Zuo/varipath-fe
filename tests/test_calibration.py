import copy

import numpy as np
import pytest

from varipath.calibration import (SCOPE, _binomial, generate_scenario, load_config,
                                  run_calibration, simulate_once)


def config():
    return load_config(dict(scope=SCOPE, seed=7, simulations=10,
        reference_initial_draws=16, generating_assumptions=["Independent synthetic unit-level controls"],
        model=dict(distribution="normal", degrees_of_freedom=8, theta=np.zeros((2, 2)).tolist(),
            method_standard_deviations=(np.arange(1, 9)/100).tolist(), method_biases=[0.]*8,
            correlation=.25, reference_correlation=.125, prediction_sd=.001, intervention_sd=.001,
            reference_frame_sd=.025, mass_effect=.025, risk_effect=.075, shape_effect=.025,
            geometry_bias=0., half_dose_residual=0., intervention_mass_bias=0., intervention_risk_bias=0.,
            target_a=[.35, .4], baseline_a=[.25, .3], failure_probability=0.,
            P_failure_probability=0., I_failure_probability=0., R_failure_probability=0.),
        cases=[dict(name="declared_alternative"),
            dict(name="weak_effect", method_standard_deviations=[.07]*8),
            dict(name="harmful", method_standard_deviations=[.2]+[.1]*7, target_a=[.125, .15]),
            dict(name="correlated", correlation=.9),
            dict(name="shared_reference", reference_frame_sd=.25),
            dict(name="heavy_finite_fourth", distribution="student_t", degrees_of_freedom=5),
            dict(name="heavy_infinite_fourth", distribution="student_t", degrees_of_freedom=3),
            dict(name="failures", failure_probability=.1, P_failure_probability=.1,
                I_failure_probability=.1, R_failure_probability=.1)]))


def test_complete_vectors_independent_roles_and_one_shared_reference():
    cfg=config();seeds=[101,202,303,404]
    draw=generate_scenario(cfg,cfg["cases"][0],seeds)
    assert draw["P"].shape==(16,552) and draw["I"].shape==(16,418)
    assert draw["F"].shape==(128,2,8,2) and draw["R"].shape==(4,4,6)
    changed=generate_scenario(cfg,cfg["cases"][0],[101,202,999,404])
    np.testing.assert_array_equal(changed["R"],draw["R"])
    np.testing.assert_array_equal(changed["P"],draw["P"])
    assert not np.array_equal(changed["F"],draw["F"])
    result=simulate_once(cfg,cfg["cases"][0],seeds)
    assert result["reference_vectors_generated"]==1
    assert result["reference_applications"]==32
    assert len(set(result["reference_ids_used"]))==4
    assert result["gate_count"]==48
    correlated=generate_scenario(cfg,cfg["cases"][3],seeds)
    assert np.corrcoef(correlated["F"][:,0,0,0],correlated["F"][:,0,1,0])[0,1]>.6


def test_failure_units_remain_undefined_and_infinite_fourth_is_not_certified():
    cfg=config()
    case=dict(cfg["cases"][0],name="all_failed",failure_probability=1.,P_failure_probability=1.,I_failure_probability=1.,R_failure_probability=1.)
    report=simulate_once(cfg,case,[1,2,3,4])
    assert not report["joint_success"] and not report["all_gates_defined"]
    assert not report["simultaneous_coverage"]
    assert not report["false_acceptance"]
    heavy=simulate_once(cfg,cfg["cases"][6],[1,2,3,4])
    assert not heavy["marginal_conditions_confirmed_for_generating_model"]


def test_resume_replays_exact_rng_and_preserves_assumptions(tmp_path):
    cfg=config();cfg["simulations"]=10;cfg["cases"]=cfg["cases"][:1]
    partial=run_calibration(cfg,tmp_path/"resumed",stop_after=3)
    assert partial["status"]=="checkpointed" and partial["completed"]==3
    resumed=run_calibration(cfg,tmp_path/"resumed",resume=True)
    uninterrupted=run_calibration(cfg,tmp_path/"continuous")
    assert resumed["cases"]==uninterrupted["cases"]
    assert resumed["total_simulations"]==10 and resumed["scope"]==SCOPE
    assert resumed["cases"]["declared_alternative"]["joint_power_gate"]=="unresolved"
    changed=copy.deepcopy(cfg);changed["model"]["prediction_sd"]*=2
    with pytest.raises(ValueError,match="不能改变"):
        run_calibration(changed,tmp_path/"resumed",resume=True)
    with pytest.raises(FileExistsError):
        run_calibration(cfg,tmp_path/"resumed")


def test_exact_binomial_power_bound_and_explicit_simulation_count():
    assert config()["simulations"]==10
    assert _binomial(1000,1000)["one_sided_95_lower"]==pytest.approx(.05**.001)
    assert _binomial(0,1000)["one_sided_95_lower"]==0


@pytest.mark.parametrize("case_index",[1,2,3,4,5,7])
def test_all_declared_scenario_branches_execute_without_changing_family(case_index):
    cfg=config();case=cfg["cases"][case_index]
    result=simulate_once(cfg,case,[17,29,43,61])
    assert set(result["family_coverage"])=={"P","I","E3","R"}
    assert result["dimensions"]==dict(P=[16,552],I=[16,418],F=[128,2,8,2],R=[4,4,6])
    assert result["reference_applications"]==32
    if case["name"]=="harmful":
        assert not result["alternative"]
    if case["name"]=="failures":
        assert not result["all_gates_defined"]
