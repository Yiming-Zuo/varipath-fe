import math

import numpy as np
import pytest

from varipath.inference import (
    Interval, RoleSeal, absolute_mse_interval, bonferroni_t, intervention_interval_names,
    paired_mse_interval, panel_agreement, panel_manifest, prediction_gate, repair_intervals,
    select_same_side_pair,
)


def test_simultaneous_families_preserve_failure_and_reference_uncertainty():
    rng = np.random.default_rng(1024)
    values = rng.normal(size=(128, 2))
    intervals = bonferroni_t(values, "I")
    assert all(x.lower < x.upper for x in intervals)
    with pytest.raises(ValueError, match="manifest"):
        bonferroni_t(values, "P")
    with pytest.raises(ValueError, match="418"):
        bonferroni_t(values, "I", events=400)
    values[0, 0] = np.nan
    assert bonferroni_t(values, "I")[0].status == "undefined"
    variant = np.zeros(128)
    baseline = np.ones(128)
    exact = paired_mse_interval(variant, baseline, Interval(0., 0.), reference_center=0.)
    uncertain = paired_mse_interval(variant, baseline, Interval(-1., 1.), reference_center=0.)
    assert exact["status"] == "supported"
    assert uncertain["status"] == "unresolved"
    absolute = absolute_mse_interval(variant, Interval(-.2, .2), reference_center=0.)
    assert absolute["interval"].upper == pytest.approx(.04)
    assert absolute["status"] == "unresolved"
    variant[3] = np.nan
    assert absolute_mse_interval(variant, Interval(0., 0.), reference_center=0.)["status"] == "unresolved"


def test_repair_projection_catches_overshoot_and_zero_deficit():
    reference, baseline = Interval(.2, .21), Interval(.1, .11)
    repair = repair_intervals(reference, baseline, Interval(.03, .04))
    assert repair["deficit"].lower == pytest.approx(.09)
    assert repair["gain"].lower > 0
    overshoot = repair_intervals(reference, baseline, Interval(.25, .26))
    assert overshoot["gain"].upper < 0
    uncertain = repair_intervals(Interval(.1, .2), Interval(.15, .25), Interval(.01, .02))
    assert uncertain["deficit"].lower < 0 < uncertain["deficit"].upper


def test_gate_projects_every_point_in_prediction_rectangle():
    assert prediction_gate(Interval(.01, .011), Interval(.0105, .0115), .005)["status"] == "supported"
    assert prediction_gate(Interval(.01, .011), Interval(.04, .05), .005)["status"] == "unsupported"
    assert prediction_gate(Interval(-.1, .1), Interval(-.001, .001), .005)["status"] == "unresolved"


def test_manifest_keeps_all_failed_or_unrun_slots_and_same_side_separate():
    cells = panel_manifest()
    assert len(cells) == 96
    assert sum(cell["panel"] == "main" for cell in cells) == 72
    assert sum(cell["panel"] == "same_side" for cell in cells) == 24
    assert all(cell["status"] == "not_run" for cell in cells)
    missing = panel_agreement({}, {}, {})
    assert missing["main"]["status"] == "unresolved"
    assert missing["main"]["cells"] == 72
    assert missing["same_side"]["cells"] == 24
    names = intervention_interval_names()
    assert len(names) == 418 and len(set(names)) == 418
    dl = math.log(1.02)
    candidate = {"data_role": "P", "anchor": 0, "window": 2,
                 "mass": [Interval(.01, .02), Interval(.012, .02)],
                 "log_risk": [Interval(-3*dl, -2*dl), Interval(2*dl, 3*dl)]}
    assert select_same_side_pair([candidate], .04)["status"] == "P_selected"
    assert select_same_side_pair([candidate], .005) is None
    with pytest.raises(ValueError, match="只能由 P"):
        select_same_side_pair([{**candidate, "data_role": "I"}], .04)


def test_role_seals_enforce_chain_independence_and_blinding(tmp_path):
    seals = RoleSeal(tmp_path / "roles.json")
    with pytest.raises(ValueError, match="封存"):
        seals.open("I", ["ladder-I"])
    seals.open("P", ["ladder-P"])
    output = tmp_path / "prediction.json"
    output.write_text("{}", encoding="utf-8")
    seals.seal("P", [str(output)])
    with pytest.raises(ValueError, match="共同 chain"):
        seals.open("I", ["ladder-P"])
    seals.open("I", ["ladder-I"])
    intervention = tmp_path / "intervention.json"
    intervention.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="同一输出"):
        seals.seal("I", [str(output)])
    seals.seal("I", [str(intervention)])
    with pytest.raises(ValueError, match="封存"):
        seals.open("R", ["ladder-R"])
    seals.open("F", ["ladder-F"])
    production = tmp_path / "production.json"
    production.write_text("{}", encoding="utf-8")
    seals.seal("F", [str(production)])
    seals.open("R", ["ladder-R"])
    resumed = RoleSeal(tmp_path / "roles.json")
    assert resumed.state["roles"]["P"]["status"] == "sealed"
    with pytest.raises(ValueError, match="更换"):
        resumed.open("P", ["other-ladder"])
