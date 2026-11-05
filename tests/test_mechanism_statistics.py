import math

import numpy as np
import pytest
import torch

from varipath.mechanism import (_delta_rows, _intervention_risk_statistics,
    _intervention_shape_statistics, _panels_and_sources, _prediction_moment_statistics,
    _validate_resumed_sources, SourceBatch)


DTYPE = torch.float64


def test_prediction_global_weighted_risk_and_conditional_variance_keep_block_scales():
    blocks = []
    for logweight, score, derivative, labels in (
        ([0., 0.2, -.1], [1., 3., -2.], [2., -1., 1.], [0, 1, 1]),
        ([3., 4., 2.], [-1., 2., 5.], [-2., 3., 4.], [0, 0, 1]),
        ([1., 1.3, .7], [6., 1., 4.], [1., 2., -4.], [1, 0, 0]),
    ):
        blocks.append(dict(labels=torch.tensor(labels), response={
            "logweight": torch.tensor(logweight, dtype=DTYPE),
            "score": torch.tensor(score, dtype=DTYPE),
            "dot_logweight": torch.tensor(derivative, dtype=DTYPE)}))
    epsilon = .02
    result = _prediction_moment_statistics(blocks, epsilon)
    logw = torch.cat([b["response"]["logweight"] for b in blocks])
    derivative = torch.cat([b["response"]["dot_logweight"] for b in blocks])
    score = torch.cat([b["response"]["score"] for b in blocks])
    region = torch.cat([b["labels"] == 0 for b in blocks])
    exact_risk = float(2*epsilon*(torch.softmax(2*logw, 0)*derivative).sum())
    exact_shape = float(epsilon**2*score[region].var(unbiased=False)/8)
    assert result["log_risk"]["point"] == pytest.approx(exact_risk, abs=1e-14)
    assert result["conditional_h2"]["point"] == pytest.approx(exact_shape, abs=1e-16)
    for field in ("log_risk", "conditional_h2"):
        assert result[field]["rows"].mean() == pytest.approx(result[field]["point"], abs=1e-14)
    old_block_mean = np.mean([float(2*epsilon*(torch.softmax(2*b["response"]["logweight"], 0)*b["response"]["dot_logweight"]).sum()) for b in blocks])
    assert abs(old_block_mean-exact_risk) > .01


def test_intervention_global_log_moment_contrasts_and_pairing():
    blocks = []
    for source, full, half in (
        ([0., .3], [.5, -.2], [.2, .1]),
        ([5., 3.], [3., 3.5], [4., 2.5]),
        ([2., 1.], [2.5, 2.], [2.1, 1.4]),
    ):
        blocks.append({"result":{branch:{"logweight":torch.tensor(value, dtype=DTYPE)}
                                 for branch, value in (("baseline", source), ("full", full), ("half", half))}})
    result = _intervention_risk_statistics(blocks)
    moments = {key:float(torch.logsumexp(2*torch.cat([b["result"][key]["logweight"] for b in blocks]), 0)-math.log(6)) for key in ("baseline", "full", "half")}
    assert result["full"]["point"] == pytest.approx(moments["full"]-moments["baseline"], abs=1e-14)
    assert result["full_minus_twice_half"]["point"] == pytest.approx(moments["full"]+moments["baseline"]-2*moments["half"], abs=1e-14)
    for contrast in ("full", "full_minus_twice_half"):
        assert result[contrast]["rows"].mean() == pytest.approx(result[contrast]["point"], abs=1e-14)
    old = np.mean([float(torch.logsumexp(2*b["result"]["full"]["logweight"], 0)-torch.logsumexp(2*b["result"]["baseline"]["logweight"], 0)) for b in blocks])
    assert abs(old-result["full"]["point"]) > .5


    for b in blocks:
        b["result"]["full"]["logweight"] += 1000
    shifted = _intervention_risk_statistics(blocks)
    assert shifted["full"]["point"] == pytest.approx(result["full"]["point"]+2000, abs=1e-12)
    np.testing.assert_allclose(shifted["full"]["rows"]-2000, result["full"]["rows"], atol=1e-12)


def test_conditional_h2_uses_joint_affinity_and_masses_not_mean_block_ratios():
    blocks = []
    raw = [(0., .5, .25, .1, .05), (.75, .5, .75, .55, .65), (.75, .5, .5, .4, .35)]
    for p0, pf, ph, af, ah in raw:


        shapes = {}
        for name, pe, affinity in (("shape_full", pf, af), ("shape_half", ph, ah)):
            shapes[name] = dict(rho0=torch.tensor([p0], dtype=DTYPE), rhoe=torch.tensor([pe], dtype=DTYPE),
                affinity_integrand=torch.full((8,), affinity, dtype=DTYPE), labels=torch.zeros(8, dtype=torch.long))
        blocks.append({"result":{"baseline":{"logweight":torch.zeros(4, dtype=DTYPE)}, **shapes}})
    result = _intervention_shape_statistics(blocks)
    p0, pf, ph, af, ah = np.mean(raw, axis=0)
    full = 1-af/math.sqrt(p0*pf)
    half = 1-ah/math.sqrt(p0*ph)
    assert result["full"]["point"] == pytest.approx(full, abs=1e-14)
    assert result["full_minus_four_half"]["point"] == pytest.approx(full-4*half, abs=1e-14)
    assert np.isfinite(result["full"]["rows"]).all()
    assert result["full"]["rows"].mean() == pytest.approx(full, abs=1e-14)


def test_delta_rows_match_analytic_influence_and_zero_mass_stays_undefined():
    moments = torch.tensor([[1., 2.], [2., -1.], [4., 3.]], dtype=DTYPE)
    point, rows = _delta_rows(moments, lambda m: m[1]/m[0])
    mean = moments.mean(0)
    expected = point+(moments-mean)@torch.tensor([-mean[1]/mean[0]**2, 1/mean[0]], dtype=DTYPE)
    np.testing.assert_allclose(rows, expected.numpy(), atol=1e-15)
    blocks = [dict(labels=torch.ones(2, dtype=torch.long), response=dict(logweight=torch.zeros(2), score=torch.ones(2), dot_logweight=torch.ones(2))) for _ in range(2)]
    result = _prediction_moment_statistics(blocks, .1)
    assert math.isnan(result["conditional_h2"]["point"])
    assert np.isnan(result["conditional_h2"]["rows"]).all()
    blocks[1]["response"]["logweight"] = torch.zeros(3)
    with pytest.raises(ValueError, match="等大小"):
        _prediction_moment_statistics(blocks, .1)


def test_paired_units_and_old_sealed_statistics_cannot_be_relabelled():
    batch = SourceBatch(torch.zeros(5, 1), ["p0", "p0", "p1", "p1", "p1"], [0, 0, 1, 1, 1], "P")
    class Panel:
        class system:
            name = "ALDP"
    with pytest.raises(ValueError, match="等大小"):
        _panels_and_sources([Panel()], {"ALDP": batch}, {}, "P")
    with pytest.raises(ValueError, match="统计定义不匹配"):
        _validate_resumed_sources(dict(sealed=True), {}, {})
    blocks = [{"result": {key: {"logweight":torch.zeros(2)} for key in ("baseline", "full", "half")}} for _ in range(2)]
    blocks[0]["result"]["half"]["logweight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="同一完整初态块"):
        _intervention_risk_statistics(blocks)
    shapes = dict(rho0=torch.tensor([0.]), rhoe=torch.tensor([.5]),
                  affinity_integrand=torch.zeros(4), labels=torch.zeros(4))
    blocks = [{"result":dict(baseline={"logweight":torch.zeros(2)}, shape_full=shapes, shape_half=shapes)} for _ in range(2)]
    result = _intervention_shape_statistics(blocks)
    assert math.isnan(result["full"]["point"])
    assert np.isnan(result["full"]["rows"]).all()
