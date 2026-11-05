import math
import numpy as np
import pytest
import torch

from varipath.readouts import estimate
from varipath.records import Budget, check_pool_roles, independent_seed, run_lock


def test_floor_preserves_zero_status_and_signed_task():
    lw = torch.zeros(4, dtype=torch.float64)
    source = dict(a=torch.tensor([True, True, False, False]), b=torch.tensor([False, False, True, True]))
    target = dict(a=torch.tensor([False, False, False, False]), b=torch.tensor([True, True, True, True]))
    a = estimate(lw, source, target, "ALDP")
    b = estimate(lw, source, target, "ALA4")
    assert not a["native_defined"] and a["native_preference"] is None and a["floor_triggered"]
    assert a["preference"] == pytest.approx(math.log(1e4))
    assert b["preference"] == pytest.approx(-.00831446261815324*300*math.log(1e4))
    assert a["endpoint_populations"]["1_a"] == 0
    with pytest.raises(FloatingPointError):
        estimate(torch.full_like(lw, float("nan")), source, target, "ALDP")


def test_independent_role_seeds_no_shared_ladders_and_exclusive_lock(tmp_path):
    keys = [independent_seed(1024, "ALA4", role, method="V") for role in ("D-fit", "D-select", "P", "I", "F", "R")]
    assert len(set(keys)) == len(keys)
    with pytest.raises(ValueError, match="leaks"):
        check_pool_roles(dict(role="P", ladder_ids=["ladder-0"]), dict(role="I", ladder_ids=["ladder-0"]))
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with run_lock(tmp_path):
                pass
    with pytest.raises(TimeoutError):
        budget = Budget(1., previous={"wall_seconds": 2.})
        budget.check()


def test_joint_interval_outputs_serialize_valid_and_undefined_states():
    from varipath.inference import Interval, undefined
    from varipath.workflow import _jsonable
    result = _jsonable(dict(valid=Interval(-.1, .2), missing=undefined("not observed")))
    assert result["valid"]["lower"] == -.1
    assert result["missing"]["lower"] is None and result["missing"]["status"] == "undefined"
