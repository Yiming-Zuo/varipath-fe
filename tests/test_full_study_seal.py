import json
import pytest

from varipath.records import METHODS
from varipath.workflow import seal_deployments


def test_auxiliary_deployments_are_frozen_before_shared_reference(tmp_path):
    root = tmp_path/"runs"
    config = dict(molecule="ALDP", experiment=dict(methods=list(METHODS), replicas=128,
        auxiliary_replicas=32, direct_md_replicas=128, budget_multipliers=[.5, 1, 2]))
    def write(budget, system, method, replica):
        directory = root/f"budget-{budget}"/system/method/str(replica)
        directory.mkdir(parents=True)
        value = dict(molecule=system, method=method, replica=replica, mode="formal",
            budget_ratio=budget, status="no_legal_output", deployed=None,
            data_unit_ids={"fallback":[f"contract-only:{budget}:{system}:{method}:{replica}"]})
        (directory/"result.json").write_text(json.dumps(value))
    for system in ("ALDP", "ALA4"):
        for method in METHODS:
            for replica in range(128):
                write(1, system, method, replica)
    for replica in range(128):
        write(1, "ALA4", "endpoint-MD", replica)
    registry = tmp_path/"roles.json"
    with pytest.raises(ValueError, match="尚缺 1024"):
        seal_deployments(config, root, registry)
    assert not registry.exists()
    for budget in (.5, 2):
        for system in ("ALDP", "ALA4"):
            for method in METHODS:
                for replica in range(32):
                    write(budget, system, method, replica)
    result = seal_deployments(config, root, registry)
    assert result["deployments"] == 3200
    assert result["main_e3_deployments"] == 2048 and result["auxiliary_e3_deployments"] == 1024
    assert result["endpoint_md_deployments"] == 128
