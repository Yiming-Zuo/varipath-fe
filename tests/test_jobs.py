from varipath.jobs import paper_jobs, sensitivity_jobs
from varipath.records import METHODS, read_json, write_json


def job_configs(directory, names=("ALDP", "ALA4")):
    paths = []
    for name in names:
        path = directory/f"{name}.json"
        write_json(path, dict(molecule=name, seed=17, budget_seconds=20,
            experiment=dict(methods=list(METHODS), replicas=128, direct_md_replicas=128)))
        paths.append(path)
    return paths


def test_full_paper_panel_and_actual_auxiliary_budget_configs(tmp_path):
    configs = job_configs(tmp_path/"inputs")
    jobs = paper_jobs(configs, tmp_path/"runs", tmp_path/"prepared")
    assert len(jobs) == 2048
    assert len({(j["molecule"], j["method"], j["replica"]) for j in jobs}) == 2048
    full = paper_jobs(configs, tmp_path/"runs", tmp_path/"prepared", include_endpoint_md=True)
    assert len(full) == 2176 and sum(j["method"] == "endpoint-MD" for j in full) == 128
    half = paper_jobs(configs[:1], tmp_path/"runs", tmp_path/"prepared", .5)
    assert len(half) == 256 and half[0]["budget_multiplier"] == .5
    assert (tmp_path/"runs/configs/ALDP-budget-0.5.json").exists()
    assert read_json(tmp_path/"runs/configs/ALDP-budget-0.5.json")["seed"] != 17


def test_seven_actual_one_factor_settings_are_separate_development_jobs(tmp_path):
    configs = job_configs(tmp_path/"inputs", ("ALA4",))
    jobs = sensitivity_jobs(configs, tmp_path/"D", tmp_path/"prepared")
    assert len(jobs) == 7*16 and len({j["setting"] for j in jobs}) == 7
    configs = [read_json(path) for path in (tmp_path/"D/configs").glob("*.json")]
    assert all(c["mode"] == "development" for c in configs)
    assert len({c["seed"] for c in configs}) == 7
    assert sorted(c["transport"]["spline_count"] for c in configs) == [25, 50, 50, 50, 50, 50, 100]
