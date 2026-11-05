from pathlib import Path
import argparse
import copy
import json
import shlex
import numpy as np

from .records import METHODS, read_json, write_json


def paper_jobs(config_paths, output, preparation_root, budget_multiplier=1., include_endpoint_md=False):
    if budget_multiplier not in (.5, 1., 2.):
        raise ValueError("The paper budgets are B*/2, B*, and 2B*")
    if include_endpoint_md and budget_multiplier != 1:
        raise ValueError("The fixed endpoint-MD diagnostic panel is at the primary budget")
    output, preparation_root = Path(output), Path(preparation_root)
    jobs = []
    for config_path in config_paths:
        config_path = Path(config_path).resolve()
        config = read_json(config_path)
        if config.get("experiment", {}).get("replicas") != 128 or config.get("experiment", {}).get("methods") != list(METHODS):
            raise ValueError("The full panel needs exactly eight methods and 128 independent deployments")
        system = config["molecule"]
        configuration = config_path
        if budget_multiplier != 1:
            configuration = output/"configs"/f"{system}-budget-{budget_multiplier:g}.json"
            changed = copy.deepcopy(config)
            changed["budget_seconds"] = config["budget_seconds"]*budget_multiplier
            changed["budget_ratio"] = budget_multiplier
            changed["primary_seed"] = config["seed"]
            changed["seed"] = int(np.random.SeedSequence([config["seed"], 2026,
                1 if budget_multiplier == .5 else 2]).generate_state(1)[0])
            if configuration.exists():
                if read_json(configuration) != changed:
                    raise ValueError("An auxiliary budget configuration already differs")
            else:
                write_json(configuration, changed)
        replicas = 128 if budget_multiplier == 1 else config["experiment"].get("auxiliary_replicas", 32)
        if replicas != (128 if budget_multiplier == 1 else 32):
            raise ValueError("The current manuscript uses 128 primary and 32 auxiliary deployments")
        for method in METHODS:
            for replica in range(replicas):
                directory = output/f"budget-{budget_multiplier:g}"/system/method/str(replica)
                argv = ["varipath", "run", str(configuration), "--preparation", str(preparation_root/system),
                        "--method", method, "--replica", str(replica), "--output", str(directory)]
                jobs.append(dict(molecule=system, method=method, replica=replica,
                                 budget_multiplier=budget_multiplier, argv=argv,
                                 command=shlex.join(argv), resume_command=shlex.join(argv+["--resume"])))
        if include_endpoint_md and system == "ALA4":
            if config["experiment"].get("direct_md_replicas") != 128:
                raise ValueError("The current endpoint-MD panel has 128 independent deployments")
            for replica in range(128):
                directory = output/"budget-1"/system/"endpoint-MD"/str(replica)
                argv = ["varipath", "endpoint-md", str(configuration), "--preparation", str(preparation_root/system),
                        "--replica", str(replica), "--output", str(directory)]
                jobs.append(dict(molecule=system, method="endpoint-MD", replica=replica,
                                 budget_multiplier=1., argv=argv, command=shlex.join(argv),
                                 resume_command=shlex.join(argv+["--resume"])))
    return jobs


def sensitivity_jobs(config_paths, output, preparation_root):

    output, preparation_root = Path(output), Path(preparation_root)
    settings = [("main", None, None, None), ("splines-25", "transport", "spline_count", 25),
                ("splines-100", "transport", "spline_count", 100),
                ("starts-16", "catalog", "starts", 16), ("starts-64", "catalog", "starts", 64),
                ("centers-2", "transport", "centers", 2), ("centers-8", "transport", "centers", 8)]
    jobs = []
    for config_path in config_paths:
        base = read_json(config_path)
        system = base["molecule"]
        for index, (name, section, key, value) in enumerate(settings):
            config = copy.deepcopy(base)
            config["mode"] = "development"
            config["seed"] = int(np.random.SeedSequence([base["seed"], 2026, 100+index]).generate_state(1)[0])
            config.setdefault("transport", {}).update(spline_count=50, centers=4)
            config.setdefault("catalog", {}).update(starts=32, centers=4)
            if section:
                config[section][key] = value
            config["catalog"]["centers"] = config["transport"]["centers"]
            config["sensitivity"] = dict(setting=name, replicas=16, role="D", formal=False)
            path = output/"configs"/f"{system}-{name}.json"
            if path.exists() and read_json(path) != config:
                raise ValueError("A frozen sensitivity configuration already differs")
            if not path.exists():
                write_json(path, config)
            for replica in range(16):
                argv = ["varipath", "run", str(path), "--preparation", str(preparation_root/system),
                        "--method", "V", "--replica", str(replica), "--output", str(output/system/name/str(replica))]
                jobs.append(dict(molecule=system, setting=name, method="V", replica=replica,
                                 role="D", argv=argv, command=shlex.join(argv),
                                 resume_command=shlex.join(argv+["--resume"])))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate molecular deployment and sensitivity jobs.")
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--budget-multiplier", type=float, default=1.)
    parser.add_argument("--include-endpoint-md", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args(argv)
    if args.sensitivity and (args.include_endpoint_md or args.budget_multiplier != 1):
        parser.error("Sensitivity is a separate fixed development panel")
    jobs = (sensitivity_jobs(args.configs, args.output, args.preparation_root) if args.sensitivity else
            paper_jobs(args.configs, args.output, args.preparation_root, args.budget_multiplier, args.include_endpoint_md))
    print(json.dumps(jobs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
