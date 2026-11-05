from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

from .catalog import physical_potential, search_catalog
from .geometry import MolecularProtection, fit_protection
from .physics import AffineSystem, MolecularSystem
from .readouts import R_KJ, estimate
from .records import Budget, METHODS, independent_seed, read_json, run_lock, save_checkpoint, write_json
from .sampling import RANDOM_STREAM_VERSION, run_sampling, verify_bank_system
from .training import TrainingConfig, train_transport
from .transport import Transport


def load_config(path):
    config = read_json(path) if not isinstance(path, dict) else dict(path)
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象")
    sections = ("sampling", "development", "fallback", "fit", "select", "production",
                "prediction", "intervention", "reference", "transport", "training", "catalog",
                "feat", "mnf", "mbar", "mechanism", "qualification", "endpoint_md",
                "endpoint_md_qualification", "experiment", "runtime", "protection", "diagnostic_inference")
    for section in sections:
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"配置 {section} 必须是 JSON 对象")
    threads = config.get("runtime", {}).get("threads", 1)
    if type(threads) is not int or threads < 1:
        raise ValueError("runtime.threads 必须是正整数")
    if config.get("molecule") not in ("ALDP", "ALA4"):
        raise ValueError("配置必须指定 molecule=ALDP 或 ALA4")
    if config.get("mode", "development") not in ("development", "formal"):
        raise ValueError("部署模式必须是 development 或 formal")
    if type(config.get("seed", 1024)) is not int or config.get("seed", 1024) < 0:
        raise ValueError("主随机种子必须是非负整数")
    seconds = config.get("budget_seconds", 14400)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("冷启动预算必须是有限正数")
    return config


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _setup(config):
    if config.get("runtime", {}).get("device", "cpu") != "cpu":
        raise ValueError("当前完整工作流只验收 CPU 物理路径；不能静默改用其他设备")
    torch.set_num_threads(config.get("runtime", {}).get("threads", 1))
    return MolecularSystem(config["molecule"])


def _contract(output, value, resume):
    value = dict(value, workflow_contract_version=2, random_stream_version=RANDOM_STREAM_VERSION)
    path = Path(output) / "run-config.json"
    if path.exists():
        if not resume:
            raise FileExistsError("运行目录已存在；请显式 --resume 或使用新目录")
        previous = read_json(path)
        if previous.get("workflow_contract_version") != 2 or previous.get("random_stream_version") != RANDOM_STREAM_VERSION:
            raise ValueError("旧工作流或随机流版本不能与当前协议混用；保留旧产物作已有验证，不自动重新起跑")
        if previous != value:
            raise ValueError("恢复配置、方法或重复编号与原运行不一致")
    elif resume:
        raise FileNotFoundError("没有可恢复的运行配置")
    else:
        write_json(path, value)


def _budget(config, output, resume, preparation_charge=0., research=False):
    path = Path(output) / "cost.json"
    previous = read_json(path) if resume and path.exists() else {"wall_seconds": preparation_charge}
    return Budget(config.get("budget_seconds", 14400), previous, research=research)


def _checker(budget, output):
    last_saved = -math.inf
    def check():
        nonlocal last_saved
        budget.check()
        now = time.monotonic()
        if now-last_saved >= 1.:
            write_json(Path(output) / "cost.json", _cost_record(budget))
            last_saved = now
    return check


def _cost_record(budget):
    return dict(budget.record(), checkpoint_interval_seconds=1.,
                interrupt_accounting="resume from last persisted elapsed time; an interrupted operation may be unrecorded")


def _sampling_config(config, stage, role, seed, states=None):
    defaults = dict(count=256, warmup=100, ladders=4, step_size=.05,
                    leapfrog_steps=8, hmc_trials=1, checkpoint_every=10,
                    analysis_block_size=1)
    defaults.update(config.get("sampling", {}))
    if stage.startswith("development."):
        defaults.update(config.get("development", {}).get(stage.split(".", 1)[1], {}))
    else:
        defaults.update(config.get(stage, {}))
    defaults.update(role=role, seed=int(seed), mode=config.get("mode", "development"))
    defaults["random_stream_version"] = RANDOM_STREAM_VERSION
    if states is not None:
        defaults["states"] = list(states)
    return defaults


def _qualify(system, bank, sampling_config, config, enforce_formal=True):
    from .diagnostics import qualify_bank
    criteria = dict(config.get("qualification", {}))
    protocol = dict(criteria.get("protocol", {}))
    protocol.update(kernel="metropolis_hmc_replica_exchange",
                    warmup=sampling_config["warmup"], step_size=sampling_config["step_size"],
                    leapfrog_steps=sampling_config["leapfrog_steps"])
    criteria["protocol"] = protocol
    result = qualify_bank(system, bank, criteria)
    bank["qualification_result"] = result
    bank["eligible_for_formal"] = bool(result.get("eligible_for_formal", False))
    if enforce_formal and config.get("mode") == "formal" and not bank["eligible_for_formal"]:
        raise ValueError("实际采样池未通过正式平衡、混合和数值资格门：" + "; ".join(result.get("reasons", [])))
    return result


def _bank(system, sample_config, directory, config, check, require_qualified=True, extend=False):
    directory = Path(directory)
    settings = directory / "sampling-config.json"
    bank_path = directory / "bank.pt"
    if bank_path.exists():
        if not settings.exists():
            raise ValueError("已完成采样池没有对应的采样配置")
        saved_config = read_json(settings)
        if saved_config != sample_config and not extend:
            raise ValueError("已完成采样池的配置发生变化")
        bank = torch.load(bank_path, weights_only=False)
        bank["frozen_system"] = verify_bank_system(system, bank, directory)
        if bank["positions"].shape[1] != sample_config["count"] or extend:
            bank = run_sampling(system, sample_config, directory, resume=True, check_budget=check, extend=extend)
    else:
        bank = run_sampling(system, sample_config, directory, resume=settings.exists(), check_budget=check)
    check()
    if require_qualified:
        _qualify(system, bank, sample_config, config)
        save_checkpoint(bank_path, bank)
    return bank


def _endpoint(bank, system, endpoint):
    state = float(system.state(endpoint))
    matches = [index for index, value in enumerate(bank["states"]) if math.isclose(value, state, rel_tol=0, abs_tol=1e-10)]
    if len(matches) != 1:
        raise ValueError("采样池必须含一个明确的物理端点状态")
    return bank["positions"][:, :, matches[0]].flatten(0, 1)


def _chain_labels(bank, offset=0):
    ladders, draws = bank["positions"].shape[:2]
    return torch.arange(offset, offset + ladders).repeat_interleave(draws)


def _assert_independent(*banks):
    seen = set()
    for bank in banks:
        units = set(bank["source_unit_ids"])
        if seen & units:
            raise ValueError("独立数据角色或部署复用了同一 ladder")
        seen.update(units)


def _regions(system, points):
    regions = system.regions(points)
    if "valid" in regions and not bool(regions["valid"].all()):
        raise FloatingPointError("区域成员资格不可判定；整批失败")
    return regions


def _floor(config):
    ratio = config.get("budget_ratio", 1.)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("相对主预算的倍数必须为正")
    floor = 1e-4 / ratio
    if not 0 < floor < 1:
        raise ValueError("当前预算的报告地板无效")
    return floor


def _identity_readout(system, source, config, check):
    check()
    with torch.no_grad():
        u0 = _batch_endpoint_u(system, source, 0, config, check)
        u1 = _batch_endpoint_u(system, source, 1, config, check)
        regions = _regions(system, source)
        result = estimate(u0-u1, regions, regions, system.name, _floor(config))
    check()
    return result


def _batch_endpoint_u(system, points, endpoint, config, check):
    batch_size = int(config.get("runtime", {}).get("production_batch_size", 32))
    if batch_size < 1:
        raise ValueError("物理能量计算分块必须为正")
    values = []
    for batch in points.split(batch_size):
        check()
        values.append(torch.vmap(lambda z: system.effective_u(z, endpoint))(batch))
    return torch.cat(values)


def doctor(config):
    config = load_config(config)
    md_platform = config.get("endpoint_md", {}).get("platform", "CPU")
    if "endpoint_md" in config:
        from .endpoint_md import require_platform
        require_platform(md_platform)
    system = _setup(config)
    z = system.initial_coordinates()
    x = system.positions(z)
    oracle = system.oracle()
    errors = []
    try:
        for endpoint in (0, 1):
            s = float(endpoint) if system.name == "ALA4" else 1.
            native, _ = oracle.evaluate(x.detach().numpy(), s)
            errors.append(abs(float(system.energy(x, s))-native))
    finally:
        oracle.close()
    if system.dim != {"ALDP": 48, "ALA4": 123}[system.name] or max(errors) > 1e-6:
        raise RuntimeError("物理接口维数或原生 OpenMM 能量核对失败")
    return dict(status="passed", molecule=system.name, shape_dimension=system.dim,
                random_stream_version=RANDOM_STREAM_VERSION,
                energy_error_kjmol=max(errors), device="cpu", precision="float64",
                endpoint_md_platform=md_platform,
                versions={name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "openmm", "pymbar", "ase")},
                checks=["shape_dimension", "endpoint_energies"])


def prepare(config, output, resume=False):
    config, output = load_config(config), Path(output)
    with run_lock(output):
        _contract(output, dict(action="prepare", config=config), resume)
        completed = output / "prepared.pt"
        if completed.exists():
            bank_path = output/"fit/bank.pt"
            if not bank_path.is_file():
                raise ValueError("准备材料缺少原 D-fit 银行的物理冻结证据，只能读取旧输出")
            current_system = _setup(config)
            fit_bank = torch.load(bank_path, weights_only=False)
            verify_bank_system(current_system, fit_bank, bank_path.parent)
            return read_json(output / "result.json")
        budget = _budget(config, output, resume)
        check = _checker(budget, output)
        system = _setup(config)
        try:
            banks = []
            for stage, role in (("fit", "D-fit"), ("select", "D-select")):
                seed = independent_seed(config.get("seed", 1024), system.name, role)
                sampling = _sampling_config(config, "development." + stage, role, seed)
                with budget.phase("D-" + stage):
                    banks.append(_bank(system, sampling, output / stage, config, check, require_qualified=False))
            _assert_independent(*banks)
            source = _endpoint(banks[0], system, 0)
            location = source.mean(0)
            scale = source.std(0, unbiased=False).clamp_min(config.get("development", {}).get("minimum_scale", .05))
            standardized = AffineSystem(system, location, scale)
            protection_config = dict(config.get("protection", {}))
            with budget.phase("protection"):
                protection = fit_protection(standardized, (source-location)/scale, **protection_config)
                heldout = ( _endpoint(banks[1], system, 0)-location)/scale
                heldout_values = protection(heldout)
                heldout_coverage = dict(plateau=float((heldout_values == 1).double().mean()),
                                        transition=float(((heldout_values > 0)&(heldout_values < 1)).double().mean()),
                                        identity=float((heldout_values == 0).double().mean()),
                                        sample_count=len(heldout))
            specification = protection.specification()
            specification.pop("lipschitz", None)
            check()
            artifact = dict(molecule=system.name, location=location, scale=scale,
                            protection=specification, config=config, cost=_cost_record(budget),
                            data_roles=[bank["source_unit_ids"] for bank in banks])
            result = dict(status="prepared", molecule=system.name, shape_dimension=system.dim,
                          training_data_reused=False, eligible_for_formal=False,
                          protection=protection.specification(), preparation_cost=_cost_record(budget),
                          independent_D_holdout_coverage=heldout_coverage,
                          qualification="ordinary D geometry only; every deployment samples its own data")
            write_json(output / "result.json", result)
            save_checkpoint(completed, artifact)
            return result
        finally:
            write_json(output / "cost.json", _cost_record(budget))


def sample(config, role, output, resume=False, look=1):
    config, output = load_config(config), Path(output)
    if role not in ("P", "I", "R"):
        raise ValueError("独立采样角色必须是 P、I 或 R")
    if type(look) is not int or look not in (1, 2, 3, 4) or (role != "R" and look != 1):
        raise ValueError("只有 R 可以选择固定的四次查看长度")
    if role == "R" and look > 1 and (output/"run-config.json").exists():
        resume = True
    with run_lock(output):
        _contract(output, dict(action="sample", role=role, config=config), resume)
        previous = read_json(output/"result.json") if (output/"result.json").exists() else None
        if previous is not None:
            previous_look = int(previous.get("look", 1))
            if role != "R" or look == previous_look:
                if role == "R":
                    actual = torch.load(output/"sampling/bank.pt", weights_only=False)["positions"].shape[1]
                    if actual != previous.get("production_draws_per_ladder", actual):
                        raise ValueError("R 银行已增长，旧终止记录不能冒充当前查看；请继续相应 --look")
                return previous
            if look < previous_look:
                raise ValueError("R 已进入更长的保存长度，不能把较新银行冒充此前查看")
        stop_path = output/"sampling/reference-status.json"
        stop = read_json(stop_path) if stop_path.exists() else {}
        if role == "R" and (stop.get("final") or stop.get("stopped")):
            raise ValueError("R 已停止或标为 final，不能延长到下一次查看")
        system = _setup(config)
        stage = {"P": "prediction", "I": "intervention", "R": "reference"}[role]
        research_seconds = config.get(stage, {}).get("budget_seconds", config.get("budget_seconds", 14400))
        budget = _budget(dict(config, budget_seconds=research_seconds), output, resume, research=True)
        check = _checker(budget, output)
        seed = independent_seed(config.get("seed", 1024), system.name, role)
        sampling = _sampling_config(config, stage, role, seed)
        base_count = sampling["count"]
        if role == "R":
            sampling["count"] = base_count*2**(look-1)
        settings_path = output/"sampling/sampling-config.json"
        saved_sampling = read_json(settings_path) if settings_path.exists() else None
        extend = role == "R" and saved_sampling is not None and sampling["count"] > saved_sampling["count"]
        try:
            with budget.phase("sample-" + role):
                bank = _bank(system, sampling, output / "sampling", config, check,
                             require_qualified=role != "R", extend=extend)
                if role == "R":
                    _qualify(system, bank, sampling, config, enforce_formal=False)
                    bank["reference_base_count"] = base_count
                    save_checkpoint(output/"sampling/bank.pt", bank)
                    progress_path = output/"sampling/sampling-progress.json"
                    progress = read_json(progress_path)
                    progress["base_count"] = base_count
                    write_json(progress_path, progress)
            result = dict(status="completed_sampling", role=role, molecule=system.name,
                          eligible_for_formal=bank["eligible_for_formal"],
                          qualification=bank["qualification_result"], independent_units=bank["source_unit_ids"])
            if role == "R":

                result["qualification"] = dict(status="sealed_diagnostics", reference_summary_required=True)
                result["eligible_for_formal"] = False
                result["reference_access"] = "sealed; use the explicit reference command after P, I and F are sealed"
                result.update(look=look, reference_base_count=base_count,
                              production_draws_per_ladder=sampling["count"], sampling_elapsed_seconds=bank["elapsed_seconds"])
            write_json(output / "result.json", _jsonable(result))
            return _jsonable(result)
        finally:
            write_json(output / "cost.json", _cost_record(budget))


class _WindowSystem(nn.Module):

    def __init__(self, base, physical_states, windows):
        super().__init__()
        self.base = base
        self.name, self.dim, self.metadata, self.chart = base.name, base.dim, base.metadata, base.chart

        if hasattr(base, "scale"):
            self.system = base.system
            self.register_buffer("scale", base.scale.clone())
        self.register_buffer("physical_states", torch.tensor(physical_states, dtype=torch.float64))
        encoded = np.linspace(0, 1, len(windows)) if self.name == "ALA4" else np.arange(1, len(windows)+1, dtype=float)
        self.encoded = encoded.tolist()
        self.register_buffer("encoded_states", torch.tensor(encoded, dtype=torch.float64))
        self.windows = windows
        self.register_buffer("window_definition", torch.tensor(list(json.dumps(windows, sort_keys=True).encode("utf-8")), dtype=torch.uint8))
        if any(windows[j].get("energy") or windows[j].get("core") for j in (0, len(windows)-1)):
            raise ValueError("EB-MBAR 两个物理端态窗口须无偏置，偏置放在中间窗口")

    def _index(self, state):
        return self.encoded.index(float(state))

    def _bias(self, z, index):
        from .baselines.mbar import PeriodicCoreBias
        spec = self.windows[index]
        value = z.new_zeros(())
        if "energy" in spec:
            coordinate = physical_potential(self.base, z, float(self.physical_states[index])) / (R_KJ*300)
            energy = spec["energy"]
            value = value + .5*energy["stiffness"]*(coordinate-energy["center"])**2
        if "core" in spec:
            core = spec["core"]
            bias = PeriodicCoreBias(lambda point: torch.deg2rad(self.base.regions(point)["angles"]),
                                    core["centers"], core["width"], core["strength"])
            value = value + bias(z)
        return value

    def state(self, endpoint):
        physical = float(self.base.state(endpoint))
        return self.encoded[self.physical_states.tolist().index(physical)]

    def positions(self, z):
        return self.base.positions(z)

    def initial_coordinates(self, index=0):
        return self.base.initial_coordinates(index)

    def in_domain(self, z):
        return self.base.in_domain(z)

    def regions(self, z):
        return self.base.regions(z)

    def unrestricted_u_state(self, z, state):
        index = self._index(state)
        return self.base.unrestricted_u_state(z, float(self.physical_states[index])) + self._bias(z, index)

    def effective_u_state(self, z, state):
        index = self._index(state)
        return self.base.effective_u_state(z, float(self.physical_states[index])) + self._bias(z, index)


def _window_readout(system, bank, config, check):
    from .baselines.mbar import fit_mbar
    points = bank["positions"].permute(2, 0, 1, 3).flatten(0, 2)
    cross = bank["u_kn"].permute(2, 3, 0, 1).flatten(1)
    count = bank["positions"].shape[0]*bank["positions"].shape[1]
    batch_size = int(config.get("runtime", {}).get("production_batch_size", 32))
    if batch_size < 1:
        raise ValueError("MBAR 读出的物理计算分块必须为正")
    chunks = [_regions(system, batch) for batch in points.split(batch_size)]
    flags = {key:torch.cat([chunk[key] for chunk in chunks]) for key in ("a", "b", "fine")}
    cells = 25 if system.name == "ALA4" else 5
    indicators = torch.stack([flags["a"], flags["b"]] + [flags["fine"] == j for j in range(cells)], -1)
    endpoint_u = torch.stack([_batch_endpoint_u(system, points, endpoint, config, check) for endpoint in (0, 1)])
    check()
    result = fit_mbar(cross, [count]*len(bank["states"]), endpoint_u=endpoint_u,
                      region_flags=indicators, min_overlap=config.get("mbar", {}).get("min_overlap", 1e-8))
    check()
    logs = np.asarray(result["endpoint_log_probabilities"])[:, :2]
    stable = np.maximum(logs, math.log(_floor(config)))
    odds = float((stable[1, 1]-stable[1, 0])-(stable[0, 1]-stable[0, 0]))
    units = R_KJ*300 if system.name == "ALA4" else 1.
    native_defined = bool(np.isfinite(logs).all())
    native_odds = float((logs[1, 1]-logs[1, 0])-(logs[0, 1]-logs[0, 0])) if native_defined else None
    sign = -units if system.name == "ALA4" else 1.
    return dict(delta_f=result["endpoint_delta_f"], delta_free_energy=units*result["endpoint_delta_f"],
                preference=sign*odds, native_preference=None if native_odds is None else sign*native_odds,
                native_defined=native_defined, reporting_floor=_floor(config),
                floor_triggered=bool((logs < math.log(_floor(config))).any()),
                endpoint_populations={f"{endpoint}_{name}":result["endpoint_probabilities"][endpoint][j] for endpoint in (0, 1) for j, name in enumerate(("a", "b"))},
                fine_populations={str(endpoint):result["endpoint_probabilities"][endpoint][2:] for endpoint in (0, 1)},
                native_mbar=result, units="kJ/mol" if system.name == "ALA4" else "dimensionless")


def _generic_centers(points, count, seed, bandwidth_pool=None):
    if count < 1 or len(points) < count:
        raise ValueError("通用中心数量超过本方已付费拟合池的容量")
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(points)))]
    distances = (points-points[chosen[0]]).square().sum(-1)
    for _ in range(1, count):
        distances[chosen] = -1
        selected = int(distances.argmax())
        chosen.append(selected)
        distances = torch.minimum(distances, (points-points[selected]).square().sum(-1))
    centers = points[chosen].detach().clone()
    bandwidth_pool = points if bandwidth_pool is None else bandwidth_pool
    widths = torch.stack([torch.quantile((bandwidth_pool-center).norm(dim=-1), .1) for center in centers])
    if not (widths > 0).all():
        raise ValueError("本方通用中心的第十分位带宽退化")
    return centers, widths, torch.full((count,), .5, dtype=torch.float64)


def _make_transport(system, protection, config, centers=None, widths=None, amplitudes=None, time_centers=None, seed=None):
    t = config.get("transport", {})
    model = Transport(system.dim, protection, spline_count=t.get("spline_count", 50), steps=t.get("steps"),
                      centers=centers, bandwidths=widths, amplitudes=amplitudes, time_centers=time_centers,
                      seed=config.get("seed", 1024) if seed is None else seed)
    maximum = config.get("runtime", {}).get("max_transport_steps")
    if maximum is not None and model.n_steps > maximum:
        raise RuntimeError(f"认证网格需要 {model.n_steps} 步，超过配置资源上限 {maximum}；未下调步数")
    return model


def _fit_transport(model, system, fit, select, config, output, check):
    settings = dict(config.get("training", {}))
    names = {field.name for field in fields(TrainingConfig)}
    if set(settings) - names:
        raise ValueError("未知的训练配置字段：" + ", ".join(sorted(set(settings)-names)))
    path = Path(output)
    return train_transport(model, _endpoint(fit, system, 0), _endpoint(select, system, 0),
                           lambda z: system.effective_u(z, 0), lambda z: system.effective_u(z, 1),
                           TrainingConfig(**settings), checkpoint=path, resume=path.exists(),
                           train_chain_ids=_chain_labels(fit), validation_chain_ids=_chain_labels(select, 1000000),
                           scale_potential=[
                               lambda z: physical_potential(system, z, 0. if system.name == "ALA4" else 1.)/(R_KJ*(550. if system.name == "ALDP" else 300.)),
                               lambda z: physical_potential(system, z, 1.)/(R_KJ*300.),
                           ], should_stop=lambda: (check() or False))


def _transport_readout(model, system, source, config, check):
    with torch.no_grad():
        batch_size = int(config.get("runtime", {}).get("production_batch_size", 32))
        if batch_size < 1:
            raise ValueError("生产计算分块须为正；分块不改变预定生产样本数")
        exposure = source.new_zeros(len(model.centers))
        fractions = source.new_zeros(3)
        outputs, determinants, weights = [], [], []
        for batch in source.split(batch_size):
            y, logdet = batch, batch.new_zeros(len(batch))
            for k in range(model.n_steps):
                check()
                b = model.protection(y)
                fractions += torch.stack([(b == 0).double().sum(), ((b > 0)&(b < 1)).double().sum(), (b == 1).double().sum()])
                if len(model.centers):
                    centers = model.centers_at(k)
                    psi = torch.exp(-(y[:, None]-centers).square().sum(-1)/(2*model.bandwidths.square()))
                    exposure += (b[:, None]*psi).sum(0)
                y, term = model.step(y, k)
                logdet += term
            loga = torch.vmap(lambda z: system.effective_u(z, 0))(batch)-torch.vmap(lambda z: system.effective_u(z, 1))(y)+logdet
            outputs.append(y)
            determinants.append(logdet)
            weights.append(loga)
        y, logdet, loga = torch.cat(outputs), torch.cat(determinants), torch.cat(weights)
        result = estimate(loga, _regions(system, source), _regions(system, y), system.name, _floor(config))
        regions = _regions(system, y)
        result.update(proposal_populations={key:float(regions[key].double().mean()) for key in ("a", "b", "remainder")},
                      anchor_exposure=(exposure/(model.n_steps*len(source))).tolist(), protection_fractions=(fractions/(model.n_steps*len(source))).tolist(),
                      frozen_strengths=model.alpha.detach().tolist())
    check()
    return result, dict(source=source, target=y, log_weights=loga, ordinary_logdet=logdet)


def _bridge_readout(system, result):
    readout = result["readout"]
    solved = result.get("native_nebar", result.get("native_bar"))
    units = R_KJ*300 if system.name == "ALA4" else 1.
    sign = units if system.name == "ALA4" else -1.
    raw = readout["raw_dimensionless_preference"]
    return dict(delta_f=solved["dimensionless_free_energy"], delta_free_energy=units*solved["dimensionless_free_energy"],
                preference=sign*readout["reported_dimensionless_preference"],
                native_preference=None if raw is None else sign*raw, native_defined=raw is not None,
                floor_triggered=readout["floor_triggered"], reporting_floor=readout["probability_floor"],
                endpoint_populations={f"{endpoint}_{region}":readout["endpoint_region_probabilities"][endpoint][j] for endpoint in (0, 1) for j, region in enumerate(("a", "b"))},
                native_estimator=solved, bridge_diagnostics=readout,
                units="kJ/mol" if system.name == "ALA4" else "dimensionless")


def run_deployment(config, preparation, method, replica, output, resume=False):
    config, output = load_config(config), Path(output)
    if method not in METHODS or type(replica) is not int or replica < 0:
        raise ValueError("方法名称或独立部署编号无效")
    prepared = torch.load(Path(preparation)/"prepared.pt", weights_only=False)
    if prepared["molecule"] != config["molecule"]:
        raise ValueError("准备材料与部署物理体系不一致")
    with run_lock(output):
        _contract(output, dict(action="run", config=config, method=method, replica=replica,
                               preparation_config=prepared["config"]), resume)
        if (output / "result.json").exists():
            return read_json(output / "result.json")
        preparation_cost_path = Path(preparation)/"cost.json"
        preparation_charge = float((read_json(preparation_cost_path) if preparation_cost_path.exists() else prepared["cost"])["wall_seconds"])
        budget = _budget(config, output, resume, preparation_charge)
        check = _checker(budget, output)
        raw = _setup(config)
        system = AffineSystem(raw, prepared["location"], prepared["scale"])
        protection = MolecularProtection(system, **prepared["protection"])
        phase_role = "F" if config.get("mode") == "formal" else "D-fit"
        seed = independent_seed(config.get("seed", 1024), system.name, phase_role, replica, method)
        result = dict(method=method, replica=replica, molecule=system.name, mode=config.get("mode", "development"),
                      random_stream_version=RANDOM_STREAM_VERSION,
                      preparation_cold_start_charge_seconds=preparation_charge, eligible_for_formal=False,
                      budget_ratio=config.get("budget_ratio", 1.), budget_seconds=config.get("budget_seconds", 14400),
                      execution_unit=dict(device="cpu", cpu_threads=torch.get_num_threads(),
                                          logical_cpu_count=os.cpu_count(), processor=platform.processor() or platform.machine(),
                                          machine=platform.machine(), system=platform.system()))
        fallback = None
        try:

            with budget.phase("paid-identity-fallback"):
                fb_cfg = _sampling_config(config, "fallback", "fallback", seed+1)
                fb_bank = _bank(system, fb_cfg, output/"fallback", config, check)
                fallback = _identity_readout(system, _endpoint(fb_bank, system, 0), config, check)
                write_json(output/"fallback.json", fallback)
            try:
                if method in ("E-MBAR", "EB-MBAR"):
                    sampler_system = system
                    states = config.get("mbar", {}).get("states", config.get("sampling", {}).get("states"))
                    states = states or ([300.,338.7,382.4,431.8,487.5,550.] if system.name == "ALDP" else np.linspace(0, 1, 17).tolist())
                    if method == "EB-MBAR":
                        windows = config.get("mbar", {}).get("biased_windows")
                        if windows is None or len(windows) != len(states) or not any(window for window in windows):
                            raise ValueError("EB-MBAR 必须配置并实际采样非零已知偏置窗口")
                        sampler_system = _WindowSystem(system, states, windows)
                        states = sampler_system.encoded
                    p_cfg = _sampling_config(config, "production", "F-production", seed+4, states)
                    with budget.phase("native-window-production"):
                        production = _bank(sampler_system, p_cfg, output/"production", config, check)
                        _assert_independent(fb_bank, production)
                        primary = _window_readout(system, production, config, check)
                    diagnostics = dict(sampling_qualification=production["qualification_result"],
                                       actual_biased_windows=config.get("mbar", {}).get("biased_windows") if method == "EB-MBAR" else None)
                else:
                    banks = []
                    for stage, role, offset in (("fit", "F-fit", 2), ("select", "F-select", 3)):
                        sample_cfg = _sampling_config(config, stage, role, seed+offset)
                        with budget.phase("sample-" + role):
                            banks.append(_bank(system, sample_cfg, output/stage, config, check))
                    fitting, selection = banks
                    _assert_independent(fb_bank, fitting, selection)
                    source = _endpoint(fitting, system, 0)
                    if method in ("V", "V0", "G", "MG"):
                        centers = widths = amplitudes = time_centers = None
                        count = config.get("transport", {}).get("centers", 4)
                        if method == "V":
                            cat_cfg = dict(config.get("catalog", {}), seed=seed+5, centers=count)
                            starts_count = cat_cfg.get("starts", 32)
                            if len(source) < starts_count:
                                raise ValueError("本方拟合源池不足以提供预定鞍点起点")
                            generator = torch.Generator().manual_seed(seed+5)
                            starts = source[torch.randperm(len(source), generator=generator)[:starts_count]]
                            with budget.phase("dimer-catalog"):
                                if (output/"catalog.json").exists():
                                    catalog = read_json(output/"catalog.json")
                                else:
                                    catalog = search_catalog(system, starts, source, protection, cat_cfg,
                                        checkpoint_callback=lambda rows: (save_checkpoint(output/"catalog-progress.pt", rows), check()),
                                        check_budget=check,
                                        initial_records=torch.load(output/"catalog-progress.pt", weights_only=False) if (output/"catalog-progress.pt").exists() else None)
                                catalog.update(mode=config.get("mode", "development"), molecule=system.name,
                                               fitting_source_units=fitting["source_unit_ids"], method=method, replica=replica)
                                write_json(output/"catalog.json", _jsonable(catalog))
                            selected = catalog["selected"]
                            if not catalog.get("complete") or len(selected) != count:
                                raise ValueError("本次付费鞍点目录未取得预定数量的合格中心；主程序整批失败")
                            if selected:
                                centers = torch.tensor([row["z"] for row in selected], dtype=torch.float64)
                                widths = torch.tensor([row["sigma"] for row in selected], dtype=torch.float64)
                                amplitudes = torch.tensor([row["amplitude"] for row in selected], dtype=torch.float64)
                        elif method in ("G", "MG"):
                            eligible_centers = source[protection(source) >= 1-1e-12]
                            centers, widths, amplitudes = _generic_centers(eligible_centers, count, seed+5, bandwidth_pool=source)
                            if not bool((protection(centers) >= 1-1e-12).all()):
                                raise ValueError("通用中心未全部位于共同保护平台")
                            if method == "MG":
                                with budget.phase("owned-moving-center-pretraining"):
                                    from .centers import freeze_centers
                                    full_grid = _make_transport(system, protection, config, centers, widths, amplitudes, seed=seed+7).n_steps
                                    pretrain_config = dict(config, transport=dict(config.get("transport", {}), steps=full_grid))
                                    pretrain = _make_transport(system, protection, pretrain_config, seed=seed+8)
                                    _fit_transport(pretrain, system, fitting, selection, pretrain_config, output/"pretrain.pt", check)
                                    folder = output/"moving-centers"
                                    time_centers = freeze_centers(pretrain, centers, folder, run_root=output,
                                        chunk_steps=config.get("runtime", {}).get("center_chunk_steps", 128),
                                        resume=(folder/"progress.pt").exists(), check_budget=check)
                        with budget.phase("joint-transport-training"):
                            model = _make_transport(system, protection, config, centers, widths, amplitudes, time_centers, seed=seed+7)
                            training = _fit_transport(model, system, fitting, selection, config, output/"training.pt", check)
                            save_checkpoint(output/"model.pt", dict(state_dict=model.state_dict(), steps=model.n_steps,
                                            centers=model.centers, bandwidths=model.bandwidths, amplitudes=model.amplitudes,
                                            time_centers=model.time_centers, center_provider=model.center_provider_descriptor(),
                                            center_provider_source_state=model.center_provider_source_state(),
                                            training=training, certificate=model.bound_certificate))
                        diagnostics = dict(training=training, map_certificate=model.bound_certificate)
                    else:
                        if method == "FEAT":
                            from .baselines.feat import FEAT, FEATConfig
                            model = FEAT(system, FEATConfig(**dict(config.get("feat", {}), seed=seed+5)))
                        else:
                            from .baselines.mnf import MNFConfig, MNFIC
                            model = MNFIC(system, MNFConfig(**dict(config.get("mnf", {}), seed=seed+5)))
                        with budget.phase("native-transport-training"):
                            path = output/"native-training.pt"
                            training = model.fit(source, _endpoint(fitting, system, 1), checkpoint=path,
                                                 resume=path.exists(), check_budget=check)
                        diagnostics = dict(training=training, selection_data="reserved; native training uses its fixed native schedule")
                    p_cfg = _sampling_config(config, "production", "F-production", seed+4)
                    with budget.phase("complete-independent-production"):
                        production = _bank(system, p_cfg, output/"production", config, check)
                        _assert_independent(fb_bank, fitting, selection, production)
                        p0, p1 = _endpoint(production, system, 0), _endpoint(production, system, 1)
                        if method in ("V", "V0", "G", "MG"):
                            primary, arrays = _transport_readout(model, system, p0, config, check)
                            save_checkpoint(output/"production-output.pt", arrays)
                        else:
                            kwargs = dict(eta=_floor(config), min_overlap=config.get("mbar", {}).get("min_overlap", 1e-8),
                                          check_budget=check, batch_size=config.get("runtime", {}).get("production_batch_size", 32))
                            if method == "FEAT":
                                kwargs["generator"] = torch.Generator().manual_seed(seed+6)
                            native = model.evaluate(p0, p1, **kwargs)
                            primary = _bridge_readout(system, native)
                            save_checkpoint(output/"production-output.pt", native)
                check()
                result.update(status="completed", fallback_used=False, native=primary, deployed=primary,
                              diagnostics=diagnostics, eligible_for_formal=config.get("mode") == "formal")
            except (ValueError, RuntimeError, FloatingPointError, TimeoutError, np.linalg.LinAlgError) as exc:
                result.update(status="fallback", fallback_used=True, primary_failure=dict(type=type(exc).__name__, reason=str(exc)),
                              native=None, deployed=fallback,
                              eligible_for_formal=config.get("mode") == "formal" and fb_bank["eligible_for_formal"])
        except (ValueError, RuntimeError, FloatingPointError, TimeoutError, np.linalg.LinAlgError) as exc:
            result.update(status="no_legal_output", fallback_used=False, native=None, deployed=None,
                          failure=dict(type=type(exc).__name__, reason=str(exc)), eligible_for_formal=False)
        finally:
            write_json(output/"cost.json", _cost_record(budget))
        result["data_unit_ids"] = {}
        for stage in ("fallback", "fit", "select", "production"):
            bank_path = output/stage/"bank.pt"
            result["data_unit_ids"][stage] = torch.load(bank_path, weights_only=False)["source_unit_ids"] if bank_path.exists() else []
        result["cost"] = _cost_record(budget)
        write_json(output/"result.json", _jsonable(result))
        return _jsonable(result)


def _preference_only(readout):
    if readout is None:
        return None
    keep = ("preference", "native_preference", "native_defined", "endpoint_populations",
            "reporting_floor", "floor_triggered", "units", "native_zero_hit", "sample_counts")
    result = {key:readout[key] for key in keep if key in readout}
    result.update(delta_f=None, delta_free_energy=None,
                  free_energy_status="not_an_estimand_of_endpoint_MD")
    return result


def run_md(config, preparation, replica, output, resume=False):

    from .endpoint_md import require_platform, run_endpoint_md
    config, output = load_config(config), Path(output)
    if config["molecule"] != "ALA4" or type(replica) is not int or replica < 0:
        raise ValueError("直接端态 MD 只接受 ALA4 和非负独立部署编号")
    md_platform = config.get("endpoint_md", {}).get("platform", "CPU")
    require_platform(md_platform)
    prepared = torch.load(Path(preparation)/"prepared.pt", weights_only=False)
    if prepared["molecule"] != "ALA4":
        raise ValueError("端态 MD 准备材料须来自当前 52 原子 ALA4")
    with run_lock(output):
        _contract(output, dict(action="endpoint-md", config=config, replica=replica,
                               preparation_config=prepared["config"]), resume)
        if (output/"result.json").exists():
            return read_json(output/"result.json")
        cost_path = Path(preparation)/"cost.json"
        charge = float((read_json(cost_path) if cost_path.exists() else prepared["cost"])["wall_seconds"])
        budget = _budget(config, output, resume, charge)
        check = _checker(budget, output)
        system = _setup(config)
        phase_role = "F" if config.get("mode") == "formal" else "D-fit"
        seed = independent_seed(config.get("seed", 1024), "ALA4", phase_role, replica, chain=9000)
        md_seed = independent_seed(config.get("seed", 1024), "ALA4", phase_role, replica, chain=8753)
        result = dict(method="endpoint-MD", molecule="ALA4", replica=replica,
                      random_stream_version=RANDOM_STREAM_VERSION,
                      mode=config.get("mode", "development"), budget_ratio=config.get("budget_ratio", 1.),
                      budget_seconds=config.get("budget_seconds", 14400), preparation_cold_start_charge_seconds=charge,
                      execution_unit=dict(device="cpu+cuda" if md_platform == "CUDA" else "cpu", cpu_threads=torch.get_num_threads(), logical_cpu_count=os.cpu_count(),
                                          processor=platform.processor() or platform.machine(), machine=platform.machine(),
                                          system=platform.system(), native_md_platform=config.get("endpoint_md", {}).get("platform", "CPU")),
                      eligible_for_formal=False, native=None, deployed=None, fallback_used=False,
                      data_unit_ids=dict(fallback=[], production=[]), delta_f=None, delta_free_energy=None)
        fallback = None
        native_result = None
        try:
            with budget.phase("paid-identity-fallback"):
                sample_config = _sampling_config(config, "fallback", "fallback", seed)
                fallback_bank = _bank(system, sample_config, output/"fallback", config, check)
                result["data_unit_ids"]["fallback"] = fallback_bank["source_unit_ids"]
                fallback = _identity_readout(system, _endpoint(fallback_bank, system, 0), config, check)
                write_json(output/"fallback.json", _preference_only(fallback))
            with budget.phase("native-endpoint-md"):
                native_result = run_endpoint_md(dict(config, replica=replica, seed=md_seed), output/"native-md", resume=(output/"native-md/endpoint-md-config.json").exists(),
                    preparation=dict(system=system, budget=budget, fallback=fallback,
                                     fallback_qualified=fallback_bank["eligible_for_formal"], fallback_within_same_budget=True),
                    check_budget=check)
            primary = native_result.get("status") in ("completed", "completed_development")
            result.update(status="completed" if primary else "fallback" if native_result.get("output") is not None else "no_legal_output",
                          native=_preference_only(native_result.get("native")) if primary else None,
                          deployed=_preference_only(native_result.get("output")),
                          fallback_used=bool(native_result.get("fallback_triggered")),
                          eligible_for_formal=bool(native_result.get("eligible_for_formal")),
                          diagnostics=native_result)
        except (ValueError, RuntimeError, FloatingPointError, TimeoutError) as exc:
            legal_fallback = fallback is not None
            result.update(status="fallback" if legal_fallback else "no_legal_output", fallback_used=legal_fallback,
                          native=None, deployed=_preference_only(fallback),
                          eligible_for_formal=bool(legal_fallback and config.get("mode") == "formal" and fallback_bank["eligible_for_formal"]),
                          primary_failure=dict(type=type(exc).__name__, reason=str(exc)))
        finally:
            write_json(output/"cost.json", _cost_record(budget))
        units = set((native_result or {}).get("source_unit_ids", []))
        if (output/"fallback/bank.pt").exists():
            result["data_unit_ids"]["fallback"] = torch.load(output/"fallback/bank.pt", weights_only=False)["source_unit_ids"]
        for path in (output/"native-md").glob("endpoint-*-ladder-*.pt"):
            record = torch.load(path, weights_only=False)
            if record.get("unit_id"):
                units.add(record["unit_id"])
        result["data_unit_ids"]["production"] = sorted(units)
        result["cost"] = _cost_record(budget)
        write_json(output/"result.json", _jsonable(result))
        return _jsonable(result)


def prepare_mechanism(config, preparation, deployment, output, catalog_path=None, resume=False):
    from .mechanism import prepare_panel
    config, output = load_config(config), Path(output)
    deployment_result = read_json(Path(deployment)/"result.json")
    if deployment_result.get("mode") != "development":
        raise ValueError("E1 冻结基流只能来自专用 D 开发运行，不能读取正式 E3 的 F 部署")
    if deployment_result.get("molecule") != config["molecule"]:
        raise ValueError("E1 配置与开发基流的分子体系不同")
    with run_lock(output):
        _contract(output, dict(action="mechanism-prepare", config=config,
                               deployment=str(Path(deployment).resolve()),
                               catalog=None if catalog_path is None else str(Path(catalog_path).resolve())), resume)
        if (output/"panel.pt").exists():
            _require_panel_schema(torch.load(output/"panel.pt", weights_only=False))
            return read_json(output/"result.json")
        prepared = torch.load(Path(preparation)/"prepared.pt", weights_only=False)
        if prepared["molecule"] != config["molecule"]:
            raise ValueError("E1 准备材料与配置的分子体系不同")
        system = AffineSystem(_setup(config), prepared["location"], prepared["scale"])
        protection = MolecularProtection(system, **prepared["protection"])
        saved = torch.load(Path(deployment)/"model.pt", weights_only=False)
        time_centers = saved["time_centers"]
        if saved.get("center_provider") is not None:
            from .centers import DiskCenters
            time_centers = DiskCenters.from_descriptor(saved["center_provider"], Path(deployment),
                expected_source_state=saved["center_provider_source_state"])
        model_config = dict(config, transport=dict(config.get("transport", {}), steps=saved["steps"]))
        model = _make_transport(system, protection, model_config, saved["centers"], saved["bandwidths"],
                                saved["amplitudes"], time_centers)
        for key, value in model.state_dict().items():
            if key.startswith("protection.") and (key not in saved["state_dict"] or not torch.equal(value, saved["state_dict"][key])):
                raise ValueError("冻结基流的物理势、坐标或保护几何与准备材料不一致")
        model.load_state_dict(saved["state_dict"])
        with torch.no_grad():
            model.alpha.zero_()
        catalog = read_json(catalog_path or Path(deployment)/"catalog.json")
        own_catalog = (Path(deployment)/"catalog.json").resolve()
        inferred_mode = "development" if Path(catalog_path or own_catalog).resolve() == own_catalog else None
        if catalog.get("mode", inferred_mode) != "development":
            raise ValueError("E1 鞍点目录必须来自独立 D 开发数据，不能读取正式 F 目录")
        bank = torch.load(Path(preparation)/"fit/bank.pt", weights_only=False)
        samples = {name: (_endpoint(bank, system, endpoint)-prepared["location"])/prepared["scale"]
                   for name, endpoint in (("source", 0), ("target", 1))}
        panel = prepare_panel(model, system, protection, catalog["records"], samples, config.get("mechanism", {}))
        result = dict(status="prepared", molecule=system.name, cells=len(panel.cells),
                      development=_jsonable(panel.development), eligible_for_formal=False)
        write_json(output/"result.json", result)
        save_checkpoint(output/"panel.pt", panel)
        return result


def _mechanism_banks(panels, bank_paths):
    banks = {}
    for panel in panels:
        name = panel.system.name
        if name not in bank_paths:
            raise ValueError(f"缺少 {name} 的独立源银行")
        bank = torch.load(bank_paths[name], weights_only=False)
        bank = dict(bank)
        system = panel.system
        if hasattr(system, "raw"):
            bank["positions"] = (bank["positions"]-system.location)/system.scale
        banks[name] = bank
    return banks


def _require_panel_schema(panel):
    if panel.development.get("geometry_certificate_schema") != "interval_heavy_basin_v1":
        raise ValueError("旧 E1 几何证书版本不能按当前算法恢复；保留旧结果，使用新的 D 面板与 P/I 输出目录")


def predict(config, panel_paths, bank_paths, output, resume=False):
    from .mechanism import run_predictions
    config = load_config(config)
    panels = [torch.load(path, weights_only=False) for _, path in sorted(panel_paths.items())]
    for panel in panels:
        _require_panel_schema(panel)
    if {panel.system.name for panel in panels} != {"ALDP", "ALA4"}:
        raise ValueError("完整 E1 预测须同时提供 ALDP 和 ALA4 两个固定面板")
    result = run_predictions(panels, _mechanism_banks(panels, bank_paths), output,
                             config.get("mechanism", {}), resume=resume)
    return dict(status="sealed_predictions", systems=sorted(panel_paths),
                output=str(Path(output)/"predictions.pt"),
                eligible_for_formal=result.get("eligible_for_formal", False))


def intervene(config, predictions_path, bank_paths, output, resume=False):
    from .mechanism import load_mechanism_result, run_interventions
    config = load_config(config)
    prediction = load_mechanism_result(predictions_path)
    panels = prediction["panels"]
    for panel in panels:
        _require_panel_schema(panel)
    result = run_interventions(prediction, _mechanism_banks(panels, bank_paths), output,
                               config.get("mechanism", {}), resume=resume)
    return dict(status="sealed_interventions", output=str(Path(output)/"interventions.pt"),
                eligible_for_formal=result.get("eligible_for_formal", False))


def seal_deployments(config, runs, registry_path):

    from .inference import RoleSeal
    config, runs = load_config(config), Path(runs)
    experiment = config.get("experiment", {})
    methods = tuple(experiment.get("methods", METHODS))
    replicas = int(experiment.get("replicas", 128))
    if set(methods) != set(METHODS) or replicas != 128:
        raise ValueError("F 封存需要八法及每体系每法 128 个完整部署")
    budgets = experiment.get("budget_multipliers", [1.])
    if (not isinstance(budgets, list) or 1 not in budgets or len(set(budgets)) != len(budgets)
            or any(type(b) not in (int, float) or b not in (.5, 1., 2.) for b in budgets)):
        raise ValueError("F 预算面板须含主预算，辅助档仅允许 B*/2 和 2B*")
    auxiliary = experiment.get("auxiliary_replicas", 32)
    if type(auxiliary) is not int or auxiliary != 32:
        raise ValueError("辅助预算每方法每体系需要 32 次部署")
    expected = {(float(budget), system, method, replica)
                for budget in budgets for system in ("ALDP", "ALA4") for method in methods
                for replica in range(replicas if budget == 1 else auxiliary)}
    md_replicas = experiment.get("direct_md_replicas", 0)
    if type(md_replicas) is not int or md_replicas not in (0, 128):
        raise ValueError("正式直接端态 MD 须为 ALA4 的 128 个独立部署")
    if md_replicas:
        expected.update((1., "ALA4", "endpoint-MD", replica) for replica in range(md_replicas))
    included_methods = set(methods) | ({"endpoint-MD"} if md_replicas else set())
    found, all_units, paths = {}, [], []
    for path in sorted(runs.rglob("result.json")):
        result = read_json(path)
        if result.get("method") not in included_methods:
            continue
        key = (result.get("budget_ratio", 1.), result.get("molecule"), result["method"], result.get("replica"))
        if key not in expected:
            continue
        if key in found:
            raise ValueError("同一正式部署在运行目录内出现重复记录")
        if result.get("status") not in ("completed", "fallback", "no_legal_output") or result.get("mode") != "formal":
            raise ValueError("F 封存包含未终止或非正式运行")
        units = [unit for values in result.get("data_unit_ids", {}).values() for unit in values]
        if set(units) & set(all_units):
            raise ValueError("正式部署之间复用了实际 ladder 数据")
        all_units.extend(units)
        found[key] = result
        paths.append(str(path.resolve()))
    missing = expected-set(found)
    if missing:
        raise ValueError(f"冻结预算面板尚缺 {len(missing)} 个终止部署；不封存不完整 F")
    registry = RoleSeal(registry_path)
    registry.open("F", sorted(set(all_units)))
    registry.seal("F", paths)
    return dict(status="sealed", role="F", deployments=len(found), independent_units=len(set(all_units)),
                main_e3_deployments=2*len(methods)*replicas, endpoint_md_deployments=md_replicas,
                auxiliary_e3_deployments=sum(2*len(methods)*auxiliary for b in budgets if b != 1),
                frozen_budget_multipliers=budgets,
                no_legal_outputs=sum(value["status"] == "no_legal_output" for value in found.values()))


def reference(config, bank_paths, registry_path, output, look=1):
    from .evaluation import reference_summary
    from .inference import RoleSeal
    config, output = load_config(config), Path(output)
    registry = RoleSeal(registry_path)
    if any(registry.state["roles"].get(role, {}).get("status") != "sealed" for role in ("P", "I", "F")):
        raise ValueError("读取 R 银行前必须先完整封存 P、I 和 F")
    if set(bank_paths) != {"ALDP", "ALA4"}:
        raise ValueError("R 须一次登记 ALDP 和 ALA4 两套独立银行")
    for path in bank_paths.values():
        status_path = Path(path).parent/"reference-status.json"
        status = read_json(status_path) if status_path.exists() else {}
        if status.get("final") or status.get("stopped"):
            raise ValueError("R 已明确停止或标为 final，不能新增查看")
        if int(status.get("look", 0)) >= look:
            raise ValueError("这次 R 查看已计算，不能换输出目录重复揭盲")
    with run_lock(output):
        if (output/"result.json").exists():
            raise FileExistsError("本次参考查看已经有输出，不覆盖已揭盲结果")
        banks = {name: torch.load(path, weights_only=False) for name, path in bank_paths.items()}
        units = [unit for bank in banks.values() for unit in bank["source_unit_ids"]]
        registry.open("R", sorted(units))
        results = {}
        for name, bank in banks.items():
            reference_config = dict(config.get("reference", {}))
            certificates = reference_config.get("numerical_bias_certificates", {})
            if name in certificates:
                reference_config["numerical_bias_certificate"] = certificates[name]
            per_system_config = dict(config, reference=reference_config)
            result = reference_summary(bank, MolecularSystem(name), per_system_config, look=look, registry=registry)
            write_json(output/(name+".json"), _jsonable(result))
            results[name] = result
        summary = dict(status="reference_summary", look=look, references=_jsonable(results))
        write_json(output/"result.json", summary)
        reached_precision = all(result.get("eligible_for_formal") is True and result.get("precision_gates")
                                and all(gate == "supported" for gate in result["precision_gates"].values())
                                for result in results.values())
        stopping = dict(look=look, stopped=reached_precision, final=reached_precision or look == 4,
                        reason="precision_and_mixing" if reached_precision else "four_looks_exhausted" if look == 4 else "continue",
                        output=str((output/"result.json").resolve()))
        for path in bank_paths.values():
            write_json(Path(path).parent/"reference-status.json", stopping)
        return summary


def evaluate(config, runs, references, registry_path, output):
    from .evaluation import aggregate_results
    config, output = load_config(config), Path(output)
    with run_lock(output):
        if (output/"result.json").exists():
            raise FileExistsError("联合评估输出已经存在，不覆盖已登记结果")
        result = aggregate_results(runs, references, config, registry=registry_path)
        if config.get("experiment", {}).get("direct_md_replicas", 0):
            from .md_evaluation import endpoint_md_summary
            result["endpoint_md_diagnostic"] = endpoint_md_summary(runs, references, config, registry=registry_path)
        mechanism_directory = Path(registry_path).parent
        if all((mechanism_directory/name).is_file() for name in ("predictions.pt", "interventions.pt")):
            from .evaluation import _interval
            from .mechanism import evaluate_reference_claims, load_mechanism_result
            reference_populations = {}
            for name, path in references.items():
                reference_value = read_json(path) if isinstance(path, (str, Path)) else path
                reference_populations[name] = _interval(reference_value.get("intervals", {}).get("target_a"))
            result["mechanism_reference_claims"] = evaluate_reference_claims(
                load_mechanism_result(mechanism_directory/"predictions.pt"),
                load_mechanism_result(mechanism_directory/"interventions.pt"), reference_populations)
        else:
            result["mechanism_reference_claims"] = dict(status="unavailable", reason="No sealed P/I artifacts in the registry directory")
        write_json(output/"result.json", _jsonable(result))
        return _jsonable(result)
