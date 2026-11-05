from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
from scipy.stats import t
import torch

from .evaluation import _serialize
from .inference import Interval, RoleSeal, combine_gates, equivalence_gate, negative_gate, positive_gate, undefined
from .mechanism import _delta_rows
from .readouts import R_KJ
from .records import read_json, write_json


DIAGNOSTIC_EVENTS = 31


def _paths(paths):
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    result = []
    for path in paths:
        path = Path(path)
        result.extend(sorted(path.rglob("diagnostic.json")) if path.is_dir() else [path])
    if len(set(path.resolve() for path in result)) != len(result):
        raise ValueError("重复读取了同一诊断报告")
    return result


def _reveal(registry):
    registry = registry if isinstance(registry, RoleSeal) else RoleSeal(registry) if registry is not None else None
    if registry is None or any(registry.state["roles"].get(role, {}).get("status") != "sealed" for role in ("P", "I", "F")) or "R" not in registry.state["roles"]:
        raise ValueError("E2 读取 R 前须有已封存 P/I/F 和已明确揭盲的 R")
    return registry


def _interval(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        return undefined("独立单元不足或包含未定义读出")
    critical = t.ppf(1-.05/(2*DIAGNOSTIC_EVENTS), len(values)-1)
    center = float(values.mean())
    half = float(critical*values.std(ddof=1)/math.sqrt(len(values)))
    return Interval(center-half, center+half)


def _probability(interval):
    if not interval.finite:
        return interval
    return Interval(max(0., interval.lower), min(1., interval.upper))


def _proposal_reports(paths, system, registry):
    grouped = {method: [] for method in ("V", "V0")}
    identities, used_units = set(), set()
    registered = set(registry.state["roles"]["F"]["unit_ids"])
    reference = set(registry.state["roles"]["R"]["unit_ids"])
    for path in _paths(paths):
        report = read_json(path)
        if report.get("molecule") != system.name:
            raise ValueError("诊断报告与所给物理体系不同")
        production = Path(report.get("reused_production", ""))
        if not production.is_absolute():
            production = path.parent/production
        source_path = production/"result.json"
        if not source_path.is_file():
            raise ValueError("诊断没有可追溯的实际部署 result.json")
        source = read_json(source_path)
        method = report.get("method", source.get("method"))
        replica = report.get("replica", source.get("replica"))
        if method not in grouped or source.get("method") != method or source.get("molecule") != system.name or source.get("status") != "completed":
            raise ValueError("E2 只汇总取得原生冻结图的 V/V0 部署")
        if source.get("budget_ratio", 1.) != 1.:
            raise ValueError("E2 联合诊断需要主预算的128次部署")
        if type(replica) is not int or not 0 <= replica < 128 or source.get("replica") != replica:
            raise ValueError("部署编号不在预定128次内或报告与原记录不符")
        if (method, replica) in identities:
            raise ValueError("同一独立部署出现多个诊断报告")
        identities.add((method, replica))
        roles = report.get("data_unit_ids", source.get("data_unit_ids"))
        if not isinstance(roles, dict) or roles != source.get("data_unit_ids"):
            raise ValueError("报告与实际部署的数据单元证据缺失或不一致")
        units = [unit for values in roles.values() for unit in values]
        if not units or len(units) != len(set(units)) or not set(units).issubset(registered) or reference.intersection(units) or used_units.intersection(units):
            raise ValueError("部署单元缺失、未登记、跨部署复制或与 R 泄漏")
        used_units.update(units)
        q = np.asarray(report.get("main_partition", {}).get("proposal"), dtype=float)
        if q.shape != (3,) or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any() or abs(q.sum()-1) > 1e-8:
            raise ValueError("提议诊断须给出完整的 a/b/remainder 三概率")
        sample_count = source.get("native", {}).get("sample_count")
        if type(sample_count) is not int or sample_count < 1:
            raise ValueError("原生部署缺少实际完整生产样本数")
        grouped[method].append(dict(replica=replica, proposal=q, report=report,
            source_unit_ids=units, source_count=sample_count, eligible=source.get("eligible_for_formal") is True))
    return grouped


def _reference_moments(bank, system, config):
    points = torch.as_tensor(bank["positions"], dtype=torch.float64)
    states = np.asarray(bank["states"], dtype=float)
    units = list(bank["source_unit_ids"])
    cross = torch.as_tensor(bank["u_kn"])
    if points.ndim != 4 or points.shape[0] != len(units) or points.shape[2:] != (len(states), system.dim) or cross.shape != (*points.shape[:2], len(states), len(states)):
        raise ValueError("R 的实际构型、交叉能和 ladder 维数不一致")
    if points.shape[1] < 1 or not torch.isfinite(points).all() or not torch.isfinite(cross).all() or not np.isfinite(states).all() or not np.all(np.diff(states) > 0):
        raise ValueError("R 含无效保存数、非有限数据或重复状态；不删帧")
    if len(set(units)) != len(units) or not units:
        raise ValueError("R 的独立单元缺失或重复")
    if any(torch.equal(points[left], points[right]) for left in range(len(units)) for right in range(left)):
        raise ValueError("R 在不同 ladder ID 下复制了同一轨迹")
    endpoint = []
    for index in (0, 1):
        matches = np.flatnonzero(np.isclose(states, system.state(index), rtol=0, atol=1e-10))
        if len(matches) != 1:
            raise ValueError("R 没有唯一的物理端点")
        endpoint.append(int(matches[0]))
    batch_size = int(config.get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("R 实际物理算子的分块大小须为正")
    raw = getattr(system, "system", system)
    device = getattr(getattr(raw, "chart", None), "masses", torch.empty(0)).device
    probabilities = np.empty((len(units), len(states), 3))
    conditional = np.empty((len(units), len(states), 4)) if system.name == "ALA4" else None
    with torch.no_grad():
        for ladder in range(len(units)):
            for state in range(len(states)):
                counts = torch.zeros(3, dtype=torch.float64)
                sums = torch.zeros(2, dtype=torch.float64)
                for batch in points[ladder, :, state].split(batch_size):
                    batch = batch.to(device)
                    if hasattr(system, "raw") and config.get("reference_coordinates", "raw") == "raw":
                        batch = (batch-system.location)/system.scale
                    if hasattr(system, "in_domain") and not bool(system.in_domain(batch).all()):
                        raise ValueError("R 包含全 L 支持域之外的实际构型")
                    regions = system.regions(batch)
                    if "valid" in regions and not bool(regions["valid"].all()):
                        raise ValueError("R 包含无法判区的构型")
                    flags = torch.stack((regions["a"], regions["b"], ~(regions["a"]|regions["b"])), -1)
                    if bool((regions["a"]&regions["b"]).any()):
                        raise ValueError("R 的预定 a/b 硬区域相互重叠")
                    counts += flags.double().sum(0).cpu()
                    if conditional is not None:
                        if hasattr(system, "polar_energy"):
                            polar = system.polar_energy(batch)
                        else:


                            polar = torch.vmap(lambda z: raw.energy.gb_components(system.positions(z))[1])(batch)
                        if polar.shape != (len(batch),) or not torch.isfinite(polar).all():
                            raise ValueError("R 的实际 GB 极性算子无效")
                        sums += (flags[:, :2]*polar[:, None]/(R_KJ*300)).sum(0).cpu()
                probabilities[ladder, state] = counts.numpy()/points.shape[1]
                if conditional is not None:
                    conditional[ladder, state] = np.r_[probabilities[ladder, state, :2], sums.numpy()/points.shape[1]]
    return probabilities, conditional, states, endpoint


def _mismatch_projection(q, p):
    if not all(value.finite for value in (*q, *p)):
        return undefined("三概率的共同集合未定义")
    lower = .5*sum(max(0., qv.lower-pv.upper, pv.lower-qv.upper) for qv, pv in zip(q, p))
    upper = .5*sum(max(abs(qv.lower-pv.upper), abs(qv.upper-pv.lower)) for qv, pv in zip(q, p))
    return Interval(lower, min(1., upper))


def _closure(conditional, states, endpoints, config):
    if not np.isclose(states[0], 0.) or not np.isclose(states[-1], 1.):
        raise ValueError("ALA4 闭合需要完整 s=0 到1的物理窗口")
    matrix = torch.tensor(conditional.reshape(len(conditional), -1), dtype=torch.float64)
    if bool((matrix.mean(0).reshape(-1, 4)[:, :2] <= 0).any()):
        return dict(status="unestimable", reason="共同 R 原始矩中至少一个窗口核心零命中，不施加地板")
    weights = np.zeros(len(states))
    weights[:-1] += np.diff(states)/2;weights[1:] += np.diff(states)/2
    weights = torch.tensor(weights, dtype=torch.float64)
    def integral(m):
        m = m.reshape(-1, 4)
        return (weights*(m[:, 3]/m[:, 1]-m[:, 2]/m[:, 0])).sum()
    def preference(m):
        m = m.reshape(-1, 4)
        source, target = m[endpoints[0]], m[endpoints[1]]
        return -(target[1].log()-target[0].log()-source[1].log()+source[0].log())
    report = {}
    for name, functional in (("integral", integral), ("endpoint_preference", preference), ("finite_grid_difference", lambda m: integral(m)-preference(m))):
        point, rows = _delta_rows(matrix, functional)
        report[name] = dict(point=point, interval=_interval(rows))
    numerical = config.get("quadrature_error_bound")
    report["continuous_difference_interval"] = report["finite_grid_difference"]["interval"].widen(float(numerical)) if numerical is not None else undefined("没有冻结的连续积分求积误差界")
    tolerance = config.get("closure_tolerance")
    if tolerance is not None and (not math.isfinite(float(tolerance)) or float(tolerance) <= 0):
        raise ValueError("预定闭合等效带必须有限且为正")
    report.update(status="diagnostic", units="dimensionless (physical energy divided by R*300K)",
        conditional_mean_difference=(conditional.mean(0)[:, 3]/conditional.mean(0)[:, 1]-conditional.mean(0)[:, 2]/conditional.mean(0)[:, 0]).tolist(),
        gate=equivalence_gate(report["continuous_difference_interval"], float(tolerance)) if tolerance is not None else "not_predeclared",
        scope="全R原始矩的全局plug-in与配对cluster-delta；有限窗口采样区间不自动覆盖求积偏差")
    return report


def summarize_e2(report_paths, reference_bank, system, registry, config=None):
    config = {} if config is None else dict(config)
    if config.get("reference_coordinates", "raw") not in ("raw", "system"):
        raise ValueError("R 构型只能明确声明为 raw 或 system 坐标")
    if config.get("expected_deployments", 128) != 128 or config.get("diagnostic_alpha", .05) != .05:
        raise ValueError("E2 保留128部署和独立于正式族的95%诊断集合")
    registry = _reveal(registry)

    bank = torch.load(reference_bank, weights_only=False) if isinstance(reference_bank, (str, Path)) else reference_bank
    units = list(bank.get("source_unit_ids", []))
    roles = registry.state["roles"]
    occupied = {unit for role in ("P", "I", "F") for unit in roles[role]["unit_ids"]}
    if bank.get("role") != "R" or not units or not set(units).issubset(roles["R"]["unit_ids"]) or occupied.intersection(units):
        raise ValueError("实际 R 单元未登记、角色不符或与其他角色泄漏")
    grouped = _proposal_reports(report_paths, system, registry)
    probabilities, conditional, states, endpoints = _reference_moments(bank, system, config)
    pcenter = probabilities[:, endpoints[1]].mean(0)
    pintervals = [_probability(_interval(probabilities[:, endpoints[1], k])) for k in range(3)]
    reference_ready = len(units) >= 4 and bank.get("eligible_for_formal") is True and bank.get("qualification_result", {}).get("eligible_for_formal") is True
    result = dict(status="diagnostic", molecule=system.name, confidence=.95, diagnostic_interval_events=DIAGNOSTIC_EVENTS,
        independent_confirmation=False, formal_error_budget_unchanged=True,
        reference=dict(source_unit_ids=units, independent_ladders=len(units), frames_per_state=int(bank["positions"].shape[1]),
            qualification_confirmed=reference_ready, target_probabilities=pcenter.tolist(), target_probability_intervals=pintervals),
        mismatch={}, frozen_2x2={})
    for method, records in grouped.items():
        records.sort(key=lambda row: row["replica"])
        q = np.asarray([row["proposal"] for row in records]).reshape(-1, 3)
        qintervals = [_probability(_interval(q[:, k])) for k in range(3)]
        qcenter = q.mean(0) if len(q) else np.full(3, np.nan)
        mismatch_values = .5*np.abs(q-pcenter).sum(1)
        estimate = float(mismatch_values.mean()) if len(q) else None
        interval = _interval(mismatch_values)
        if interval.finite and all(value.finite for value in pintervals):
            radius = .5*sum(max(abs(value.lower-center), abs(value.upper-center)) for value, center in zip(pintervals, pcenter))
            interval = interval.widen(radius)
            interval = Interval(max(0., interval.lower), min(1., interval.upper))
        else:
            interval = undefined("部署均值或共同R三概率集合未定义")
        complete = len(records) == 128
        qualified = complete and reference_ready and all(row["eligible"] for row in records)
        result["mismatch"][method] = dict(independent_deployments=len(records), expected_deployments=128,
            missing_replicas=sorted(set(range(128))-{row["replica"] for row in records}),
            source_unit_ids=[unit for row in records for unit in row["source_unit_ids"]],
            production_sample_counts=[row["source_count"] for row in records],
            scope="all_128_native_maps" if complete else "available_native_maps_only_conditional",
            proposal_mean=qcenter.tolist(), proposal_mean_intervals=qintervals,
            mismatch_of_mean_proposal=_mismatch_projection(qintervals, pintervals),
            mean_deployment_mismatch=dict(point=estimate, interval=interval,
                conditional_sampling_gate=positive_gate(interval, .005),
                full_deployment_gate=positive_gate(interval, .005) if qualified else "unresolved"))
    paired, missing = [], []
    for row in grouped["V"]:
        cross = row["report"].get("frozen_2x2", {})
        moments = cross.get("moments", {})
        keys = ("V_on", "V_off", "V0_on", "V0_off")
        if cross.get("status") != "diagnostic" or any(not isinstance(moments.get(key), (int, float)) or not math.isfinite(moments[key]) for key in keys):
            missing.append(row["replica"]);continue
        d_v, d_v0 = moments["V_on"]-moments["V_off"], moments["V0_on"]-moments["V0_off"]
        vector = np.array([cross.get("d_V"), cross.get("d_V0"), cross.get("interaction")], dtype=float)
        if not np.isfinite(vector).all() or not np.allclose(vector, [d_v, d_v0, d_v-d_v0], rtol=1e-10, atol=1e-12):
            raise ValueError("冻结2×2诊断与其四张真实图矩不一致")
        paired.append(vector)
    matrix = np.asarray(paired).reshape(-1, 3)
    intervals = {name:_interval(matrix[:, index]) for index, name in enumerate(("d_V", "d_V0", "interaction"))}
    result["frozen_2x2"] = dict(valid_paired_deployments=len(paired), expected_deployments=128,
        missing_or_invalid_replicas=sorted(set(range(128))-{row["replica"] for row in grouped["V"] if row["replica"] not in missing}),
        scope="conditional_on_obtaining_all_four_frozen_maps", means=matrix.mean(0).tolist() if len(matrix) else None,
        joint_intervals=intervals, beneficial_and_route_specific=combine_gates([negative_gate(intervals["d_V"]),negative_gate(intervals["interaction"])]),
        V0_equivalence=equivalence_gate(intervals["d_V0"], math.log(1.02)))
    if conditional is not None:
        result["polar_closure"] = _closure(conditional, states, endpoints, config)
    return _serialize(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize joint molecular transport diagnostics.")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():raise FileExistsError("E2联合诊断输出已存在")
    config = read_json(args.config) if args.config else {}
    from .physics import AffineSystem, MolecularSystem
    if args.preparation:
        prepared = torch.load(args.preparation/"prepared.pt", weights_only=False)
        system = AffineSystem(MolecularSystem(prepared["molecule"]), prepared["location"], prepared["scale"])
    else:
        names = _paths(args.reports)
        if not names:raise ValueError("没有实际 diagnostic.json")
        system = MolecularSystem(read_json(names[0])["molecule"])
    report = summarize_e2(args.reports, args.reference, system, args.registry, config.get("diagnostic_inference", config))
    write_json(args.output, report)


if __name__ == "__main__":main()
