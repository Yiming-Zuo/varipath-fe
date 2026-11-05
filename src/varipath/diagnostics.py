from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm, rankdata


DEFINITIONS = {
    "independent_unit": "one complete replica-exchange ladder; states and walkers are never separate chains",
    "rhat": "max(rank-normalized split Rhat, rank-normalized folded split Rhat); average tie ranks; Blom transform (r-3/8)/(N+1/4)",
    "bulk_ess": "split rank-normalized draws; N / pymbar.timeseries.statistical_inefficiency_multiple(fast=False)",
    "tail_ess": "minimum N/g of pooled 5% and 95% quantile indicators on split chains; same PyMBAR estimator",
    "binary_ess": "N/g of the original split indicator; quantile tail ESS is inapplicable to binary data",
    "periodic_angles": "separate sin and cos time series, not angles across the periodic branch cut",
    "qualification": "conditional finite-run diagnostics with a separately recorded sampling protocol; no proof of equilibrium or unseen-tail control",
}


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _split(chains):
    half = chains.shape[1] // 2
    return np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)


def _rank_normalize(chains):
    ranks = rankdata(chains.ravel(), method="average")
    return norm.ppf((ranks - 3/8) / (len(ranks) + 1/4)).reshape(chains.shape)


def _rhat(chains):
    count = chains.shape[1]
    within = np.var(chains, axis=1, ddof=1).mean()
    if not math.isfinite(float(within)) or within <= 0:
        return None
    between = count * np.var(chains.mean(axis=1), ddof=1)
    value = math.sqrt(((count - 1) * within / count + between / count) / within)
    return float(value) if math.isfinite(value) else None


def _multiple_ess(chains):
    if np.var(chains) == 0:
        return None
    from pymbar.timeseries import statistical_inefficiency_multiple

    inefficiency = float(statistical_inefficiency_multiple(chains, fast=False))
    if not math.isfinite(inefficiency) or inefficiency < 1:
        return None
    return float(chains.size / inefficiency)


def chain_diagnostics(chains, *, binary=False, rhat_limit=1.01,
                      minimum_ess=400., minimum_draws=8, minimum_tail_count=20) -> dict:
    values = _array(chains).astype(float)
    result = {"status": "unresolved", "rhat": None, "rank_split_rhat": None,
              "folded_split_rhat": None, "bulk_ess": None, "tail_ess": None,
              "indicator_ess": None, "reasons": []}
    if values.ndim != 2:
        raise ValueError("诊断数组必须是 [independent_ladder,time]")
    result.update(ladders=int(values.shape[0]), draws_per_ladder=int(values.shape[1]), binary=bool(binary))
    if values.shape[0] < 4:
        result["reasons"].append("至少需要四条独立 ladder")
    if values.shape[1] < minimum_draws:
        result["reasons"].append("每条 ladder 的保存时间点不足")
    if not np.isfinite(values).all():
        result["reasons"].append("存在非有限观测；不删除该观测")
    if result["reasons"]:
        return result
    if np.unique(values).size == 1:
        result["reasons"].append("全常量序列无法识别混合，不能赋 Rhat=1")
        return result
    if binary and not np.isin(values, [0., 1.]).all():
        raise ValueError("indicator 诊断需要原始 0/1 值")
    split = _split(values)
    ranked = _rank_normalize(split)
    folded = _rank_normalize(np.abs(split - np.median(split)))
    rank_rhat, folded_rhat = _rhat(ranked), _rhat(folded)
    result["rank_split_rhat"], result["folded_split_rhat"] = rank_rhat, folded_rhat
    available = [x for x in (rank_rhat, folded_rhat) if x is not None]
    result["rhat"] = max(available) if available else None
    if rank_rhat is None:
        result["reasons"].append("链内方差为零或 rank Rhat 无定义")
    if folded_rhat is None and not binary:
        result["reasons"].append("folded Rhat 无定义")
    result["bulk_ess"] = _multiple_ess(ranked)
    if binary:
        result["indicator_ess"] = _multiple_ess(split)
        result["tail_applicable"] = False
        for event in (split == 0, split == 1):
            if int(event.sum()) < minimum_tail_count:
                result["reasons"].append("indicator 事件或其补集过于稀疏")
                break
        effective = [result["bulk_ess"], result["indicator_ess"]]
    else:
        result["tail_applicable"] = True
        q05, q95 = np.quantile(split, [.05, .95])
        tails = [(split <= q05).astype(float), (split <= q95).astype(float)]
        counts = [min(int(event.sum()), int(event.size - event.sum())) for event in tails]
        result["tail_event_counts"] = counts
        if min(counts) < minimum_tail_count:
            result["reasons"].append("5% 或 95% 尾事件过于稀疏")
        estimates = [_multiple_ess(event) for event in tails]
        result["tail_ess"] = min(estimates) if all(x is not None for x in estimates) else None
        effective = [result["bulk_ess"], result["tail_ess"]]
    if any(x is None for x in effective):
        result["reasons"].append("至少一个 ESS 无定义")
    if result["reasons"]:
        return result
    failed = []
    if result["rhat"] > rhat_limit:
        failed.append("Rhat 超过冻结阈值")
    if min(effective) < minimum_ess:
        failed.append("ESS 低于冻结阈值")
    result["status"] = "failed" if failed else "passed"
    result["reasons"] = failed
    return result


def _protocol_report(bank, config):
    protocol = dict(config.get("protocol", {}))
    for key in ("warmup", "step_size", "leapfrog_steps"):
        if key not in protocol and key in config:
            protocol[key] = config[key]
    reasons = []
    if protocol.get("kernel") not in ("metropolis_hmc", "metropolis_hmc_replica_exchange"):
        reasons.append("未记录经 Metropolis 校正的实际采样核")
    for key in ("warmup", "step_size", "leapfrog_steps"):
        value = protocol.get(key)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            reasons.append(f"缺少合法的 {key} 协议")
    for key in ("warmup_selection", "initialization_strategy"):
        if not isinstance(protocol.get(key), str) or not protocol[key].strip():
            reasons.append(f"未记录 {key}")
    if protocol.get("numerical_validation_passed") is not True:
        reasons.append("未接入 HMC 可逆性、接受率及数值核的准入结果")
    mode = bank.get("mode", config.get("mode", "smoke"))
    if mode != "formal":
        reasons.append("smoke/development 数据不获得 formal 资格")
    return {"status": "passed" if not reasons else "unresolved", "protocol": protocol,
            "mode": mode, "reasons": reasons}


def qualify_bank(system, bank: dict, config: dict | None = None) -> dict:
    config = {} if config is None else config
    positions = _array(bank["positions"])
    states = np.asarray(bank["states"], dtype=float)
    if positions.ndim != 4 or positions.shape[2] != len(states):
        raise ValueError("采样银行必须是 [ladder,time,state,shape_dimension]")
    ladders, draws, nstates, dimension = positions.shape
    units = list(bank.get("source_unit_ids", []))
    if len(units) != ladders or len(set(units)) != ladders:
        raise ValueError("每条 ladder 必须有唯一独立 source_unit_id")
    for first in range(ladders):
        for second in range(first):
            if np.array_equal(positions[first], positions[second]):
                raise ValueError("不同 ID 的 ladder 含完全相同轨迹，不能视为独立重复")
    if not np.isfinite(positions).all() or not np.isfinite(states).all():
        raise ValueError("银行位置或物理状态非有限")
    if len(set(states.tolist())) != nstates:
        raise ValueError("物理窗口状态重复")
    options = {"rhat_limit": float(config.get("rhat_limit", 1.01)),
               "minimum_ess": float(config.get("minimum_ess", 400.)),
               "minimum_draws": int(config.get("minimum_draws", 8)),
               "minimum_tail_count": int(config.get("minimum_tail_count", 20))}
    if options["rhat_limit"] <= 1 or options["minimum_ess"] <= 0 or options["minimum_draws"] < 8:
        raise ValueError("诊断阈值无效")
    protocol = _protocol_report(bank, config)
    report = {"status": "unresolved", "eligible_for_formal": False, "conditional_checks_passed": False,
              "definitions": DEFINITIONS.copy(), "thresholds": options, "source_unit_ids": units,
              "ladders": ladders, "draws_per_ladder": draws, "protocol": protocol,
              "diagnostics": {}, "endpoint_support": {}, "reasons": []}
    if dimension != int(system.dim):
        raise ValueError("银行形状维数与当前体系不一致")
    reference_tensor = next(system.parameters(), None) if hasattr(system, "parameters") else None
    device = reference_tensor.device if reference_tensor is not None else torch.device("cpu")
    dtype = torch.float64
    with torch.no_grad():
        all_domain = _array(system.in_domain(torch.tensor(positions.reshape(-1, dimension), dtype=dtype, device=device)))
    report["all_windows_in_domain"] = bool(all_domain.all())
    if not report["all_windows_in_domain"]:
        report["reasons"].append("至少一个物理窗口有全 L 型域失败；不能事后删除构型")
    report["acceptance"] = {key: bank.get(key) for key in ("hmc_acceptance", "exchange_acceptance")}
    if bank.get("hmc_acceptance") is None or bank.get("exchange_acceptance") is None:
        report["reasons"].append("缺少本次各窗口 HMC 与交换接受率记录")
    else:
        rates = np.r_[np.asarray(bank["hmc_acceptance"], dtype=float).ravel(),
                      np.asarray(bank["exchange_acceptance"], dtype=float).ravel()]
        if not np.isfinite(rates).all() or (rates < 0).any() or (rates > 1).any():
            report["reasons"].append("接受率记录无效")
        if (np.asarray(bank["hmc_acceptance"], dtype=float) < float(config.get("minimum_hmc_acceptance", .01))).any():
            report["reasons"].append("至少一个窗口的 HMC 接受率低于冻结下限")
    cross = _array(bank["u_kn"]) if "u_kn" in bank else None
    if cross is not None and cross.shape != (ladders, draws, nstates, nstates):
        raise ValueError("交叉能量必须保持 [ladder,time,evaluation_state,sample_state] 次序")
    if cross is None:
        report["reasons"].append("缺少窗口交叉能量")
    else:
        for index, state in enumerate(states):
            report["diagnostics"][f"state/{state:g}/effective_energy"] = chain_diagnostics(cross[:, :, index, index], **options)
    endpoints = {}
    for endpoint in (0, 1):
        matching = np.flatnonzero(np.isclose(states, float(system.state(endpoint)), rtol=0, atol=1e-10))
        if len(matching) != 1:
            raise ValueError("银行缺少当前物理源端或目标端状态")
        index = int(matching[0])
        endpoints[endpoint] = index
        points = torch.tensor(positions[:, :, index].reshape(-1, dimension), dtype=dtype, device=device)
        with torch.no_grad():
            regions = system.regions(points)
            domain = _array(system.in_domain(points)).reshape(ladders, draws)
        valid = _array(regions["valid"]).reshape(ladders, draws)
        if not domain.all() or not valid.all():
            report["reasons"].append(f"端点 {endpoint} 有全 L 域或区域标签失败；不删点")
        support = {"state_index": index, "state": float(states[index]), "all_in_domain": bool(domain.all()), "all_labels_valid": bool(valid.all())}
        for region in ("a", "b"):
            indicators = _array(regions[region]).reshape(ladders, draws).astype(float)
            counts = indicators.sum(axis=1).astype(int)
            support[region] = {"counts_by_ladder": counts.tolist(), "total": int(counts.sum()),
                               "raw_population": float(indicators.mean()), "zero_hit": bool(counts.sum() == 0),
                               "any_ladder_zero_hit": bool((counts == 0).any())}
            report["diagnostics"][f"endpoint/{endpoint}/indicator/{region}"] = chain_diagnostics(indicators, binary=True, **options)
            if counts.sum() < int(config.get("minimum_region_hits", 20)) or (counts < int(config.get("minimum_region_hits_per_ladder", 2))).any():
                report["reasons"].append(f"端点 {endpoint} 区域 {region} 原生支撑不足")
        angles = _array(regions["angles"]).reshape(ladders, draws, -1)
        for angle in range(angles.shape[-1]):
            radians = np.deg2rad(angles[:, :, angle])
            for name, transform in (("sin", np.sin), ("cos", np.cos)):
                report["diagnostics"][f"endpoint/{endpoint}/angle/{angle}/{name}"] = chain_diagnostics(transform(radians), **options)
        report["endpoint_support"][str(endpoint)] = support
    if cross is not None:
        source, target = endpoints[0], endpoints[1]
        logweight = cross[:, :, source, source] - cross[:, :, target, source]
        report["diagnostics"]["source/identity_logweight"] = chain_diagnostics(logweight, **options)
        if np.isfinite(logweight).all():
            scaled = np.exp(logweight - logweight.max())
            report["diagnostics"]["source/identity_weight"] = chain_diagnostics(scaled, **options)
            report["identity_weight_summary"] = {"importance_ess": float(scaled.sum()**2 / np.square(scaled).sum()),
                                                   "raw_count": int(scaled.size), "logweight_range": [float(logweight.min()), float(logweight.max())],
                                                   "interpretation": "importance ESS is a separate weight-dispersion diagnostic, not evidence of equilibrium"}
    diagnostics = list(report["diagnostics"].values())
    failed = [key for key, value in report["diagnostics"].items() if value["status"] == "failed"]
    unresolved = [key for key, value in report["diagnostics"].items() if value["status"] == "unresolved"]
    report["failed_diagnostics"], report["unresolved_diagnostics"] = failed, unresolved
    report["conditional_checks_passed"] = bool(diagnostics) and not failed and not unresolved and not report["reasons"] and protocol["status"] == "passed"
    if report["conditional_checks_passed"]:
        report["status"], report["eligible_for_formal"] = "passed_conditional_diagnostics", True
    elif failed:
        report["status"] = "not_qualified"
    report["reasons"].extend(protocol["reasons"])
    return report
