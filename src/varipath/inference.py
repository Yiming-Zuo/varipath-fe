from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


FAMILY_ALPHA = {"P": .005, "I": .015, "E3": .02, "R": .01}
FAMILY_EVENTS = {"I": 418, "E3": 64, "R": 24}
DELTA_MASS = .005
DELTA_LOG_RISK = math.log(1.02)


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    status: str = "valid"
    reason: str = ""

    def __post_init__(self):
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise ValueError("区间端点无效")

    @property
    def finite(self):
        return self.status == "valid" and math.isfinite(self.lower) and math.isfinite(self.upper)

    def widen(self, error: float):
        if not math.isfinite(error) or error < 0:
            raise ValueError("数值误差界必须有限且非负")
        return Interval(self.lower - error, self.upper + error, self.status, self.reason)


def undefined(reason: str) -> Interval:
    return Interval(-math.inf, math.inf, "undefined", reason)


def bonferroni_t(observations, family: str, *, events: int | None = None,
                 independent_unit_ids=None) -> list[Interval]:

    values = np.asarray(observations, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or family not in FAMILY_ALPHA:
        raise ValueError("区间输入或检验族无效")
    if independent_unit_ids is not None:
        ids = list(independent_unit_ids)
        if len(ids) != len(values) or len(set(ids)) != len(ids):
            raise ValueError("独立单元 ID 缺失或重复")
    if events is None:
        if family == "P":
            raise ValueError("P 族须由冻结的预测 manifest 显式给出事件总数")
        events = FAMILY_EVENTS[family]
    if events < values.shape[1] or events < 1:
        raise ValueError("错误率事件数不得小于实际区间数")
    if family in FAMILY_EVENTS and events != FAMILY_EVENTS[family]:
        raise ValueError("I、E3、R 族事件数须采用现稿的 418、64、24")
    critical = student_t.ppf(1 - FAMILY_ALPHA[family] / (2 * events), len(values) - 1) if len(values) > 1 else math.inf
    intervals = []
    for column in values.T:
        if len(column) < 2 or not np.isfinite(column).all():
            intervals.append(undefined("独立单元不足或存在失败／非有限输出"))
            continue
        center = float(column.mean())
        half = float(critical * column.std(ddof=1) / math.sqrt(len(column)))
        intervals.append(Interval(center - half, center + half))
    return intervals


def equivalence_gate(interval: Interval, half_width: float) -> str:
    if not interval.finite:
        return "unresolved"
    if interval.lower >= -half_width and interval.upper <= half_width:
        return "supported"
    if interval.upper < -half_width or interval.lower > half_width:
        return "unsupported"
    return "unresolved"


def positive_gate(interval: Interval, threshold: float = 0.) -> str:
    if not interval.finite:
        return "unresolved"
    if interval.lower > threshold:
        return "supported"
    if interval.upper <= threshold:
        return "unsupported"
    return "unresolved"


def negative_gate(interval: Interval, threshold: float = 0.) -> str:
    return positive_gate(Interval(-interval.upper, -interval.lower, interval.status, interval.reason), -threshold)


def prediction_gate(prediction: Interval, measured: Interval, delta: float) -> dict:
    if not prediction.finite or not measured.finite:
        return {"status": "unresolved", "maximum_standardized_error": math.inf}
    if delta <= 0:
        raise ValueError("效应尺度必须为正")
    candidates = [prediction.lower, prediction.upper]
    for p in (0., -2.5 * delta, 2.5 * delta, measured.lower, measured.upper):
        if prediction.lower <= p <= prediction.upper:
            candidates.append(p)
    maximum, minimum, standardized = -math.inf, math.inf, 0.
    for p in candidates:
        tolerance = max(delta / 2, .2 * abs(p))
        errors = [abs(measured.lower - p), abs(measured.upper - p)]
        closest = min(max(p, measured.lower), measured.upper)
        maximum = max(maximum, max(errors) - tolerance)
        minimum = min(minimum, abs(closest - p) - tolerance)
        standardized = max(standardized, max(errors) / tolerance)
    status = "supported" if maximum <= 0 else ("unsupported" if minimum > 0 else "unresolved")
    return {"status": status, "maximum_standardized_error": standardized}


def combine_gates(statuses) -> str:
    statuses = list(statuses)
    if not statuses or any(x not in ("supported", "unsupported", "unresolved") for x in statuses):
        raise ValueError("门判决缺失或状态无效")
    if "unsupported" in statuses:
        return "unsupported"
    return "supported" if all(x == "supported" for x in statuses) else "unresolved"


def repair_intervals(reference_population: Interval, baseline_population: Interval,
                     mass_change: Interval) -> dict[str, Interval]:
    if not all(x.finite for x in (reference_population, baseline_population, mass_change)):
        return {"deficit": undefined("参考或干预集合未定义"), "gain": undefined("参考或干预集合未定义")}
    d = Interval(reference_population.lower - baseline_population.upper,
                 reference_population.upper - baseline_population.lower)
    x = mass_change
    points = [(a, b) for a in (d.lower, d.upper) for b in (x.lower, x.upper)]
    if d.lower <= 0 <= d.upper:
        points += [(0., b) for b in (x.lower, x.upper)]
    overlap_lo, overlap_hi = max(d.lower, x.lower), min(d.upper, x.upper)
    if overlap_lo <= overlap_hi:
        points += [(overlap_lo, overlap_lo), (overlap_hi, overlap_hi)]
    gains = [abs(a) - abs(b - a) for a, b in points]
    return {"deficit": d, "gain": Interval(min(gains), max(gains))}


def _product(a: Interval, b: Interval) -> Interval:
    candidates = [x * y for x in (a.lower, a.upper) for y in (b.lower, b.upper)]
    return Interval(min(candidates), max(candidates))


def paired_mse_interval(variant, baseline, reference: Interval, *, reference_center: float,
                        independent_unit_ids=None) -> dict:

    v = np.asarray(variant, dtype=float) - reference_center
    b = np.asarray(baseline, dtype=float) - reference_center
    if v.shape != b.shape or v.ndim != 1:
        raise ValueError("方法输出必须是同部署配对的一维向量")
    moments = bonferroni_t(np.column_stack((v * v - b * b, v - b)), "E3", independent_unit_ids=independent_unit_ids)
    if not reference.finite or not all(x.finite for x in moments):
        return {"interval": undefined("部署失败或参考未定义"), "status": "unresolved", "moments": moments}
    delta = Interval(reference.lower - reference_center, reference.upper - reference_center)
    product = _product(delta, moments[1])
    interval = Interval(moments[0].lower - 2 * product.upper, moments[0].upper - 2 * product.lower)
    return {"interval": interval, "status": negative_gate(interval), "moments": moments}


def absolute_mse_interval(outputs, reference: Interval, *, reference_center: float,
                          independent_unit_ids=None) -> dict:
    values = np.asarray(outputs, dtype=float) - reference_center
    if values.ndim != 1:
        raise ValueError("部署输出必须是一维向量")
    moments = bonferroni_t(np.column_stack((values * values, values)), "E3", independent_unit_ids=independent_unit_ids)
    if not reference.finite or not all(x.finite for x in moments):
        return {"interval": undefined("部署失败或参考未定义"), "status": "unresolved", "moments": moments}
    lo, hi = reference.lower - reference_center, reference.upper - reference_center
    terms = []
    for mean in (moments[1].lower, moments[1].upper):
        for delta in (lo, hi, min(max(mean, lo), hi)):
            terms.append(delta ** 2 - 2 * delta * mean)
    interval = Interval(max(0., moments[0].lower + min(terms)), max(0., moments[0].upper + max(terms)))
    status = "supported" if interval.upper <= .01 else ("unsupported" if interval.lower > .01 else "unresolved")
    return {"interval": interval, "status": status, "moments": moments}


def panel_manifest() -> list[dict]:
    cells = []
    for system in ("ALA4", "ALDP"):
        for anchor in range(4):
            for window in range(3):
                for offset in (-1., 0., 1., .5):
                    cells.append({"id": f"{system}-a{anchor}-w{window}-r{offset:+g}",
                                  "system": system, "anchor": anchor, "window": window,
                                  "offset": offset, "panel": "same_side" if offset == .5 else "main",
                                  "doses": ["full", "half"], "status": "not_run", "reason": ""})
    return cells


def intervention_interval_names() -> list[str]:
    cells = panel_manifest()
    names = [f"{cell['id']}/{readout}/{contrast}" for cell in cells
             for readout in ("mass", "log_risk") for contrast in ("full", "full_minus_twice_half")]
    for system in ("ALA4", "ALDP"):
        names.append(f"{system}/baseline_a")
        names.append(f"{system}/geometric_squared_error_difference")
        for shape in range(3):
            names.extend([f"{system}/shape{shape}/h2", f"{system}/shape{shape}/h2_full_minus_four_half"])
            names.extend(f"{system}/shape{shape}/mass_{region}" for region in ("a", "b", "remainder"))
    return names


def panel_agreement(predictions: dict[str, Interval], measured: dict[str, Interval],
                    half_dose_residuals: dict[str, Interval], *,
                    numerical_error_bounds: dict[str, float] | None = None) -> dict:

    results = {}
    panel_statuses = {"main": [], "same_side": []}
    maxima = {"main": 0., "same_side": 0.}
    for cell in panel_manifest():
        statuses = []
        for readout, delta in (("mass", DELTA_MASS), ("log_risk", DELTA_LOG_RISK)):
            key = cell["id"] + "/" + readout
            prediction = predictions.get(key, undefined("缺少冻结的 P 预测"))
            actual = measured.get(key, undefined("缺少 I 结果，空位不能补零"))
            residual = half_dose_residuals.get(key, undefined("缺少配对半剂量分支"))
            if numerical_error_bounds is None or key not in numerical_error_bounds:
                prediction = undefined("未提供该运行的数值误差认证界")
            else:
                prediction = prediction.widen(numerical_error_bounds[key])
            magnitude = prediction_gate(prediction, actual, delta)
            linearity = equivalence_gate(residual, delta / 2)
            status = combine_gates([magnitude["status"], linearity])
            results[key] = {"status": status, "magnitude": magnitude, "half_dose": linearity}
            statuses.append(status)
            maxima[cell["panel"]] = max(maxima[cell["panel"]], magnitude["maximum_standardized_error"])
        panel_statuses[cell["panel"]].extend(statuses)
    return {"main": {"status": combine_gates(panel_statuses["main"]), "maximum_standardized_error": maxima["main"], "cells": 72},
            "same_side": {"status": combine_gates(panel_statuses["same_side"]), "maximum_standardized_error": maxima["same_side"], "cells": 24},
            "readouts": results}


def conditional_h2_interval(restricted_affinity: Interval, rho0: Interval, rhoe: Interval) -> Interval:

    if not all(x.finite for x in (restricted_affinity, rho0, rhoe)) or min(rho0.lower, rhoe.lower) <= 0:
        return undefined("格质量尚不可辨，条件 Hellinger 距离未定义")
    denominators = (math.sqrt(rho0.lower * rhoe.lower), math.sqrt(rho0.upper * rhoe.upper))
    ratios = [value / divisor for value in (restricted_affinity.lower, restricted_affinity.upper) for divisor in denominators]
    return Interval(1 - max(ratios), 1 - min(ratios))


def shape_claim_gate(mass_changes: list[Interval], conditional_h2: Interval,
                     predicted_h2: Interval, half_dose_residual: Interval) -> dict:
    if len(mass_changes) != 3:
        raise ValueError("仅盆内形变主张必须覆盖 a、b、remainder 三大区")
    statuses = [equivalence_gate(x, DELTA_MASS) for x in mass_changes]
    statuses += [positive_gate(conditional_h2, .01),
                 prediction_gate(predicted_h2, conditional_h2, .01)["status"],
                 equivalence_gate(half_dose_residual, .005)]
    return {"status": combine_gates(statuses), "component_statuses": statuses}


def select_same_side_pair(candidates: list[dict], deficit_development: float) -> dict | None:

    qualified = []
    for candidate in candidates:
        if candidate.get("data_role") != "P":
            raise ValueError("同侧候选只能由 P 选择")
        mass, risk = candidate["mass"], candidate["log_risk"]
        if len(mass) != 2 or len(risk) != 2 or not all(x.finite for x in mass + risk):
            continue
        if not all(x.lower > DELTA_MASS and x.upper < 2 * deficit_development for x in mass):
            continue
        for negative, positive in ((0, 1), (1, 0)):
            if risk[negative].upper < -DELTA_LOG_RISK and risk[positive].lower > DELTA_LOG_RISK:
                mass_points = candidate.get("mass_points", [(x.lower + x.upper) / 2 for x in mass])
                risk_points = candidate.get("log_risk_points", [(x.lower + x.upper) / 2 for x in risk])
                margin = min(mass_points[0] / DELTA_MASS, mass_points[1] / DELTA_MASS,
                             -risk_points[negative] / DELTA_LOG_RISK, risk_points[positive] / DELTA_LOG_RISK)
                qualified.append((margin, int(candidate["anchor"]), int(candidate["window"]), negative, candidate))
    if not qualified:
        return None
    winner = sorted(qualified, key=lambda x: (-x[0], x[1], x[2]))[0]
    return {**winner[4], "margin": winner[0], "negative_risk_route": winner[3], "status": "P_selected"}


class RoleSeal:


    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"roles": {}}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def open(self, role: str, independent_unit_ids: list[str]):
        if role not in ("D_fit", "D_select", "P", "I", "F", "R"):
            raise ValueError("数据角色无效")
        if len(independent_unit_ids) == 0 or len(set(independent_unit_ids)) != len(independent_unit_ids):
            raise ValueError("独立 chain/ladder 身份缺失或重复")
        roles = self.state["roles"]
        required = ("P",) if role == "I" else (("P", "I", "F") if role == "R" else ())
        if any(roles.get(key, {}).get("status") != "sealed" for key in required):
            raise ValueError("未满足 P 先封存、R 后揭盲的依赖")
        if role in roles:
            if roles[role]["unit_ids"] != independent_unit_ids:
                raise ValueError("断点恢复时不能更换数据角色的独立单元")
            return roles[role]
        other_ids = {unit for entry in roles.values() for unit in entry["unit_ids"]}
        if other_ids.intersection(independent_unit_ids):
            raise ValueError("数据角色之间出现共同 chain/ladder")
        roles[role] = {"unit_ids": independent_unit_ids, "status": "open"}
        self._save()
        return roles[role]

    def seal(self, role: str, output_paths: list[str], *, complete: bool = True):
        entry = self.state["roles"].get(role)
        if entry is None:
            raise ValueError("该角色尚未登记")
        if entry["status"] == "sealed":
            if entry["output_paths"] != output_paths:
                raise ValueError("已封存输出不可替换")
            return
        if not complete or not output_paths or not all(Path(path).is_file() for path in output_paths):
            raise ValueError("只有实际存在的完整输出可以封存")
        previous_paths = {str(Path(path).resolve()) for other_role, other_entry in self.state["roles"].items()
                          if other_role != role for path in other_entry.get("output_paths", [])}
        if previous_paths.intersection(str(Path(path).resolve()) for path in output_paths):
            raise ValueError("不同数据角色不能封存同一输出文件")
        entry.update(status="sealed", output_paths=output_paths,
                     sealed_at=datetime.now(timezone.utc).isoformat())
        self._save()
