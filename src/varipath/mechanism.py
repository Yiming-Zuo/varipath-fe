from __future__ import annotations

import math
import os
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from .inference import (DELTA_LOG_RISK, DELTA_MASS, Interval, RoleSeal, bonferroni_t,
                        combine_gates, intervention_interval_names, negative_gate,
                        panel_agreement, panel_manifest, positive_gate,
                        repair_intervals, select_same_side_pair, shape_claim_gate, undefined)
from .records import save_checkpoint
from .geometry_certificate import certify_geometry
from .interventions import (ConjugateRoute, GeometryQualification, IdentityLoop,
                            InterventionError, ProtectedTranslation, finite_intervention,
                            geometric_mass_prediction, response_samples, window_weights)


def _tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _first(value):
    return value[0] if isinstance(value, tuple) else value


def _base_step(base, z, k):
    return base.single_step(z, k) if hasattr(base, "single_step") else _first(base.step(z[None], k))[0]


def _base_inverse(base, z, k):
    return _first(base.single_inverse_step(z, k)) if hasattr(base, "single_inverse_step") else _first(base.inverse_step(z[None], k))[0]


def basin_levels(system, batch):

    if hasattr(system, "mechanism_boundary_levels"):
        values = system.mechanism_boundary_levels(batch)
        return values[:, None] if values.ndim == 1 else values
    angles = system.regions(batch)["angles"].reshape(len(batch), -1, 2)
    difference = (angles - angles.new_tensor([-70., -40.]) + 180) % 360 - 180
    return difference.square().sum(-1) / 1600 - 1


def main_labels(system, batch):
    regions = system.regions(batch)
    labels = torch.where(regions["a"], 0, torch.where(regions["b"], 1, 2)).long()
    return torch.where(regions["valid"], labels, torch.full_like(labels, -1))


def _regular_boundary(base, system, source, config, endpoint=None, reference_endpoint=None):
    endpoint = _first(base.forward(source)).detach() if endpoint is None else endpoint
    levels = basin_levels(system, endpoint).detach().cpu().numpy()
    candidates = []
    margin = float(config.get("other_patch_margin", 1e-4))
    for active in range(levels.shape[1]):
        other = np.delete(levels, active, axis=1)
        allowed = (other < -margin).all(-1) if other.shape[1] else np.ones(len(levels), bool)
        inside = np.flatnonzero(allowed & (levels[:, active] < 0))
        outside = np.flatnonzero(allowed & (levels[:, active] > 0))
        if len(inside) == 0 or len(outside) == 0:
            continue
        tree = cKDTree(endpoint[outside].cpu().numpy())
        distances, nearest = tree.query(endpoint[inside].cpu().numpy(), k=1)
        for distance, i, j in zip(distances, inside, nearest):
            outside_index = int(outside[j])
            midpoint = .5*(endpoint[int(i)]+endpoint[outside_index])
            proximity = float((midpoint-reference_endpoint).norm()) if reference_endpoint is not None else 0.
            candidates.append((proximity, float(distance), active, int(i), outside_index))
    candidates.sort()
    attempted = []
    for _, _, active, inside, outside in candidates[:int(config.get("boundary_max_brackets", 64))]:
        start, difference = endpoint[inside], endpoint[outside] - endpoint[inside]
        phi = lambda z: basin_levels(system, z[None])[0, active]
        try:
            fraction = brentq(lambda value: float(phi(start + value*difference).detach()), 0., 1.,
                              xtol=float(config.get("boundary_root_tolerance", 1e-12)))
            point = (start + fraction*difference).detach()
            values = basin_levels(system, point[None])[0]
            others = torch.cat((values[:active], values[active+1:]))
            gradient = torch.func.grad(phi)(point)
            if (not bool(system.in_domain(point)) or not bool(torch.isfinite(gradient).all())
                    or float(gradient.norm()) < float(config.get("boundary_gradient_minimum", 1e-8))
                    or abs(float(values[active])) > float(config.get("boundary_level_tolerance", 1e-8))
                    or (len(others) and not bool((others < -margin).all()))):
                raise InterventionError("根不在所需正则边界片内")
            return point, gradient / gradient.norm(), active, {"inside_frame": inside, "outside_frame": outside,
                    "active_patch": active, "endpoint_level_residual": abs(float(values[active])),
                    "other_levels": others.tolist(), "gradient_norm": float(gradient.norm()), "attempts": attempted}
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            attempted.append(str(exc))
    raise InterventionError("D 端点没有找到满足真实 a 边界的正则根；" + "; ".join(attempted[-3:]))


def _track_boundary(base, system, endpoint, normal, config):
    points, normals = [None] * (base.n_steps + 1), [None] * (base.n_steps + 1)
    points[-1], normals[-1] = endpoint, normal
    tolerance = float(config.get("inverse_tolerance", 1e-10))
    for k in reversed(range(base.n_steps)):
        point = _base_inverse(base, points[k+1], k).detach()
        residual = float((_base_step(base, point, k) - points[k+1]).detach().abs().max())
        if residual > tolerance or not bool(system.in_domain(point)):
            raise InterventionError("逆追边界的基流逆图或全 L 域检查失败")
        jacobian = torch.func.jacrev(lambda z: _base_step(base, z, k))(point)
        pulled = jacobian.T @ normals[k+1]
        if not bool(torch.isfinite(pulled).all()) or float(pulled.norm()) == 0:
            raise InterventionError("逆像分界的法向无定义")
        points[k], normals[k] = point, (pulled / pulled.norm()).detach()
    return torch.stack(points), torch.stack(normals)


def _tube_on_plateau(protection, start, end, radius):

    if hasattr(protection, "clearance") and hasattr(protection, "width"):

        clearance = float(protection.clearance(start).detach())
        lower = clearance - float((end-start).norm()) - radius
        return lower >= protection.width, {"method": "one_lipschitz_clearance_ball", "lower": lower, "plateau_threshold": protection.width, "radius": radius}
    if hasattr(protection, "inner_radius") and hasattr(protection, "center"):
        maximum = max(float((start-protection.center).norm()), float((end-protection.center).norm())) + radius
        return maximum <= protection.inner_radius, {"method": "convex_radial_plateau_tube", "maximum_radius": maximum, "inner_radius": protection.inner_radius, "tube_radius": radius}
    return False, {"method": "unavailable", "reason": "保护函数没有可用的连续平台管证书"}


def _potential_batches(system, values, endpoint, source_batch_size=32):
    size = _source_batch_size(source_batch_size)
    if values.ndim != 2 or not len(values):
        raise ValueError("物理势读出需要完整非空的源构型矩阵")

    return torch.cat([system.effective_u(batch, endpoint) for batch in values.split(size)])


def _smoothstep(value):
    value = min(1., max(0., value))
    return value**3 * (10 + value*(6*value-15))


def _make_loops(base, protection, points, normals, center, sigma, window, offset, eta, config):
    active = torch.nonzero(eta > 0).flatten().tolist()
    first, last = active[0] / base.n_steps, active[-1] / base.n_steps
    ramp = float(config.get("loop_ramp_time", 1/12))
    left, right = max(0., first-ramp), min(1., last+ramp)
    loops, certificates, boundaries = [], [], []
    radius = sigma * float(config.get("activity_tube_radius_sigma", 1.))
    for k in range(base.n_steps+1):
        t = k / base.n_steps
        factor = 1. if first <= t <= last else (_smoothstep((t-left)/(first-left)) if left < t < first else (_smoothstep((right-t)/(right-last)) if last < t < right else 0.))
        if k in (0, base.n_steps):
            factor = 0.
        displacement = factor * (center + offset*sigma*normals[k] - points[k])
        target = points[k] + displacement
        if factor > 0:
            valid, certificate = _tube_on_plateau(protection, points[k], target, radius)
            if not valid:
                raise InterventionError(f"路线 r={offset:+g} 节点 {k} 的连续平移管不在保护平台")
            loops.append(ProtectedTranslation(protection, displacement, max_microsteps=int(config.get("max_loop_microsteps", 10000))))
            certificates.append({"node": k, **certificate})
        else:
            loops.append(IdentityLoop())
        boundaries.append(target)
    return loops, certificates, torch.stack(boundaries)


def _local_stages(route, z, k, epsilon):
    if route.conjugated_kicks:
        base_input = route.loops[k].single_inverse(z)
        after_kick = base_input + route._increment(base_input, k, epsilon)
        physical_kick = route.loops[k].single_forward(after_kick)
        endpoint = route.loops[k+1].single_forward(route._base_step(after_kick, k))
        return physical_kick, endpoint
    after_kick = z + route._increment(z, k, epsilon)
    endpoint = route.loops[k+1].single_forward(route._base_step(route.loops[k].single_inverse(after_kick), k))
    return after_kick, endpoint


def _batch_stages(route, batch, k, epsilon):
    pairs = [_local_stages(route, point, k, epsilon) for point in batch]
    return torch.stack([row[0] for row in pairs]).detach(), torch.stack([row[1] for row in pairs]).detach()


def _source_batch_size(value):
    if type(value) is not int or value < 1:
        raise ValueError("轨迹计算的源构型批量必须为正整数")
    return value


@torch.no_grad()
def _trace(route, samples, epsilon):

    results = []
    for initial in samples:
        z, nodes = initial, [initial]
        for k in range(route.n_steps):
            kick, z = _local_stages(route, z, k, epsilon)
            nodes.extend((kick.detach(), z.detach()))
        results.append(torch.stack(nodes))
    return torch.stack(results)


@torch.no_grad()
def _dose_check(route, samples, epsilon, sigma, source_batch_size=32, check_budget=None):
    route.certify_epsilon(epsilon)
    route._validate_batch(samples)
    size = _source_batch_size(source_batch_size)
    maxima = []
    for batch in samples.split(size):
        off, on = batch.detach(), batch.detach()
        maximum = batch.new_zeros(len(batch))
        for k in range(route.n_steps):
            if check_budget is not None:
                check_budget()
            off_kick, off = _batch_stages(route, off, k, 0.)
            on_kick, on = _batch_stages(route, on, k, epsilon)
            maximum = torch.maximum(maximum, (on_kick-off_kick).norm(dim=-1))
            maximum = torch.maximum(maximum, (on-off).norm(dim=-1)).detach()
        maxima.append(maximum)
    displacement = torch.cat(maxima)
    if not bool(torch.isfinite(displacement).all()):
        raise InterventionError("D 剂量位移非有限")

    quantile, maximum = float(torch.quantile(displacement, .95)), float(displacement.max())
    return quantile <= .1*sigma, {"p95_maximum_path_displacement": quantile, "maximum_path_displacement": maximum,
                                   "limit": .1*sigma, "source_draws": len(samples), "trajectory_nodes": 2*route.n_steps+1,
                                   "source_batch_size": size, "trajectory_storage": "paired_streaming_maxima"}


@torch.no_grad()
def _primary_proxy_stream(route, samples, boundary, normals, sigma, eta, source_batch_size=32, check_budget=None):
    route._validate_batch(samples)
    total = samples.new_zeros(())
    for batch in samples.split(_source_batch_size(source_batch_size)):
        z = batch.detach()
        for k in range(route.n_steps):
            if check_budget is not None:
                check_budget()
            kick, z = _batch_stages(route, z, k, 0.)
            if not bool(eta[k] > 0):
                continue
            relative = kick-boundary[k]
            normal_distance = relative @ normals[k]
            tangential = relative-normal_distance[:, None]*normals[k]
            kernel = (normal_distance.abs() < .1*sigma)*torch.exp(-tangential.square().sum(-1)/(2*sigma*sigma))
            total = (total + eta[k]*kernel.sum()).detach()
    return total/len(samples)


@torch.no_grad()
def _geometric_prediction_stream(cell, samples, protection, source_batch_size=32, check_budget=None):

    route = cell["route"]
    route._validate_batch(samples)
    totals = {factor: samples.new_zeros(()) for factor in (.1, .05)}
    for batch in samples.split(_source_batch_size(source_batch_size)):
        z = batch.detach()
        for k in range(route.n_steps):
            if check_budget is not None:
                check_budget()
            if bool(cell["eta"][k] > 0):
                for factor in totals:
                    contribution = geometric_mass_prediction(z[None], cell["boundary"][k:k+1], cell["normals"][k:k+1],
                        cell["anchor_center"], cell["sigma"], cell["amplitude"], cell["eta"][k:k+1], cell["epsilon"],
                        protection, cell["geometry"], bandwidth_factor=factor)
                    totals[factor] = (totals[factor] + len(batch)*contribution).detach()
            _, z = _batch_stages(route, z, k, 0.)
    full, half = totals[.1]/len(samples), totals[.05]/len(samples)
    return full, {"slab_half_width_over_sigma": .1, "half_slab_width_over_sigma": .05,
                  "full_slab_prediction": float(full), "half_slab_prediction": float(half),
                  "half_minus_full": float(half-full), "source_draws": len(samples),
                  "trajectory_storage": "nodewise_exposure_accumulation"}


@dataclass
class MechanismPanel:
    base: object
    system: object
    protection: object
    cells: list[dict]
    config: dict
    development: dict


def _anchor_class(anchor):
    minima = anchor.get("minima", [])
    if len(minima) != 2:
        return None
    count = sum(bool(row["a"]) for row in minima)
    return "target" if count == 1 else ("nontarget" if count == 0 else None)


def _panel_memory_preflight(n_steps, dimension, anchors, config):


    estimate = int(anchors)*8*((17*int(n_steps)+9)*int(dimension)+12*int(n_steps))
    limit = config.get("panel_memory_limit_bytes")
    detected = None
    origin = "explicit_config" if limit is not None else "unassessed"
    if limit is None:
        try:
            total = int(os.sysconf("SC_PHYS_PAGES"))*int(os.sysconf("SC_PAGE_SIZE"))
            if total > 0:
                detected = total
        except (ValueError, OSError, AttributeError):
            pass
        for location in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
            try:
                if location.is_file():
                    text = location.read_text().strip()
                    if text.isdigit() and 0 < int(text) < 2**60:
                        detected = int(text) if detected is None else min(detected, int(text))
            except OSError:
                pass
        if detected is not None:
            limit = detected//2
            origin = "half_detected_physical_or_container_RAM"
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("panel_memory_limit_bytes必须是正整数")
    insufficient = limit is not None and estimate > limit
    return dict(status="resource_limit" if insufficient else ("planned_lower_bound_within_limit" if limit is not None else "unassessed"),
        planned_ready_tensor_lower_bound_bytes=estimate, selected_anchors=int(anchors), planned_ready_cells=12*int(anchors),
        fixed_panel_cells=48, n_steps=int(n_steps), shape_dimension=int(dimension), memory_limit_bytes=limit,
        detected_RAM_bytes=detected, limit_source=origin,
        scope="Planned all-ready-slot tensor residence lower estimate; excludes Python loop objects, AD workspace and source banks. Actual failed-slot residence may be smaller; being below this limit does not prove total memory fits.")


def prepare_panel(base, system, protection, anchors, D_samples, config=None):
    config = {} if config is None else dict(config)
    base = copy.deepcopy(base)
    if hasattr(base, "eval"):
        base.eval()
    if hasattr(base, "parameters"):
        for parameter in base.parameters():
            parameter.requires_grad_(False)
    source = _tensor(D_samples["source"] if isinstance(D_samples, dict) else D_samples)
    if source.ndim != 2 or source.shape[1] != system.dim or len(source) < 2:
        raise ValueError("面板需要当前体系的完整 D 源形状构型")
    if config.get("data_role", "D") not in ("D", "D_fit", "D_select"):
        raise ValueError("边界、剂量和主对必须在 D 上确定")
    cells = [dict(row) for row in panel_manifest() if row["system"] == system.name]
    if len(cells) != 48:
        raise ValueError("正式分子面板体系必须为 ALA4 或 ALDP")
    for cell in cells:
        cell.update(status="construction_failed", reason="缺少认证的 2+2 锚点")
    endpoints = _first(base.forward(source)).detach()
    baseline_a = float(system.regions(endpoints)["a"].to(torch.float64).mean())
    development = {"baseline_a": baseline_a, "deficit_a": None, "anchor_selection": [], "primary_pair": None,
                   "selection_role": "D", "geometry_coverage_denominator": 12, "base_frozen_copy": True,
                   "geometry_certificate_schema": "interval_heavy_basin_v1"}
    if isinstance(D_samples, dict) and D_samples.get("target") is not None:
        target = _tensor(D_samples["target"])
        development["deficit_a"] = float(system.regions(target)["a"].to(torch.float64).mean()) - baseline_a
    records = anchors.get("records", anchors.get("selected", [])) if isinstance(anchors, dict) else list(anchors)
    candidates = []
    for record in records:
        if record.get("status") != "certified" or not record.get("certificate", {}).get("certified", False):
            continue
        category = _anchor_class(record)
        if category is None:
            continue
        center, sigma = _tensor(record["z"]), float(record["sigma"])
        exposure = float(torch.exp(-(source-center).square().sum(-1)/(2*sigma*sigma)).mean()) if sigma > 0 else -math.inf
        candidates.append((category, -exposure, float(record.get("surface", 0)), int(record.get("start", 0)), record))
    chosen = []
    for category in ("target", "nontarget"):
        ranked = sorted((row for row in candidates if row[0] == category), key=lambda row: row[1:4])
        unique = []
        for row in ranked:
            if all(float((_tensor(row[4]["z"])-_tensor(other[4]["z"])).norm()) > float(config.get("anchor_chart_dedup_tolerance", 1e-8)) for other in unique):
                unique.append(row)
        chosen.extend(unique[:2] + [None] * max(0, 2-len(unique)))
    planned = _panel_memory_preflight(base.n_steps, system.dim, sum(row is not None for row in chosen), config)
    development["memory_preflight"] = planned
    if planned["status"] == "resource_limit":
        reason = (f"resource_limit:完整计划面板的tensor驻留估算下界{planned['planned_ready_tensor_lower_bound_bytes']}字节"
                  f"超过资源预算{planned['memory_limit_bytes']}字节；未计Python/AD。失败槽实际可能更小，"
                  "这不是几何或物理不可构造的结论。网格、样本和固定槽数保持原定义。")
        for cell in cells:
            cell.update(status="construction_failed", reason=reason, failure_type="resource_limit")
        return MechanismPanel(base, system, protection, cells, config, development)
    if not any(row is not None for row in chosen):
        return MechanismPanel(base, system, protection, cells, config, development)
    times = torch.arange(base.n_steps, dtype=torch.float64)/base.n_steps
    deltas = torch.full_like(times, 1/base.n_steps)
    primary_candidates = []
    for anchor_index, candidate in enumerate(chosen):
        if candidate is None:
            continue
        record = candidate[4]
        center, sigma, amplitude = _tensor(record["z"]), float(record["sigma"]), float(record["amplitude"])
        slots = [cell for cell in cells if cell["anchor"] == anchor_index]
        try:
            if not math.isfinite(sigma) or sigma <= 0 or float(protection(center)) < 1-1e-12:
                raise InterventionError("锚点带宽无定义或不在共同保护平台")
            reference_endpoint = center
            for k in range(base.n_steps//2, base.n_steps):
                reference_endpoint = _base_step(base, reference_endpoint, k).detach()
            endpoint, normal, active_patch, boundary_record = _regular_boundary(base, system, source, config, endpoints, reference_endpoint)
            points, normals = _track_boundary(base, system, endpoint, normal, config)
            construction = []
            for window in range(3):
                eta = window_weights(times, deltas, window)
                for offset in (-1., 0., 1., .5):
                    cell = next(row for row in slots if row["window"] == window and row["offset"] == offset)
                    try:
                        loops, tube, shifted_boundary = _make_loops(base, protection, points, normals, center, sigma, window, offset, eta, config)
                        route = ConjugateRoute(base, loops, protection, center, sigma, amplitude, eta,
                                               tolerance=float(config.get("inverse_tolerance", 1e-10)))
                        cell.update(route=route, boundary=shifted_boundary[:-1], normals=normals[:-1],
                                    anchor_center=center, sigma=sigma, amplitude=amplitude, eta=eta,
                                    continuous_tube_certificates=tube, boundary_record=boundary_record,
                                    active_patch=active_patch, status="dose_pending", reason="")
                        construction.append(cell)
                    except (ValueError, RuntimeError, FloatingPointError) as exc:
                        cell.update(status="construction_failed", reason=str(exc))
            main = [cell for cell in construction if cell["panel"] == "main"]
            if len(main) != 9:
                raise InterventionError("九条主路线未全部取得合法构造，无法冻结共同剂量")
            dose, dose_record = None, []
            for exponent in range(17):
                epsilon = sigma*sigma*2.**(-exponent)
                trial, valid = [], True
                for cell in main:
                    try:
                        passed, check = _dose_check(cell["route"], source, epsilon, sigma, config.get("source_batch_size", 32))
                        trial.append({"cell": cell["id"], **check})
                        valid &= passed
                    except (ValueError, RuntimeError, FloatingPointError) as exc:
                        valid = False
                        trial.append({"cell": cell["id"], "reason": str(exc)})
                dose_record.append({"exponent": exponent, "epsilon": epsilon, "passed": valid, "checks": trial})
                if valid:
                    dose = epsilon
                    break
            if dose is None:
                raise InterventionError("17 档规定剂量均未通过共同位移与收缩门")
            for cell in construction:
                try:
                    passed, check = _dose_check(cell["route"], source, dose, sigma, config.get("source_batch_size", 32))
                    half_passed, half_check = _dose_check(cell["route"], source, dose/2, sigma, config.get("source_batch_size", 32))
                    if not passed or not half_passed:
                        raise InterventionError("冻结剂量或半剂量未通过实际累计位移门")
                    cell["route"] = cell["route"].at_epsilon(dose)
                    qualification, geometry_certificate = certify_geometry(
                        cell["route"], system, cell["boundary"], cell["normals"], sigma,
                        cell["active_patch"], cell["eta"])
                    cell.update(status="ready", epsilon=dose, dose_diagnostics=check, half_dose_diagnostics=half_check,
                                dose_search=dose_record, geometry=qualification,
                                geometry_certificate=geometry_certificate)
                    if candidate[0] == "target" and cell["offset"] == 0.:
                        proxy = _primary_proxy_stream(cell["route"], source, cell["boundary"], cell["normals"], sigma,
                                                      cell["eta"], config.get("source_batch_size", 32))
                        primary_candidates.append((float(proxy), anchor_index, cell["window"]))
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    cell.update(status="construction_failed", reason=str(exc))
            development["anchor_selection"].append({"slot": anchor_index, "class": candidate[0], "surface": candidate[2],
                                                     "start": candidate[3], "D_exposure": -candidate[1], "boundary": boundary_record})
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            for cell in slots:
                cell.update(status="construction_failed", reason=str(exc))
    if primary_candidates:
        winner = sorted(primary_candidates, key=lambda row: (-row[0], row[1], row[2]))[0]
        development["primary_pair"] = {"anchor": winner[1], "window": winner[2], "offsets": [-1., 1.], "D_exposure": winner[0]}
    return MechanismPanel(base, system, protection, cells, config, development)


@dataclass
class SourceBatch:
    samples: torch.Tensor
    unit_ids: list[str]
    block_ids: list[int]
    role: str
    eligible_for_formal: bool = False

    def __post_init__(self):
        if self.samples.ndim != 2 or len(self.samples) == 0 or not torch.isfinite(self.samples).all():
            raise ValueError("预测／干预源批必须是完整的有限形状构型")
        if len(self.unit_ids) != len(self.samples) or len(self.block_ids) != len(self.samples):
            raise ValueError("每个源构型须保留独立 ladder 与推断块身份")
        assignment = {}
        for unit, block in zip(self.unit_ids, self.block_ids):
            if unit in assignment and assignment[unit] != block:
                raise ValueError("同一 ladder 不能拆成多个独立推断块")
            assignment[unit] = block
        if not assignment:
            raise ValueError("源批没有独立单位")

    @property
    def blocks(self):
        return sorted(set(self.block_ids))

    @property
    def units(self):
        return sorted(set(self.unit_ids))

    def block(self, block):
        mask = torch.tensor([value == block for value in self.block_ids], device=self.samples.device)
        return self.samples[mask]


def sources_from_bank(system, bank, *, inference_blocks=16):
    positions = _tensor(bank["positions"])
    if positions.ndim != 4 or len(bank["source_unit_ids"]) != positions.shape[0]:
        raise ValueError("源银行缺少完整 ladder/time/state/shape 身份")
    ladders, count = positions.shape[:2]
    if len(set(bank["source_unit_ids"])) != ladders:
        raise ValueError("源银行的 ladder 身份重复")
    if ladders != 2*inference_blocks:
        raise ValueError("disjoint_ladder_pairs 需要每个推断块恰好两条完整独立 ladder")
    indices = [i for i, state in enumerate(bank["states"]) if abs(float(state)-float(system.state(0))) < 1e-10]
    if len(indices) != 1:
        raise ValueError("源银行没有当前物理源端状态")
    samples = positions[:, :, indices[0]].reshape(-1, system.dim)
    units = [unit for unit in bank["source_unit_ids"] for _ in range(count)]
    blocks = [ladder//2 for ladder in range(ladders) for _ in range(count)]
    return SourceBatch(samples, units, blocks, bank.get("role", "unknown"), bool(bank.get("eligible_for_formal", False)))


def _panels_and_sources(panels, sources, config, role):
    panels = [panels] if isinstance(panels, MechanismPanel) else list(panels)
    if len({panel.system.name for panel in panels}) != len(panels):
        raise ValueError("同一分子体系不能重复进入固定面板")
    result = {}
    for panel in panels:
        entry = sources[panel.system.name] if isinstance(sources, dict) and panel.system.name in sources else sources
        if not isinstance(entry, SourceBatch):
            entry = sources_from_bank(panel.system, entry, inference_blocks=int(config.get("inference_blocks", 16)))
        if entry.role != role:
            raise ValueError(f"{role} 驱动只接受其自己的独立源银行")
        if len({len(entry.block(block)) for block in entry.blocks}) != 1:
            raise ValueError("非线性联合推断需要等大小的预定独立块；不能对不等块直接平均")
        result[panel.system.name] = entry
    counts = {len(entry.blocks) for entry in result.values()}
    if len(counts) != 1:
        raise ValueError("固定面板的两体系须使用相同数量的预定推断块")
    return panels, result


def _save(path, value):
    path = Path(path)
    if path.name in ("predictions.pt", "predictions-working.pt", "interventions.pt", "interventions-working.pt") and value.get("artifact_storage_schema") == _ARTIFACT_STORAGE_SCHEMA:
        role = value["role"]
        relative = Path("raw")/role/"context.pt"
        context = path.parent/relative
        if not context.exists():
            static = {key: value[key] for key in ("panels", "sources") if key in value}
            save_checkpoint(context, dict(artifact_storage_schema=_ARTIFACT_STORAGE_SCHEMA,
                geometry_certificate_schema=_GEOMETRY_SCHEMA, statistics_schema=_STATISTICS_SCHEMA,
                role=role, config=value["config"], **static))
        value["context_path"] = relative.as_posix()
        value["context_bytes"] = context.stat().st_size

        compact = {key: item for key, item in value.items() if key not in ("panels", "sources")}
        save_checkpoint(path, compact)
    else:
        save_checkpoint(path, value)


def load_mechanism_result(path):

    path = Path(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if "context_path" in value:
        context_path = _raw_path(path.parent, value["context_path"])
        if context_path.stat().st_size != value.get("context_bytes"):
            raise ValueError("冻结的真实模型/源构型上下文长度发生变化")
        context = torch.load(context_path, map_location="cpu", weights_only=False)
        if any(context.get(key) != value.get(key) for key in ("artifact_storage_schema", "geometry_certificate_schema", "role", "config")):
            raise ValueError("冻结上下文与P/I运行契约不一致")
        value.update({key: context[key] for key in ("panels", "sources") if key in context})
    return value


_ARTIFACT_STORAGE_SCHEMA = "artifact_storage_v1"
_GEOMETRY_SCHEMA = "interval_heavy_basin_v1"


def _raw_path(directory, relative):
    root = Path(directory).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("原始产物必须使用运行目录内的相对路径")
    path = root/relative
    if not path.is_file():
        raise ValueError(f"真实原始产物缺失：{relative}")
    return path


def load_raw_block(directory, record, *, expected_role=None):

    if "raw_path" not in record:
        if "response" in record and "endpoint" in record["response"]:
            return {"role": "P", "response": record["response"], "labels": record.get("labels")}
        if "result" in record and "endpoint" in record["result"].get("baseline", {}):
            return {"role": "I", "result": record["result"]}
        raise ValueError("紧凑记录没有对应的真实原始产物")
    path = _raw_path(directory, record["raw_path"])
    if "raw_bytes" in record and path.stat().st_size != record["raw_bytes"]:
        raise ValueError("真实原始产物长度与原子写入记录不符")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("artifact_storage_schema") != _ARTIFACT_STORAGE_SCHEMA or (expected_role and value.get("role") != expected_role):
        raise ValueError("原始产物的存储协议或数据角色不符")
    _raw_path(directory, value["source_path"])
    return value


def _validate_raw_artifacts(result, directory):
    if result.get("artifact_storage_schema") != _ARTIFACT_STORAGE_SCHEMA:
        raise ValueError("旧输出未采用artifact_storage_v1，不能执行当前P/I协议恢复")
    context = _raw_path(directory, result.get("context_path", ""))
    if context.stat().st_size != result.get("context_bytes"):
        raise ValueError("冻结模型/源构型上下文长度发生变化")
    for record in result.get("raw_sources", {}).values():
        path = _raw_path(directory, record["path"])
        if path.stat().st_size != record["bytes"]:
            raise ValueError("完整源构型产物长度发生变化")
    for cell in result.get("cells", {}).values():
        if cell.get("status") == "computed" and not cell.get("blocks"):
            raise ValueError("已完成的cell缺少真实原始块")
        for block in cell.get("blocks", []):
            path = _raw_path(directory, block.get("raw_path", ""))
            if path.stat().st_size != block.get("raw_bytes"):
                raise ValueError("真实原始cell-block产物长度发生变化")
            _raw_path(directory, block["source_path"])


def _raw_output_paths(result, directory):
    paths = {str(_raw_path(directory, result["context_path"]))}
    paths.update(str(_raw_path(directory, row["path"])) for row in result.get("raw_sources", {}).values())
    paths.update(str(_raw_path(directory, block["raw_path"])) for cell in result.get("cells", {}).values() for block in cell.get("blocks", []))
    return sorted(paths)


def _ensure_raw_sources(result, sources, directory, role):
    refs = result.setdefault("raw_sources", {})
    for name, source in sources.items():
        for block in source.blocks:
            relative = Path("raw")/role/"sources"/name/f"block-{block}.pt"
            path = Path(directory)/relative
            mask = [value == block for value in source.block_ids]
            ids = [unit for unit, included in zip(source.unit_ids, mask) if included]
            samples = source.block(block).detach().cpu()
            value = dict(artifact_storage_schema=_ARTIFACT_STORAGE_SCHEMA, role=role, molecule=name,
                         block=block, source_unit_ids=ids, samples=samples)
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if any(saved.get(key) != value[key] for key in ("artifact_storage_schema", "role", "molecule", "block", "source_unit_ids")) or not torch.equal(saved["samples"], samples):
                    raise ValueError("原始源块与冻结的真实构型或完整ladder身份不符")
            else:
                _save(path, value)
            refs[f"{name}/{block}"] = {"path": relative.as_posix(), "bytes": path.stat().st_size}
    return refs


def _raw_contract(result, cell, block, name):
    return dict(artifact_storage_schema=_ARTIFACT_STORAGE_SCHEMA, statistics_schema=_STATISTICS_SCHEMA,
        geometry_certificate_schema=_GEOMETRY_SCHEMA, role=result["role"], cell_id=cell["id"],
        block=block, source_path=result["raw_sources"][f"{name}/{block}"]["path"],
        config=result["config"], epsilon=float(cell["epsilon"]))


def _read_cached_raw(directory, relative, contract):
    path = Path(directory)/relative
    if not path.exists():
        return None
    value = torch.load(path, map_location="cpu", weights_only=False)
    if any(value.get(key) != expected for key, expected in contract.items()):
        raise ValueError("已有实际cell-block产物与冻结协议或输入身份不符")
    _raw_path(directory, value["source_path"])
    return value


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def _compact_prediction(payload, relative, directory):
    return dict(block=payload["block"], predictions=payload["predictions"], labels=payload["labels"],
        response={key: payload["response"][key] for key in ("score", "dot_logweight", "logweight")},
        geometric_slab_diagnostic=payload.get("geometric_slab_diagnostic"), raw_path=relative.as_posix(),
        raw_bytes=(Path(directory)/relative).stat().st_size, source_path=payload["source_path"])


def _compact_intervention(payload, relative, directory):
    value = payload["result"]
    compact = {key: ({name: item for name, item in branch.items() if name != "endpoint"}
                     if key in ("baseline", "full", "half") else branch) for key, branch in value.items()}
    return dict(block=payload["block"], result=compact, raw_path=relative.as_posix(),
                raw_bytes=(Path(directory)/relative).stat().st_size, source_path=payload["source_path"])


def _block_manifest(sources):
    return {system: {str(block): sorted({unit for unit, value in zip(source.unit_ids, source.block_ids) if value == block})
                     for block in source.blocks} for system, source in sources.items()}


def _prediction_names():
    names = [f"{cell['id']}/{field}" for cell in panel_manifest()
             for field in ("mass", "mass_b", "mass_remainder", "log_risk", "conditional_h2")]
    names.extend(f"{cell['id']}/geometric_mass" for cell in panel_manifest() if cell["panel"] == "main")
    return names


def _intervals_by_name(rows, names, family, events=None):
    matrix = np.asarray([[row.get(name, math.nan) for name in names] for row in rows], dtype=float)
    intervals = bonferroni_t(matrix, family, events=events)
    return dict(zip(names, intervals)), matrix


_STATISTICS_SCHEMA = "global_moments_cluster_delta_v1"
_STATISTICS_SCOPE = (
    "质量使用配对块均值；非线性读出在全体块的共同原始矩上计算 plug-in 点估计，"
    "并以一阶 cluster influence 伪值构造同一 Student-t/Bonferroni 区间。"
    "覆盖依赖独立块、相应矩及 delta-method 条件，不是非线性量的精确有限样本 t 保证。"
)


def _equal_block_size(blocks, values):
    lengths = [len(values(block)) for block in blocks]
    if not lengths or min(lengths) < 1 or len(set(lengths)) != 1:
        raise ValueError("共同矩统计必须保留全部等大小的独立块")


def _delta_rows(moments, functional):

    moments = torch.as_tensor(moments, dtype=torch.float64)
    if moments.ndim != 2 or not len(moments) or not torch.isfinite(moments).all():
        raise ValueError("原始块矩必须是完整有限的二维数组")
    mean = moments.mean(0)
    point = functional(mean)
    if not torch.isfinite(point):
        return math.nan, np.full(len(moments), math.nan)
    gradient = torch.func.grad(functional)(mean)
    if not torch.isfinite(gradient).all():
        return math.nan, np.full(len(moments), math.nan)
    rows = point+(moments-mean)@gradient
    return float(point), rows.detach().cpu().numpy()


def _prediction_moment_statistics(blocks, epsilon):

    _equal_block_size(blocks, lambda block: block["response"]["logweight"])
    logweights = [_tensor(block["response"]["logweight"]).detach().cpu() for block in blocks]
    if not all(bool(torch.isfinite(value).all()) for value in logweights):
        raise ValueError("P 的完整权重必须有限")
    shift = 2*max(float(value.max()) for value in logweights)
    risk_moments, shape_moments = [], []
    for block, logweight in zip(blocks, logweights):
        response = block["response"]
        score = _tensor(response["score"]).detach().cpu()
        derivative = _tensor(response["dot_logweight"]).detach().cpu()
        indicator = _tensor(block["labels"] == 0).detach().cpu()
        if score.shape != logweight.shape or derivative.shape != logweight.shape or indicator.shape != logweight.shape:
            raise ValueError("P 原始 score、权重导数和硬区域必须逐样本配对")
        weight2 = torch.exp(2*logweight-shift)
        risk_moments.append(torch.stack((weight2.mean(), (weight2*derivative).mean())))
        shape_moments.append(torch.stack((indicator.mean(), (indicator*score).mean(), (indicator*score.square()).mean())))
    risk_point, risk_rows = _delta_rows(torch.stack(risk_moments), lambda m: 2*epsilon*m[1]/m[0])
    shape_matrix = torch.stack(shape_moments)
    if float(shape_matrix[:, 0].mean()) > 0:
        shape_point, shape_rows = _delta_rows(shape_matrix, lambda m: epsilon**2*(m[2]/m[0]-(m[1]/m[0])**2)/8)
    else:
        shape_point, shape_rows = math.nan, np.full(len(blocks), math.nan)
    return {"log_risk": {"point": risk_point, "rows": risk_rows},
            "conditional_h2": {"point": shape_point, "rows": shape_rows},
            "logweight_square_shift": shift}


def _intervention_risk_statistics(blocks):
    _equal_block_size(blocks, lambda block: block["result"]["baseline"]["logweight"])
    branches = ("baseline", "full", "half")
    offsets = []
    columns = []
    for branch in branches:
        logweights = [_tensor(block["result"][branch]["logweight"]).detach().cpu() for block in blocks]
        if any(len(value) != len(block["result"]["baseline"]["logweight"]) for value, block in zip(logweights, blocks)):
            raise ValueError("I 满／半剂量与基准必须是同一完整初态块")
        if not all(bool(torch.isfinite(value).all()) for value in logweights):
            raise ValueError("I 的每个真实分支必须保留完整有限的权重")
        shift = 2*max(float(value.max()) for value in logweights)
        offsets.append(shift)
        columns.append(torch.stack([torch.exp(2*value-shift).mean() for value in logweights]))
    moments = torch.stack(columns, 1)
    offsets = torch.tensor(offsets, dtype=torch.float64)


    full_point, full_rows = _delta_rows(moments, lambda m: (m[1].log()+offsets[1])-(m[0].log()+offsets[0]))
    half_point, half_rows = _delta_rows(moments, lambda m: (m[1].log()+offsets[1])+(m[0].log()+offsets[0])-2*(m[2].log()+offsets[2]))
    return {"full": {"point": full_point, "rows": full_rows},
            "full_minus_twice_half": {"point": half_point, "rows": half_rows},
            "logweight_square_shifts": offsets.tolist()}


def _intervention_shape_statistics(blocks, region=0):
    _equal_block_size(blocks, lambda block: block["result"]["baseline"]["logweight"])
    moments = []
    for block in blocks:
        full, half = (block["result"][key] for key in ("shape_full", "shape_half"))
        if not torch.equal(_tensor(full["rho0"]), _tensor(half["rho0"])):
            raise ValueError("两剂量的 Hellinger 读出没有共用同一配对基准")
        restricted = [(_tensor(value["affinity_integrand"])*(_tensor(value["labels"]) == region)).mean() for value in (full, half)]
        moments.append(torch.stack((_tensor(full["rho0"])[region], _tensor(full["rhoe"])[region],
                                    _tensor(half["rhoe"])[region], *restricted)))
    moments = torch.stack(moments).detach().cpu()
    if bool((moments[:, :3].mean(0) <= 0).any()):
        missing = {"point": math.nan, "rows": np.full(len(blocks), math.nan)}
        return {"full": missing.copy(), "full_minus_four_half": missing.copy()}
    full = lambda m: 1-m[3]/torch.sqrt(m[0]*m[1])
    half = lambda m: 1-m[4]/torch.sqrt(m[0]*m[2])
    full_point, full_rows = _delta_rows(moments, full)
    residual_point, residual_rows = _delta_rows(moments, lambda m: full(m)-4*half(m))
    return {"full": {"point": full_point, "rows": full_rows},
            "full_minus_four_half": {"point": residual_point, "rows": residual_rows}}


def run_predictions(panels, P_sources, output_dir, config=None, *, resume=False):

    config = {} if config is None else dict(config)
    panels, sources = _panels_and_sources(panels, P_sources, config, "P")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path, working = directory/"predictions.pt", directory/"predictions-working.pt"
    if path.exists():
        if not resume:
            raise FileExistsError("已有冻结预测，继续须显式 resume")
        previous = load_mechanism_result(path)
        _validate_resumed_sources(previous, sources, config, directory)
        return previous
    seals = RoleSeal(directory/"roles.json")
    seals.open("P", sorted(unit for source in sources.values() for unit in source.units))
    if resume and working.exists():
        result = load_mechanism_result(working)
        _validate_resumed_sources(result, sources, config, directory)
        panels = result["panels"]
    else:
        result = {"role": "P", "panels": panels, "sources": sources, "config": config,
                  "cells": {}, "manifest": panel_manifest(), "block_manifest": _block_manifest(sources),
                  "sealed": False, "status": "running", "shape_selected": {}, "same_side_selected": {},
                  "artifact_storage_schema": _ARTIFACT_STORAGE_SCHEMA, "geometry_certificate_schema": _GEOMETRY_SCHEMA}
        raw_root = directory/"raw"/"P"
        if raw_root.exists() and list(raw_root.rglob("*.pt")):
            context_path = raw_root/"context.pt"
            if not resume or not context_path.is_file():
                raise ValueError("已有P原始产物但没有可恢复的原运行契约")
            context = torch.load(context_path, map_location="cpu", weights_only=False)
            if context.get("artifact_storage_schema") != _ARTIFACT_STORAGE_SCHEMA or context.get("geometry_certificate_schema") != _GEOMETRY_SCHEMA or context.get("role") != "P" or context.get("config") != config:
                raise ValueError("已有P上下文不匹配当前冻结协议")
            for name, source in sources.items():
                old = context["sources"].get(name)
                if old is None or old.unit_ids != source.unit_ids or old.block_ids != source.block_ids or not torch.equal(old.samples, source.samples):
                    raise ValueError("已有P上下文的完整源构型身份不符")
            result.update({key: context[key] for key in ("panels", "sources") if key in context})
            panels = result["panels"]
        _save(working, result)
    _ensure_raw_sources(result, sources, directory, "P")
    _save(working, result)
    for panel in panels:
        source = sources[panel.system.name]
        u0 = lambda z: _potential_batches(panel.system, z, 0, panel.config.get("source_batch_size", 32))
        u1 = lambda z: _potential_batches(panel.system, z, 1, panel.config.get("source_batch_size", 32))
        for cell in panel.cells:
            if cell["id"] in result["cells"]:
                continue
            record = {"status": cell["status"], "reason": cell.get("reason", ""), "blocks": []}
            if cell["status"] == "ready":
                try:
                    for block in source.blocks:
                        relative = Path("raw")/"P"/cell["id"]/f"block-{block}.pt"
                        contract = _raw_contract(result, cell, block, panel.system.name)
                        cached = _read_cached_raw(directory, relative, contract)
                        if cached is not None:
                            record["blocks"].append(_compact_prediction(cached, relative, directory))
                            del cached
                            continue
                        samples = source.block(block)
                        response = response_samples(cell["route"], samples, u0, u1)
                        labels = main_labels(panel.system, response["endpoint"])
                        if bool((labels < 0).any()):
                            raise InterventionError("P 的硬区域标签不确定")
                        weights = torch.softmax(2*response["logweight"], 0)
                        predictions = {"mass" if region == 0 else ("mass_b" if region == 1 else "mass_remainder"):
                                       float(cell["epsilon"]*((labels == region)*response["score"]).mean()) for region in range(3)}
                        predictions["log_risk"] = float(2*cell["epsilon"]*(weights*response["dot_logweight"]).sum())
                        in_a = response["score"][labels == 0]
                        predictions["conditional_h2"] = float(cell["epsilon"]**2*in_a.var(unbiased=False)/8) if len(in_a) else math.nan
                        slab_diagnostic = None
                        if cell["geometry"].qualified and cell["panel"] == "main":
                            prediction, slab_diagnostic = _geometric_prediction_stream(cell, samples, panel.protection,
                                panel.config.get("source_batch_size", 32))
                            predictions["geometric_mass"] = float(prediction)
                        payload = dict(contract, predictions=predictions, labels=labels.detach().cpu(),
                                       response=_cpu(response), geometric_slab_diagnostic=slab_diagnostic)
                        _save(directory/relative, payload)
                        record["blocks"].append(_compact_prediction(payload, relative, directory))
                        del response, payload
                    record["status"] = "computed"
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    record.update(status="prediction_failed", reason=str(exc))
            result["cells"][cell["id"]] = record
            _save(working, result)
    block_count = len(next(iter(sources.values())).blocks)
    rows = [{} for _ in range(block_count)]
    cell_epsilon = {cell["id"]: cell.get("epsilon") for panel in panels for cell in panel.cells}
    panel_systems = {panel.system.name: panel.system for panel in panels}
    point_estimates = {}
    for identifier, record in result["cells"].items():
        if record["status"] == "computed":


            for block in record["blocks"]:
                if "labels" not in block:
                    block["labels"] = main_labels(panel_systems[identifier.split("-a", 1)[0]], block["response"]["endpoint"])
            try:
                nonlinear = _prediction_moment_statistics(record["blocks"], cell_epsilon[identifier])
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                record.update(status="prediction_failed", reason=str(exc))
                continue
            for row, block in zip(rows, record["blocks"]):
                row.update({identifier+"/"+key: value for key, value in block["predictions"].items()})
            for field in ("log_risk", "conditional_h2"):
                point_estimates[identifier+"/"+field] = nonlinear[field]["point"]
                for row, value in zip(rows, nonlinear[field]["rows"]):
                    row[identifier+"/"+field] = float(value)
    names = _prediction_names()
    result["intervals"], result["matrix"] = _intervals_by_name(rows, names, "P", events=len(names))
    result["interval_names"] = names
    result.update(statistics_schema=_STATISTICS_SCHEMA, statistics_scope=_STATISTICS_SCOPE,
                  nonlinear_point_estimates=point_estimates)
    result["numerical_error_bounds"] = {}
    bound = config.get("numerical_error_bound")
    if bound is not None:
        if not math.isfinite(float(bound)) or float(bound) < 0:
            raise ValueError("数值误差界须由本次运行验证为有限非负数")
        result["numerical_error_bounds"] = {name: float(bound) for name in names}
    result["geometry_coverage"] = {panel.system.name: {
        "qualified_groups": len({(cell["anchor"], cell["window"]) for cell in panel.cells
                                  if cell["status"] == "ready" and cell["panel"] == "main" and cell["geometry"].qualified}),
        "total_groups": 12} for panel in panels}
    for panel in panels:
        candidates = []
        for anchor in range(4):
            for window in range(3):
                paired = [next(cell for cell in panel.cells if cell["anchor"] == anchor and cell["window"] == window and cell["offset"] == offset) for offset in (1., .5)]
                candidates.append({"data_role": "P", "anchor": anchor, "window": window,
                                   "mass": [result["intervals"][cell["id"]+"/mass"] for cell in paired],
                                   "log_risk": [result["intervals"][cell["id"]+"/log_risk"] for cell in paired],
                                   "cell_ids": [cell["id"] for cell in paired]})
        deficit = panel.development["deficit_a"]
        result["same_side_selected"][panel.system.name] = select_same_side_pair(candidates, deficit) if deficit is not None and deficit > 0 else None
        selected = {}
        for window in range(3):
            qualified = []
            for cell in panel.cells:
                if cell["window"] != window or cell["panel"] != "main":
                    continue
                mass = [result["intervals"][cell["id"]+"/"+key] for key in ("mass", "mass_b", "mass_remainder")]
                shape = result["intervals"][cell["id"]+"/conditional_h2"]
                if all(value.finite and value.lower >= -DELTA_MASS and value.upper <= DELTA_MASS for value in mass) and shape.finite and shape.lower > .01:
                    qualified.append((shape.lower, cell["anchor"], cell["offset"], cell["id"]))
            if qualified:
                selected[window] = sorted(qualified, key=lambda row: (-row[0], row[1], row[2]))[0][3]
        result["shape_selected"][panel.system.name] = selected
    for manifest in result["manifest"]:
        record = result["cells"].get(manifest["id"])
        if record:
            manifest.update(status=record["status"], reason=record.get("reason", ""))
    result.update(status="frozen_predictions", sealed=True,
                  eligible_for_formal=bound is not None and all(source.eligible_for_formal for source in sources.values()),
                  inference_blocks=block_count, prediction_events=len(names))
    _validate_raw_artifacts(result, directory)
    result["raw_storage"] = {"files": len(_raw_output_paths(result, directory)), "bytes": sum(Path(value).stat().st_size for value in _raw_output_paths(result, directory))}
    _save(path, result)
    seals.seal("P", [str(path.resolve())]+_raw_output_paths(result, directory))
    return result


def _validate_resumed_sources(previous, sources, config, directory=None):
    if previous.get("sealed") and previous.get("statistics_schema") != _STATISTICS_SCHEMA:
        raise ValueError("检查点的统计定义不匹配，不能恢复")
    if previous.get("artifact_storage_schema") != _ARTIFACT_STORAGE_SCHEMA or previous.get("geometry_certificate_schema") != _GEOMETRY_SCHEMA:
        raise ValueError("旧存储或几何协议仅可只读，不能作为artifact_storage_v1继续执行")
    if directory is None:
        raise ValueError("恢复必须核对实际运行目录中的全部原始产物")
    _validate_raw_artifacts(previous, directory)
    if previous["config"] != config or set(previous["sources"]) != set(sources):
        raise ValueError("断点恢复不可更改已冻结的运行配置或体系")
    for name, source in sources.items():
        old = previous["sources"][name]
        if source.unit_ids != old.unit_ids or source.block_ids != old.block_ids or not torch.equal(source.samples, old.samples):
            raise ValueError("断点恢复不可更换源构型或完整 ladder 配对")


def _geometric_advantage_rows(predictions, result, panels, rows):
    uncertainty = {}
    for panel in panels:
        participating = [cell for cell in panel.cells if cell["status"] == "ready"
                         and cell["panel"] == "main" and cell["geometry"].qualified]
        values = [[] for _ in rows]
        errors, complete = [], bool(participating)
        for cell in participating:
            identifier = cell["id"]
            group = [row for row in panel.cells if row["anchor"] == cell["anchor"]
                     and row["window"] == cell["window"] and row["panel"] == "main"]
            geometric = predictions["intervals"][identifier+"/geometric_mass"]
            group_intervals = [predictions["intervals"][row["id"]+"/mass"] for row in group]
            record = result["cells"].get(identifier, {})
            keys = [identifier+"/geometric_mass"] + [row["id"]+"/mass" for row in group]
            bounds = [predictions["numerical_error_bounds"].get(key) for key in keys]
            if (not geometric.finite or not all(value.finite for value in group_intervals)
                    or any(bound is None for bound in bounds) or record.get("status") != "computed"):
                complete = False
                break
            geom_point = .5*(geometric.lower+geometric.upper)
            blind_point = sum(.5*(value.lower+value.upper) for value in group_intervals)/3
            geom_set = geometric.widen(bounds[0])
            blind_lower = sum(value.lower-bound for value, bound in zip(group_intervals, bounds[1:]))/3
            blind_upper = sum(value.upper+bound for value, bound in zip(group_intervals, bounds[1:]))/3
            dg = max(abs(geom_set.lower-geom_point), abs(geom_set.upper-geom_point))
            db = max(abs(blind_lower-blind_point), abs(blind_upper-blind_point))
            errors.append(2*(dg+db) + dg*(2*abs(geom_point)+dg) + db*(2*abs(blind_point)+db))
            for output, block in zip(values, record["blocks"]):
                actual = float(block["result"]["mass_change"][0])
                output.append((actual-geom_point)**2-(actual-blind_point)**2)
        if complete:
            key = panel.system.name+"/geometric_squared_error_difference"
            for row, output in zip(rows, values):
                row[key] = float(np.mean(output))
            uncertainty[key] = float(np.mean(errors))
    return uncertainty


def run_interventions(predictions, I_sources, output_dir, config=None, *, resume=False):

    config = {} if config is None else dict(config)
    directory = Path(output_dir)
    if not predictions.get("sealed") or not (directory/"predictions.pt").is_file():
        raise ValueError("必须先完成并封存实际 P 输出")

    predictions = load_mechanism_result(directory/"predictions.pt")
    if not predictions.get("sealed"):
        raise ValueError("磁盘上的 P 尚未封存")
    if predictions.get("statistics_schema") != _STATISTICS_SCHEMA:
        raise ValueError("P 的非线性预测未采用全局共同矩，不能进入当前独立干预推断")
    if predictions.get("geometry_certificate_schema") != _GEOMETRY_SCHEMA:
        raise ValueError("P的几何协议过旧，不能按当前真实全邻域证书进入I")
    _validate_raw_artifacts(predictions, directory)
    panels, sources = _panels_and_sources(predictions["panels"], I_sources, config, "I")
    seals = RoleSeal(directory/"roles.json")
    seals.open("I", sorted(unit for source in sources.values() for unit in source.units))
    path, working = directory/"interventions.pt", directory/"interventions-working.pt"
    if path.exists():
        if not resume:
            raise FileExistsError("已有独立干预输出，继续须显式 resume")
        previous = load_mechanism_result(path)
        _validate_resumed_sources(previous, sources, config, directory)
        return previous
    if resume and working.exists():
        result = load_mechanism_result(working)
        _validate_resumed_sources(result, sources, config, directory)
    else:
        result = {"role": "I", "sources": sources, "config": config, "cells": {}, "baseline_a": {},
                  "manifest": panel_manifest(), "block_manifest": _block_manifest(sources), "sealed": False,
                  "artifact_storage_schema": _ARTIFACT_STORAGE_SCHEMA, "geometry_certificate_schema": _GEOMETRY_SCHEMA}
        raw_root = directory/"raw"/"I"
        if raw_root.exists() and list(raw_root.rglob("*.pt")):
            context_path = raw_root/"context.pt"
            if not resume or not context_path.is_file():
                raise ValueError("已有I原始产物但没有可恢复的原运行契约")
            context = torch.load(context_path, map_location="cpu", weights_only=False)
            if context.get("artifact_storage_schema") != _ARTIFACT_STORAGE_SCHEMA or context.get("geometry_certificate_schema") != _GEOMETRY_SCHEMA or context.get("role") != "I" or context.get("config") != config:
                raise ValueError("已有I上下文不匹配当前冻结协议")
            for name, source in sources.items():
                old = context["sources"].get(name)
                if old is None or old.unit_ids != source.unit_ids or old.block_ids != source.block_ids or not torch.equal(old.samples, source.samples):
                    raise ValueError("已有I上下文的完整源构型身份不符")
            result.update({key: context[key] for key in ("panels", "sources") if key in context})

        _save(working, result)
    _ensure_raw_sources(result, sources, directory, "I")
    _save(working, result)
    for panel in panels:
        source = sources[panel.system.name]
        u0 = lambda z: _potential_batches(panel.system, z, 0, panel.config.get("source_batch_size", 32))
        u1 = lambda z: _potential_batches(panel.system, z, 1, panel.config.get("source_batch_size", 32))
        result["baseline_a"][panel.system.name] = [float((main_labels(panel.system, _first(panel.base.forward(source.block(block)))) == 0).to(torch.float64).mean()) for block in source.blocks]
        selected = predictions["shape_selected"].get(panel.system.name, {})
        for cell in panel.cells:
            if cell["id"] in result["cells"]:
                continue
            record = {"status": cell["status"], "reason": cell.get("reason", ""), "blocks": []}
            if cell["status"] == "ready":
                try:
                    for block in source.blocks:
                        relative = Path("raw")/"I"/cell["id"]/f"block-{block}.pt"
                        contract = _raw_contract(result, cell, block, panel.system.name)
                        payload = _read_cached_raw(directory, relative, contract)
                        if payload is None:
                            value = finite_intervention(cell["route"], source.block(block), u0, u1,
                                    lambda z: main_labels(panel.system, z), 3, include_shape=cell["id"] in selected.values())
                            payload = dict(contract, result=_cpu(value))
                            _save(directory/relative, payload)
                            del value
                        record["blocks"].append(_compact_intervention(payload, relative, directory))
                        del payload
                    record["status"] = "computed"
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    record.update(status="intervention_failed", reason=str(exc))
            result["cells"][cell["id"]] = record
            _save(working, result)
    block_count = len(next(iter(sources.values())).blocks)
    rows = [{} for _ in range(block_count)]
    for system, baselines in result["baseline_a"].items():
        for row, value in zip(rows, baselines):
            row[system+"/baseline_a"] = value
    for identifier, record in result["cells"].items():
        if record["status"] != "computed":
            continue
        try:
            risk_statistics = _intervention_risk_statistics(record["blocks"])
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            record.update(status="intervention_failed", reason=str(exc))
            continue
        record["nonlinear_point_estimates"] = {key: value["point"] for key, value in risk_statistics.items() if isinstance(value, dict)}
        for row, block in zip(rows, record["blocks"]):
            value = block["result"]
            row[identifier+"/mass/full"] = float(value["mass_change"][0])
            row[identifier+"/mass/full_minus_twice_half"] = float(value["mass_half_dose_residual"][0])
        for contrast in ("full", "full_minus_twice_half"):
            for row, value in zip(rows, risk_statistics[contrast]["rows"]):
                row[identifier+"/log_risk/"+contrast] = float(value)
    for panel in panels:
        selected = predictions["shape_selected"].get(panel.system.name, {})
        for shape, identifier in selected.items():
            record = result["cells"].get(identifier, {})
            if record.get("status") != "computed":
                continue
            try:
                shape_statistics = _intervention_shape_statistics(record["blocks"])
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                record["shape_statistics_failure"] = str(exc)
                continue
            record["shape_point_estimates"] = {key: value["point"] for key, value in shape_statistics.items()}
            for row, block in zip(rows, record["blocks"]):
                value = block["result"]
                prefix = f"{panel.system.name}/shape{shape}/"
                for k, region in enumerate(("a", "b", "remainder")):
                    row[prefix+"mass_"+region] = float(value["mass_change"][k])
            for field, contrast in (("h2", "full"), ("h2_full_minus_four_half", "full_minus_four_half")):
                for row, value in zip(rows, shape_statistics[contrast]["rows"]):
                    row[f"{panel.system.name}/shape{shape}/"+field] = float(value)
    result["geometric_prediction_uncertainty_bounds"] = _geometric_advantage_rows(predictions, result, panels, rows)
    names = intervention_interval_names()
    result["intervals"], result["matrix"] = _intervals_by_name(rows, names, "I")
    for key, error in result["geometric_prediction_uncertainty_bounds"].items():
        result["intervals"][key] = result["intervals"][key].widen(error)
    measured = {cell["id"]+"/"+readout: result["intervals"][cell["id"]+"/"+readout+"/full"]
                for cell in panel_manifest() for readout in ("mass", "log_risk")}
    half = {cell["id"]+"/"+readout: result["intervals"][cell["id"]+"/"+readout+"/full_minus_twice_half"]
            for cell in panel_manifest() for readout in ("mass", "log_risk")}
    result["agreement"] = panel_agreement(predictions["intervals"], measured, half,
                                             numerical_error_bounds=predictions["numerical_error_bounds"])
    for manifest in result["manifest"]:
        record = result["cells"].get(manifest["id"])
        if record:
            manifest.update(status=record["status"], reason=record.get("reason", ""))
    result.update(status="completed_finite_interventions", sealed=True, inference_blocks=block_count,
                  interval_names=names, family_events=418,
                  statistics_schema=_STATISTICS_SCHEMA, statistics_scope=_STATISTICS_SCOPE,
                  eligible_for_formal=predictions["eligible_for_formal"] and all(source.eligible_for_formal for source in sources.values()),
                  geometry_status={panel.system.name: negative_gate(result["intervals"][panel.system.name+"/geometric_squared_error_difference"]) for panel in panels},
                  reference_status="R not opened; deficit and repair claims require later independent-reference projection")
    _validate_raw_artifacts(result, directory)
    result["raw_storage"] = {"files": len(_raw_output_paths(result, directory)), "bytes": sum(Path(value).stat().st_size for value in _raw_output_paths(result, directory))}
    _save(path, result)
    seals.seal("I", [str(path.resolve())]+_raw_output_paths(result, directory))
    return result


def evaluate_reference_claims(predictions, interventions, reference_populations: dict[str, Interval]):

    result = {"systems": {}, "reference_role": "R", "scope": "independent reference populations must use the R-family simultaneous set"}
    measured = interventions["intervals"]
    def actual(identifier, readout):
        return measured.get(identifier+"/"+readout+"/full", undefined("该 I 格未完成"))
    for panel in predictions["panels"]:
        name = panel.system.name
        baseline = measured.get(name+"/baseline_a", undefined("缺少共同 I 基准"))
        reference = reference_populations.get(name, undefined("缺少独立 R 布居"))
        system_result = {"primary_pair": None, "same_side_pair": None, "shape": {}}
        primary = panel.development["primary_pair"]
        if primary is not None:
            primary_gates, routes = [], []
            for offset in primary["offsets"]:
                cell = next(cell for cell in panel.cells if cell["anchor"] == primary["anchor"] and cell["window"] == primary["window"] and cell["offset"] == offset)
                identifier = cell["id"]
                prediction = predictions["intervals"][identifier+"/mass"]
                bound = predictions["numerical_error_bounds"].get(identifier+"/mass")
                prediction = prediction.widen(bound) if bound is not None else undefined("预测数值界缺失")
                change = actual(identifier, "mass")
                direction = positive_gate(change, DELTA_MASS) if prediction.lower > DELTA_MASS else (negative_gate(change, -DELTA_MASS) if prediction.upper < -DELTA_MASS else "unresolved")
                repair = repair_intervals(reference, baseline, change)
                magnitude = interventions["agreement"]["readouts"][identifier+"/mass"]["status"]
                primary_gates.extend([direction, magnitude])
                routes.append({"cell": identifier, "mass": change, "repair": repair, "direction_status": direction})
            if routes[0]["mass"].finite and routes[1]["mass"].finite:
                opposite = ((routes[0]["mass"].lower > DELTA_MASS and routes[1]["mass"].upper < -DELTA_MASS)
                            or (routes[1]["mass"].lower > DELTA_MASS and routes[0]["mass"].upper < -DELTA_MASS))
            else:
                opposite = False
            primary_gates.append("supported" if opposite else "unresolved")
            system_result["primary_pair"] = {"status": combine_gates(primary_gates), "routes": routes,
                                              "repair_statuses": [combine_gates([positive_gate(row["repair"]["deficit"]), positive_gate(row["repair"]["gain"])]) for row in routes]}
        selected = predictions["same_side_selected"].get(name)
        if selected is not None:
            gates, rows = [], []
            for index, identifier in enumerate(selected["cell_ids"]):
                mass, risk = actual(identifier, "mass"), actual(identifier, "log_risk")
                repair = repair_intervals(reference, baseline, mass)
                risk_gate = negative_gate(risk, -DELTA_LOG_RISK) if index == selected["negative_risk_route"] else positive_gate(risk, DELTA_LOG_RISK)
                gates.extend([positive_gate(mass, DELTA_MASS), positive_gate(repair["deficit"]), positive_gate(repair["gain"]), risk_gate,
                              interventions["agreement"]["readouts"][identifier+"/mass"]["status"],
                              interventions["agreement"]["readouts"][identifier+"/log_risk"]["status"]])
                rows.append({"cell": identifier, "mass": mass, "log_risk": risk, "repair": repair})
            system_result["same_side_pair"] = {"status": combine_gates(gates), "routes": rows}
        for shape, identifier in predictions["shape_selected"].get(name, {}).items():
            prefix = f"{name}/shape{shape}/"
            prediction = predictions["intervals"][identifier+"/conditional_h2"]
            bound = predictions["numerical_error_bounds"].get(identifier+"/conditional_h2")
            prediction = prediction.widen(bound) if bound is not None else undefined("形状预测数值界缺失")
            system_result["shape"][shape] = shape_claim_gate([measured[prefix+"mass_"+region] for region in ("a", "b", "remainder")],
                measured[prefix+"h2"], prediction, measured[prefix+"h2_full_minus_four_half"])
        result["systems"][name] = system_result
    result["eligible_for_formal"] = bool(predictions["eligible_for_formal"] and interventions["eligible_for_formal"])
    return result
