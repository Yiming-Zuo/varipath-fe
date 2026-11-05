from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor


class InterventionError(RuntimeError):
    pass


def _value(result):
    return result[0] if isinstance(result, tuple) else result


def _lip(protection) -> float:
    value = protection.lipschitz
    return float(value() if callable(value) else value)


def _guard(protection, z: Tensor) -> Tensor:
    return protection(z).reshape(())


def _inverse_residual(y: Tensor, increment: Callable[[Tensor], Tensor],
                      tolerance: float, max_iterations: int) -> Tensor:
    x = y
    for _ in range(max_iterations):
        next_x = y - increment(x)
        if bool((next_x - x).detach().abs().max() <= tolerance):
            residual = next_x + increment(next_x) - y
            if bool(residual.detach().abs().max() <= tolerance):
                return next_x
        x = next_x
    raise InterventionError("残差映射逆解未在规定迭代数内收敛")


class IdentityLoop:
    def single_forward(self, z: Tensor) -> Tensor:
        return z

    single_inverse = single_forward


class ProtectedTranslation:


    def __init__(self, protection, displacement: Tensor, *,
                 max_microsteps: int = 10000, tolerance: float = 1e-12,
                 max_iterations: int = 100):
        self.protection = protection
        self.displacement = torch.as_tensor(displacement).detach().clone()
        needed = max(1, math.ceil(float(self.displacement.norm()) * _lip(protection) / .25))
        if needed > max_microsteps:
            raise InterventionError("平移微步数超过事前上限")
        self.microsteps = needed
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def _increment(self, z: Tensor) -> Tensor:
        return _guard(self.protection, z) * self.displacement.to(z) / self.microsteps

    def single_forward(self, z: Tensor) -> Tensor:
        for _ in range(self.microsteps):
            z = z + self._increment(z)
        return z

    def single_inverse(self, z: Tensor) -> Tensor:
        for _ in range(self.microsteps):
            z = _inverse_residual(z, self._increment, self.tolerance, self.max_iterations)
        return z

    def forward(self, batch: Tensor) -> Tensor:
        return torch.stack([self.single_forward(z) for z in batch])

    def inverse(self, batch: Tensor) -> Tensor:
        return torch.stack([self.single_inverse(z) for z in batch])


def window_weights(times: Tensor, deltas: Tensor, window: int) -> Tensor:
    if window not in (0, 1, 2) or times.shape != deltas.shape:
        raise ValueError("时窗或步网格无效")
    active = (times >= window / 3) & ((times < (window + 1) / 3) if window < 2 else (times <= 1))
    raw = deltas * times * (1 - times) * active
    if not bool(raw.sum() > 0):
        raise InterventionError("该时窗没有正剂量节点")
    return raw / raw.sum()


class ConjugateRoute:


    def __init__(self, base, loops: Sequence, protection, center: Tensor,
                 sigma: float, amplitude: float, eta: Tensor, *, epsilon: float = 0.,
                 conjugated_kicks: bool = False, tolerance: float = 1e-12,
                 max_iterations: int = 100):
        self.base, self.loops, self.protection = base, tuple(loops), protection
        self.n_steps = int(base.n_steps)
        self.center = torch.as_tensor(center).detach().clone()
        self.sigma, self.amplitude = float(sigma), float(amplitude)
        self.eta = torch.as_tensor(eta).detach().clone()
        self.epsilon = float(epsilon)
        self.conjugated_kicks = bool(conjugated_kicks)
        self.tolerance, self.max_iterations = tolerance, max_iterations
        if len(self.loops) != self.n_steps + 1 or self.eta.numel() != self.n_steps:
            raise ValueError("回路或剂量数与基流步数不一致")
        if not isinstance(self.loops[0], IdentityLoop) or not isinstance(self.loops[-1], IdentityLoop):
            raise InterventionError("回路两个端点必须严格为恒等映射")
        if self.sigma <= 0 or not math.isfinite(self.sigma) or not 0 <= self.amplitude <= .5:
            raise ValueError("Gaussian 宽度或目录幅度无效")
        if not bool(torch.isfinite(self.eta).all()) or bool((self.eta < 0).any()):
            raise ValueError("剂量权重必须有限且非负")
        self.certify_epsilon(self.epsilon)

    @property
    def kick_lipschitz(self) -> float:
        return self.amplitude * (1 / self.sigma ** 2 + _lip(self.protection) / (self.sigma * math.sqrt(math.e)))

    def certify_epsilon(self, epsilon: float) -> None:
        if not math.isfinite(float(epsilon)) or abs(float(epsilon)) * float(self.eta.max()) * self.kick_lipschitz > .25:
            raise InterventionError("kick 未满足全局 0.25 收缩界")

    def _increment(self, z: Tensor, k: int, epsilon) -> Tensor:
        difference = z - self.center.to(z)
        psi = torch.exp(-difference.square().sum() / (2 * self.sigma ** 2))
        return -epsilon * self.eta[k].to(z) * self.amplitude * _guard(self.protection, z) * psi * difference / self.sigma ** 2

    def _base_step(self, z: Tensor, k: int) -> Tensor:
        if hasattr(self.base, "single_step"):
            return self.base.single_step(z, k)
        return _value(self.base.step(z.unsqueeze(0), k))[0]

    def _base_inverse(self, z: Tensor, k: int) -> Tensor:
        if hasattr(self.base, "single_inverse_step"):
            value = _value(self.base.single_inverse_step(z, k))
        else:
            value = _value(self.base.inverse_step(z.unsqueeze(0), k))[0]
        residual = (self._base_step(value, k) - z).detach().abs().max()
        if not bool(torch.isfinite(value).all()) or bool(residual > self.tolerance):
            raise InterventionError("基流逆步未达到指定残差")
        return value

    def at_epsilon(self, epsilon: float):
        self.certify_epsilon(epsilon)
        route = copy.copy(self)
        route.epsilon = float(epsilon)
        return route

    def single_local_step(self, z: Tensor, k: int, epsilon=None) -> Tensor:

        if not 0 <= k < self.n_steps:
            raise IndexError("干预局部步编号越界")
        epsilon = self.epsilon if epsilon is None else epsilon
        if self.conjugated_kicks:
            z = self.loops[k].single_inverse(z)
            z = z + self._increment(z, k, epsilon)
        else:
            z = z + self._increment(z, k, epsilon)
            z = self.loops[k].single_inverse(z)
        return self.loops[k + 1].single_forward(self._base_step(z, k))

    def single_value(self, z: Tensor, epsilon=None) -> Tensor:

        epsilon = self.epsilon if epsilon is None else epsilon
        for k in range(self.n_steps):
            z = self.single_local_step(z, k, epsilon)
        return z

    def reference_whole_logdet(self, z: Tensor, epsilon=None) -> Tensor:

        matrix = torch.func.jacrev(lambda x: self.single_value(x, epsilon))(z)
        sign, value = torch.linalg.slogdet(matrix)
        if not bool((sign > 0).detach()) or not bool(torch.isfinite(value).detach()):
            raise InterventionError("实际整图Jacobian非正或非有限")
        return value

    @torch.no_grad()
    def _single_forward_stream(self, initial: Tensor, epsilon=None, check_budget=None):
        epsilon = self.epsilon if epsilon is None else epsilon
        z = initial.detach()
        logdet = z.new_zeros(())
        for k in range(self.n_steps):
            if check_budget is not None:
                check_budget()
            def local(point):
                value = self.single_local_step(point, k, epsilon)
                return value, value
            matrix, value = torch.func.jacrev(local, has_aux=True)(z)
            sign, increment = torch.linalg.slogdet(matrix)
            if not bool(sign > 0) or not bool(torch.isfinite(increment)) or not bool(torch.isfinite(value).all()):
                raise InterventionError(f"实际局部干预图第{k}步Jacobian非正或非有限")
            z = value.detach()
            logdet = (logdet + increment).detach()
        return z, logdet

    @torch.no_grad()
    def single_logdet(self, z: Tensor, epsilon=None, check_budget=None) -> Tensor:
        return self._single_forward_stream(z, epsilon, check_budget)[1]

    def single_local_inverse(self, z: Tensor, k: int) -> Tensor:
        z = self._base_inverse(self.loops[k + 1].single_inverse(z), k)
        if self.conjugated_kicks:
            z = _inverse_residual(z, lambda x: self._increment(x, k, self.epsilon), self.tolerance, self.max_iterations)
            return self.loops[k].single_forward(z)
        z = self.loops[k].single_forward(z)
        return _inverse_residual(z, lambda x: self._increment(x, k, self.epsilon), self.tolerance, self.max_iterations)

    @torch.no_grad()
    def single_inverse(self, z: Tensor, check_budget=None) -> Tensor:
        z = z.detach()
        for k in reversed(range(self.n_steps)):
            if check_budget is not None:
                check_budget()
            z = self.single_local_inverse(z, k).detach()
            if not bool(torch.isfinite(z).all()):
                raise InterventionError(f"实际局部逆图第{k}步非有限")
        return z

    def _validate_batch(self, batch):
        if batch.ndim != 2 or not len(batch) or batch.shape[1] != self.center.numel() or batch.dtype != torch.float64 or not bool(torch.isfinite(batch).all()):
            raise InterventionError("干预初态须为完整有限的float64 [sample,dim]批次")
        self.certify_epsilon(self.epsilon)
        if hasattr(self.base, "validate_parameters"):
            self.base.validate_parameters()

    @torch.no_grad()
    def forward(self, batch: Tensor, check_budget=None) -> tuple[Tensor, Tensor]:
        self._validate_batch(batch)
        pairs = [self._single_forward_stream(z, check_budget=check_budget) for z in batch]
        return torch.stack([pair[0] for pair in pairs]), torch.stack([pair[1] for pair in pairs])

    @torch.no_grad()
    def inverse(self, batch: Tensor, check_budget=None) -> tuple[Tensor, Tensor]:
        self._validate_batch(batch)
        inputs = torch.stack([self.single_inverse(z, check_budget) for z in batch])
        pairs = [self._single_forward_stream(z, check_budget=check_budget) for z in inputs]
        restored = torch.stack([pair[0] for pair in pairs])
        residual = (restored - batch).abs().max()
        if not bool(torch.isfinite(inputs).all()) or bool(residual > 10 * self.tolerance):
            raise InterventionError("整个干预图逆解残差超出容差")
        return inputs, -torch.stack([pair[1] for pair in pairs])


def reference_response_samples_whole(route: ConjugateRoute, samples: Tensor,
                                     effective_u0: Callable[[Tensor], Tensor],
                                     effective_u1: Callable[[Tensor], Tensor]) -> dict[str, Tensor]:

    result = {name: [] for name in ("endpoint", "score", "dot_logweight", "logweight", "displacement")}
    zero = samples.new_zeros(())
    for z in samples:
        def endpoint(x, epsilon):
            return route.single_value(x, epsilon)
        def pullback(x):
            derivative = torch.func.jacrev(lambda e: endpoint(x, e))(zero)
            jacobian = torch.func.jacrev(lambda a: endpoint(a, zero))(x)
            return torch.linalg.solve(jacobian, derivative)
        y = endpoint(z, zero)
        displacement = torch.func.jacrev(lambda e: endpoint(z, e))(zero)
        a = pullback(z)
        score = (torch.func.grad(lambda x: effective_u0(x.unsqueeze(0)).sum())(z) * a).sum() - torch.trace(torch.func.jacrev(pullback)(z))
        def logweight(e):
            value = endpoint(z, e)
            jacobian = torch.func.jacrev(lambda x: endpoint(x, e))(z)
            return effective_u0(z.unsqueeze(0)).sum() - effective_u1(value.unsqueeze(0)).sum() + torch.linalg.slogdet(jacobian)[1]
        result["endpoint"].append(y)
        result["score"].append(score)
        result["dot_logweight"].append(torch.func.jacrev(logweight)(zero))
        result["logweight"].append(logweight(zero))
        result["displacement"].append(displacement)
    return {name: torch.stack(values) for name, values in result.items()}


@torch.no_grad()
def response_samples(route: ConjugateRoute, samples: Tensor,
                     effective_u0: Callable[[Tensor], Tensor],
                     effective_u1: Callable[[Tensor], Tensor], *, check_budget=None,
                     max_condition_number: float = 1e12,
                     solve_tolerance: float = 1e-10) -> dict[str, Tensor]:

    route._validate_batch(samples)
    if not math.isfinite(max_condition_number) or max_condition_number < 1 or not math.isfinite(solve_tolerance) or solve_tolerance <= 0:
        raise ValueError("响应线性解的条件数上限和残差容差非法")
    result = {name: [] for name in ("endpoint", "score", "dot_logweight", "logweight", "displacement")}
    zero = samples.new_zeros(())
    for initial in samples:
        z = initial.detach()
        dimension = z.numel()
        M = torch.eye(dimension, dtype=z.dtype, device=z.device)
        v, g = torch.zeros_like(z), torch.zeros_like(z)
        dell, logdet = z.new_zeros(()), z.new_zeros(())
        for k in range(route.n_steps):
            if check_budget is not None:
                check_budget()
            def local(point, dose):
                value = route.single_local_step(point, k, dose)
                return value, value
            (D, h), moved = torch.func.jacrev(local, argnums=(0, 1), has_aux=True)(z, zero)
            sign, local_logdet = torch.linalg.slogdet(D)
            if not bool(sign > 0) or not bool(torch.isfinite(local_logdet)):
                raise InterventionError(f"响应第{k}步实际Jacobian非正或非有限")
            def logdet_function(point, dose):
                derivative = torch.func.jacrev(lambda x: route.single_local_step(x, k, dose))(point)
                return torch.linalg.slogdet(derivative)[1]
            b, c = torch.func.grad(logdet_function, argnums=(0, 1))(z, zero)

            next_g = g + M.T @ b
            next_dell = dell + torch.dot(b, v) + c
            next_v = D @ v + h
            next_M = D @ M
            z, M, v, g = moved.detach(), next_M.detach(), next_v.detach(), next_g.detach()
            dell, logdet = next_dell.detach(), (logdet + local_logdet).detach()
            if not all(bool(torch.isfinite(value).all()) for value in (z, M, v, g, dell, logdet)):
                raise InterventionError(f"响应第{k}步前缀导数状态非有限；未截断或丢弃该样本")
        condition = float(torch.linalg.cond(M))
        if not math.isfinite(condition) or condition > max_condition_number:
            raise InterventionError(f"终点前缀Jacobian病态，cond={condition:g}超过{max_condition_number:g}；禁止伪逆补值")
        try:
            a = torch.linalg.solve(M, v)
        except torch.linalg.LinAlgError as exc:
            raise InterventionError("终点前缀Jacobian线性解失败；禁止伪逆补值") from exc
        residual = float((M @ a - v).norm() / (torch.linalg.matrix_norm(M, ord=2) * a.norm() + v.norm()).clamp_min(torch.finfo(z.dtype).tiny))
        if not bool(torch.isfinite(a).all()) or not math.isfinite(residual) or residual > solve_tolerance:
            raise InterventionError(f"终点响应线性解的相对后向残差{residual:g}超过{solve_tolerance:g}")
        grad0 = torch.func.grad(lambda x: effective_u0(x.unsqueeze(0)).sum())(initial)
        grad1 = torch.func.grad(lambda x: effective_u1(x.unsqueeze(0)).sum())(z)
        score = torch.dot(grad0 + g, a) - dell
        dot_logweight = -torch.dot(grad1, v) + dell
        logweight = effective_u0(initial.unsqueeze(0)).sum() - effective_u1(z.unsqueeze(0)).sum() + logdet
        if not all(bool(torch.isfinite(value)) for value in (score, dot_logweight, logweight)):
            raise InterventionError("冻结响应的物理能量、score或权重导数非有限")
        result["endpoint"].append(z.detach())
        result["score"].append(score.detach())
        result["dot_logweight"].append(dot_logweight.detach())
        result["logweight"].append(logweight.detach())
        result["displacement"].append(v.detach())
    return {name: torch.stack(values) for name, values in result.items()}


def response_predictions(response: dict[str, Tensor], region: Tensor, epsilon: float) -> dict[str, Tensor]:
    score, logweight = response["score"], response["logweight"]
    region = region.to(dtype=score.dtype)
    weights = torch.softmax(2 * logweight, dim=0)
    conditional = score[region.bool()]
    shape = epsilon ** 2 * conditional.var(unbiased=False) / 8 if len(conditional) else score.new_tensor(float("nan"))
    return {"mass": epsilon * (region * score).mean(),
            "log_risk": 2 * epsilon * (weights * response["dot_logweight"]).sum(),
            "conditional_h2": shape,
            "score_mean": score.mean(),
            "weight_derivative_mean": (torch.softmax(logweight, dim=0) * response["dot_logweight"]).sum(),
            "risk_identity_residual": (weights * (2 * response["dot_logweight"] + score)).sum()}


def _frozen_map(mapping, points, inverse=False, check_budget=None):
    if check_budget is not None:
        check_budget()
    function = mapping.inverse if inverse else mapping.forward
    return function(points, check_budget=check_budget) if isinstance(mapping, ConjugateRoute) else function(points)


def proposal_log_density(route, endpoints: Tensor, effective_u0, check_budget=None) -> Tensor:
    source, inverse_logdet = _frozen_map(route, endpoints, True, check_budget)
    return -effective_u0(source) + inverse_logdet


def conditional_hellinger(base, perturbed, source: Tensor, effective_u0,
                          classify: Callable[[Tensor], Tensor], n_regions: int, *, check_budget=None) -> dict[str, Tensor]:

    y0, _ = _frozen_map(base, source, check_budget=check_budget)
    ye, _ = _frozen_map(perturbed, source, check_budget=check_budget)
    y = torch.cat((y0, ye))
    logq0 = proposal_log_density(base, y, effective_u0, check_budget)
    logqe = proposal_log_density(perturbed, y, effective_u0, check_budget)
    affinity_integrand = torch.exp(math.log(2.) + .5 * (logq0 + logqe) - torch.logaddexp(logq0, logqe))
    labels0, labelse, labels = classify(y0), classify(ye), classify(y)
    if bool(((labels < 0) | (labels >= n_regions)).any()):
        raise InterventionError("区域标签不确定或超出穷尽分区")
    rho0 = torch.stack([(labels0 == k).to(source.dtype).mean() for k in range(n_regions)])
    rhoe = torch.stack([(labelse == k).to(source.dtype).mean() for k in range(n_regions)])
    restricted = torch.stack([(affinity_integrand * (labels == k)).mean() for k in range(n_regions)])
    denominator = torch.sqrt(rho0 * rhoe)
    conditional_h2 = torch.where(denominator > 0, 1 - restricted / denominator, torch.full_like(denominator, float("nan")))
    total_h2 = 1 - affinity_integrand.mean()
    partition_h2 = 1 - denominator.sum()
    return {"conditional_h2": conditional_h2, "total_h2": total_h2,
            "partition_h2": partition_h2, "shape_difference": total_h2 - partition_h2,
            "rho0": rho0, "rhoe": rhoe, "affinity_integrand": affinity_integrand,
            "labels": labels}


def finite_intervention(route: ConjugateRoute, samples: Tensor, effective_u0,
                        effective_u1, classify, n_regions: int, *, include_shape: bool = True, check_budget=None) -> dict:

    outputs = {}
    for name, dose in (("baseline", 0.), ("full", route.epsilon), ("half", route.epsilon / 2)):
        branch = route.at_epsilon(dose)
        endpoint, logdet = branch.forward(samples, check_budget=check_budget)
        logweight = effective_u0(samples) - effective_u1(endpoint) + logdet
        labels = classify(endpoint)
        if not bool(torch.isfinite(logweight).all()) or bool(((labels < 0) | (labels >= n_regions)).any()):
            raise InterventionError("有限干预权重或区域标签无效，整块失败")
        masses = torch.stack([(labels == k).to(samples.dtype).mean() for k in range(n_regions)])
        outputs[name] = {"endpoint": endpoint, "logweight": logweight, "labels": labels,
                         "mass": masses, "log_second_moment": torch.logsumexp(2 * logweight, 0) - math.log(len(samples))}
    baseline, full, half = (outputs[key] for key in ("baseline", "full", "half"))
    outputs["mass_change"] = full["mass"] - baseline["mass"]
    outputs["half_mass_change"] = half["mass"] - baseline["mass"]
    outputs["mass_half_dose_residual"] = outputs["mass_change"] - 2 * outputs["half_mass_change"]
    outputs["log_risk_change"] = full["log_second_moment"] - baseline["log_second_moment"]
    outputs["half_log_risk_change"] = half["log_second_moment"] - baseline["log_second_moment"]
    outputs["risk_half_dose_residual"] = outputs["log_risk_change"] - 2 * outputs["half_log_risk_change"]
    if include_shape:
        off = route.at_epsilon(0.)
        outputs["shape_full"] = conditional_hellinger(off, route, samples, effective_u0, classify, n_regions, check_budget=check_budget)
        outputs["shape_half"] = conditional_hellinger(off, route.at_epsilon(route.epsilon / 2), samples, effective_u0, classify, n_regions, check_budget=check_budget)
        outputs["shape_half_dose_residual"] = outputs["shape_full"]["conditional_h2"] - 4 * outputs["shape_half"]["conditional_h2"]
    return outputs


@dataclass(frozen=True)
class GeometryQualification:
    branch_unique: bool
    maximum_normal_angle: float
    maximum_vertical_departure: float
    sigma: float
    entire_tube_on_plateau: bool
    certified: bool = False
    reason: str = "未提供全分支与全活动管认证"

    @property
    def qualified(self) -> bool:
        return (self.certified and self.branch_unique and self.sigma > 0
                and self.maximum_normal_angle < .1
                and self.maximum_vertical_departure < .1 * self.sigma
                and self.entire_tube_on_plateau)


def geometric_mass_prediction(nodes: Tensor, boundary: Tensor, normals: Tensor,
                              center: Tensor, sigma: float, amplitude: float,
                              eta: Tensor, epsilon: float, protection,
                              qualification: GeometryQualification,
                              bandwidth_factor: float = .1) -> Tensor:
    if not qualification.qualified:
        raise InterventionError("几何组未通过完整资格认证：" + qualification.reason)
    if nodes.shape[0] != eta.numel():
        raise ValueError("节点轨迹与剂量不一致")
    width = bandwidth_factor * sigma
    answer = nodes.new_zeros(())
    for k in range(nodes.shape[0]):
        displacement = nodes[k] - boundary[k]
        normal_coordinate = displacement @ normals[k]
        tangent = displacement - normal_coordinate[:, None] * normals[k]
        b = protection(nodes[k])
        exposure = (b * (normal_coordinate.abs() < width) * torch.exp(-tangent.square().sum(-1) / (2 * sigma ** 2)) / (2 * width)).mean()
        offset = ((boundary[k] - center) * normals[k]).sum()
        answer = answer + epsilon * eta[k] * amplitude * offset * torch.exp(-offset ** 2 / (2 * sigma ** 2)) * exposure / sigma ** 2
    return answer
