from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TrainingConfig:
    max_steps: int = 20
    history_size: int = 10
    gradient_tolerance: float = 1e-6
    step_tolerance: float = 1e-12
    armijo: float = 1e-4
    backtracking: float = 0.5
    max_line_search: int = 24
    patience: int = 8
    scale_every: int = 10
    checkpoint_every: int = 1
    batch_size: int = 32
    checkpoint_steps: int = 64
    deadline_monotonic: float | None = None

    def __post_init__(self):
        if min(self.max_steps, self.history_size, self.max_line_search, self.patience, self.scale_every, self.checkpoint_every,self.batch_size,self.checkpoint_steps) < 1:
            raise ValueError("训练迭代、历史、线搜索和检查点间隔须为正")
        if type(self.batch_size) is not int or type(self.checkpoint_steps) is not int:
            raise ValueError("批量和节点 checkpoint 间隔必须为整数")
        if not (0 < self.armijo < 1 and 0 < self.backtracking < 1):
            raise ValueError("Armijo 与回溯系数须位于 (0,1)")
        if min(self.gradient_tolerance, self.step_tolerance) <= 0:
            raise ValueError("训练容差须为正")


TrainConfig = TrainingConfig


def _flat(parameters):
    return torch.cat([p.reshape(-1) for p in parameters])


def _assign(parameters, vector):
    with torch.no_grad():
        for p, value in zip(parameters, vector.split([p.numel() for p in parameters])):
            p.copy_(value.reshape_as(p))


def _state_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _rng_state():
    rng = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    a = state["numpy"]
    np.random.set_state((a[0], np.asarray(a[1], dtype=np.uint32), a[2], a[3], a[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _two_loop(gradient, pairs):
    q = gradient.clone()
    alphas = []
    for s, y, rho in reversed(pairs):
        alpha = rho * torch.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)
    if pairs:
        s, y, _ = pairs[-1]
        q = q * (torch.dot(s, y) / torch.dot(y, y))
    for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
        q = q + s * (alpha - rho * torch.dot(y, q))
    return -q


def sampled_gradient_scale(points, potentials, chain_ids=None, *,batch_size=32,check_budget=None) -> float:

    if len(points) < 2:
        raise ValueError("梯度差商至少需要两个不同构型")
    limit = min(len(points), max(100, 10 * points.shape[1]))
    if chain_ids is None:
        chain_ids = torch.zeros(len(points), dtype=torch.int64)
    labels = torch.as_tensor(chain_ids, device=points.device)
    if labels.shape != (len(points),):
        raise ValueError("链标签数量与训练池不符")
    groups = [torch.nonzero(labels == label).flatten() for label in torch.unique(labels, sorted=True)]
    selected = []
    cursor = 0
    while len(selected) < limit:
        progressed = False
        for group in groups:
            if cursor < len(group) and len(selected) < limit:
                selected.append(group[cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    points = points[torch.stack(selected)].detach()
    distances = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
    distances = distances.masked_fill(distances <= 1e-12, float("inf"))
    lengths, neighbors = distances.min(1)
    if not torch.isfinite(lengths).all():
        raise ValueError("训练构型没有可用的非重复近邻，无法估计梯度差商")
    if callable(potentials):
        potentials = [potentials]
    values = []
    for potential in potentials:
        gradients=[]
        for start in range(0,len(points),batch_size):
            if check_budget is not None:
                check_budget()
            gradients.append(torch.vmap(torch.func.grad(potential))(points[start:start+batch_size]).detach())
        gradient=torch.cat(gradients)
        value = ((gradient - gradient[neighbors]).norm(dim=1) / lengths).max()
        if not torch.isfinite(value):
            raise FloatingPointError("势能梯度差商非有限")
        values.append(float(value.detach()))
    return max(values)


def streamed_objective(model,points,u0,u1,*,batch_size=32,checkpoint_steps=64,
                       gradient=False,check_budget=None):

    if type(batch_size) is not int or batch_size<1 or type(checkpoint_steps) is not int or checkpoint_steps<1:
        raise ValueError("完整池流式计算的批量和 checkpoint 间隔须为正整数")
    if not len(points):
        raise ValueError("完整池不能为空")
    total=points.new_tensor(-float("inf"))
    with torch.no_grad():
        for start in range(0,len(points),batch_size):
            if check_budget is not None:
                check_budget()
            loga,_=model.log_weights(points[start:start+batch_size],u0,u1,
                                    checkpoint_steps=checkpoint_steps,check_budget=check_budget)
            total=torch.logaddexp(total,torch.logsumexp(2*loga,0))
    value=total-math.log(len(points))
    if not torch.isfinite(value):
        raise FloatingPointError("完整二阶矩训练目标非有限")
    if not gradient:
        return float(value)
    parameters=list(model.parameters())
    accumulated=[torch.zeros_like(parameter) for parameter in parameters]
    for start in range(0,len(points),batch_size):
        if check_budget is not None:
            check_budget()
        loga,_=model.log_weights(points[start:start+batch_size],u0,u1,
                                checkpoint_steps=checkpoint_steps,check_budget=check_budget)
        weights=2*torch.exp(2*loga.detach()-total)
        derivatives=torch.autograd.grad(loga,parameters,grad_outputs=weights,allow_unused=True)
        for output,derivative in zip(accumulated,derivatives):
            if derivative is not None:
                output.add_(derivative.detach())
        del loga,weights,derivatives
    result=_flat(accumulated)
    if not torch.isfinite(result).all():
        raise FloatingPointError("完整二阶矩目标梯度非有限")
    return float(value),result


def train_transport(
    model,
    train,
    validation,
    u0,
    u1,
    config: TrainingConfig | dict | None = None,
    checkpoint: str | Path | None = None,
    resume: bool = False,
    train_chain_ids=None,
    validation_chain_ids=None,
    scale_potential=None,
    should_stop=None,
):

    if config is None:
        config = TrainingConfig()
    elif isinstance(config, dict):
        config = TrainingConfig(**config)
    if hasattr(model,"validate_center_provider"):
        model.validate_center_provider()
    model._check_batch(train)
    model._check_batch(validation)
    if len(train) < 2 or len(validation) < 2:
        raise ValueError("完整训练池和独立验证池各至少含两个构型")
    if train_chain_ids is not None and validation_chain_ids is not None:
        if set(torch.as_tensor(train_chain_ids).cpu().tolist()) & set(torch.as_tensor(validation_chain_ids).cpu().tolist()):
            raise ValueError("训练与验证链角色相交")
    if resume and checkpoint is None:
        raise ValueError("恢复训练必须指定检查点")
    path = Path(checkpoint) if checkpoint is not None else None
    if path is not None and path.exists() and not resume:
        raise FileExistsError("训练检查点已存在；请显式恢复或使用新目录")
    params = list(model.parameters())
    device = train.device
    low, high = model.parameter_bounds()
    settings = asdict(config)
    settings.pop("max_steps")
    settings.pop("deadline_monotonic")
    scale_potentials = scale_potential if scale_potential is not None else (u0, u1)

    def deadline():
        if config.deadline_monotonic is not None and time.monotonic() >= config.deadline_monotonic:
            raise TimeoutError("训练超过冷启动预算；主程序整批失败")
        if should_stop is not None and should_stop():
            raise TimeoutError("训练停止条件触发；主程序整批失败")

    def evaluate(points, gradient=False):
        deadline()
        return streamed_objective(model,points,u0,u1,batch_size=config.batch_size,
                                  checkpoint_steps=config.checkpoint_steps,
                                  gradient=gradient,check_budget=deadline)

    def gradient_scale():
        transformed=[]
        with torch.no_grad():
            for start in range(0,len(train),config.batch_size):
                deadline()
                transformed.append(model(train[start:start+config.batch_size],
                                         checkpoint_steps=config.checkpoint_steps,check_budget=deadline)[0])
        return sampled_gradient_scale(torch.cat(transformed),scale_potentials,train_chain_ids,
                                      batch_size=config.batch_size,check_budget=deadline)

    history, pairs = [], []
    accepted = stalled = 0
    status = "max_steps"
    if resume:
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved["settings"] != settings:
            raise ValueError("训练算法配置与检查点不符")
        provider=model.center_provider_descriptor() if hasattr(model,"center_provider_descriptor") else None
        producer=model.center_provider_source_state() if hasattr(model,"center_provider_source_state") else None
        if saved.get("center_provider")!=provider:
            raise ValueError("恢复训练的冻结中心 descriptor 发生变化")
        old_producer=saved.get("center_provider_source_state")
        if (producer is None)!=(old_producer is None) or (producer is not None and (producer.keys()!=old_producer.keys() or any(not torch.equal(producer[key].cpu(),old_producer[key].cpu()) for key in producer))):
            raise ValueError("恢复训练的移动中心源预训练参数发生变化")
        for key, value in (("training_pool", train), ("validation_pool", validation)):
            if not torch.equal(saved[key].to(value), value):
                raise ValueError("恢复时固定训练/验证池发生变化")
        for key, value in (("train_chain_ids", train_chain_ids), ("validation_chain_ids", validation_chain_ids)):
            actual = None if value is None else torch.as_tensor(value).cpu()
            old = saved[key]
            if (old is None) != (actual is None) or (old is not None and not torch.equal(old.cpu(), actual)):
                raise ValueError("恢复时链角色发生变化")
        model.load_state_dict(saved["current_model"])
        pairs = [(s.to(device), y.to(device), rho.to(device)) for s, y, rho in saved["pairs"]]
        accepted, stalled = saved["accepted"], saved["stalled"]
        history = saved["history"]
        best_model, best_validation = saved["best_model"], saved["best_validation"]
        initial_objective, initial_scale, scale = saved["initial_objective"], saved["initial_scale"], saved["scale"]
        _restore_rng(saved["rng"])
        value, gradient = evaluate(train, True)
        if not math.isclose(value, saved["current_objective"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("恢复时端点有效势或模型目标发生变化")
    else:
        value, gradient = evaluate(train, True)
        initial_objective = value
        best_validation = evaluate(validation)
        best_model = _state_cpu(model)
        initial_scale = gradient_scale()
        scale = initial_scale

    def save():
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "settings": settings,
            "center_provider":model.center_provider_descriptor() if hasattr(model,"center_provider_descriptor") else None,
            "center_provider_source_state":model.center_provider_source_state() if hasattr(model,"center_provider_source_state") else None,
            "current_model": _state_cpu(model),
            "best_model": best_model,
            "best_validation": best_validation,
            "pairs": [(s.cpu(), y.cpu(), rho.cpu()) for s, y, rho in pairs],
            "accepted": accepted,
            "stalled": stalled,
            "history": history,
            "initial_objective": initial_objective,
            "current_objective": value,
            "initial_scale": initial_scale,
            "scale": scale,
            "training_pool": train.detach().cpu(),
            "validation_pool": validation.detach().cpu(),
            "train_chain_ids": None if train_chain_ids is None else torch.as_tensor(train_chain_ids).cpu(),
            "validation_chain_ids": None if validation_chain_ids is None else torch.as_tensor(validation_chain_ids).cpu(),
            "rng": _rng_state(),
            "status": status,
        }
        temporary = path.with_name(path.name + ".pending")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    try:
        while accepted < config.max_steps:
            stable = {
                "model": _state_cpu(model),
                "pairs": pairs.copy(),
                "accepted": accepted,
                "stalled": stalled,
                "history": history.copy(),
                "best_model": best_model,
                "best_validation": best_validation,
                "scale": scale,
                "value": value,
                "gradient": gradient,
            }
            deadline()
            vector = _flat(params).detach().clone()
            projected = gradient.clone()
            blocked = ((vector <= low + 1e-13) & (gradient > 0)) | ((vector >= high - 1e-13) & (gradient < 0))
            projected[blocked] = 0
            if float(projected.abs().max()) <= config.gradient_tolerance:
                status = "converged"
                break
            direction = _two_loop(projected, pairs)
            direction[((vector <= low + 1e-13) & (direction < 0)) | ((vector >= high - 1e-13) & (direction > 0))] = 0
            slope = torch.dot(gradient, direction)
            if not torch.isfinite(slope) or slope >= 0:
                direction = -projected
                slope = -torch.dot(projected, projected)
            feasible = torch.where(direction > 0, (high - vector) / direction.clamp(min=1e-300), torch.where(direction < 0, (low - vector) / direction.clamp(max=-1e-300), torch.full_like(direction, float("inf"))))
            trial_step = min(1.0, (1 + initial_scale) / (1 + scale), float(feasible.min()))
            if trial_step <= config.step_tolerance:
                status = "step_tolerance"
                break
            found = False
            for attempt in range(config.max_line_search):
                trial = (vector + trial_step * direction).clamp(low, high)
                _assign(params, trial)
                candidate = evaluate(train)
                if candidate <= value + config.armijo * float(torch.dot(gradient, trial - vector)):
                    found = True
                    break
                trial_step *= config.backtracking
                if trial_step <= config.step_tolerance:
                    break
            if not found:
                _assign(params, vector)
                status = "line_search_failed"
                break
            candidate, next_gradient = evaluate(train, True)
            s, y = trial - vector, next_gradient - gradient
            sy = torch.dot(s, y)
            if sy > 1e-12 * s.norm() * y.norm() and sy > 0:
                pairs.append((s.detach(), y.detach(), (1 / sy).detach()))
                pairs = pairs[-config.history_size:]
            value, gradient = candidate, next_gradient
            accepted += 1
            validation_value = evaluate(validation)
            if validation_value < best_validation:
                best_validation, best_model, stalled = validation_value, _state_cpu(model), 0
            else:
                stalled += 1
            if accepted % config.scale_every == 0:
                scale = gradient_scale()
            history.append({"accepted_step": accepted, "objective": value, "validation_objective": validation_value, "step": trial_step, "line_search_evaluations": attempt + 1, "sampled_gradient_scale": scale})
            if accepted % config.checkpoint_every == 0:
                save()
            if stalled >= config.patience:
                status = "validation_patience"
                break
    except (TimeoutError, FloatingPointError, RuntimeError, KeyboardInterrupt):

        if "stable" in locals():
            model.load_state_dict(stable["model"])
            pairs, accepted, stalled, history = stable["pairs"], stable["accepted"], stable["stalled"], stable["history"]
            best_model, best_validation, scale = stable["best_model"], stable["best_validation"], stable["scale"]
            value, gradient = stable["value"], stable["gradient"]
        status = "interrupted"
        save()
        raise
    save()
    model.load_state_dict(best_model)
    return {
        "status": status,
        "accepted_steps": accepted,
        "initial_objective": initial_objective,
        "last_training_objective": value,
        "best_validation_objective": best_validation,
        "initial_sampled_gradient_scale": initial_scale,
        "scale_uses_physical_potential": scale_potential is not None,
        "objective_evaluation":dict(full_pool=True,batch_size=config.batch_size,
                                    checkpoint_steps=config.checkpoint_steps,
                                    gradient="two-pass global logsumexp with exact weighted accumulation"),
        "history": copy.deepcopy(history),
    }
