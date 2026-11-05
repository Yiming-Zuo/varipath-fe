from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch

from ._feat_native import Coef, b_loss, eta_loss, score_PotentialNet, score_VectorFieldNet, tsm_loss
from .bridge import _classify, bridge_readout, nebar


@dataclass(frozen=True)
class FEATConfig:
    iterations: int = 800
    batch_size: int = 512
    hidden: int = 256
    layers: int = 4
    learning_rate: float = 0.001
    tsm_weight: float = 0.1
    noise: float = 0.01
    path_steps: int = 100
    seed: int = 1328100
    checkpoint_every: int = 10


def _atomic_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(data, pending)
    pending.replace(path)


def _pool(points, dimension):
    if points.ndim != 2 or points.shape[1] != dimension or not len(points) or points.dtype != torch.float64 or not torch.isfinite(points).all():
        raise ValueError("FEAT需要完整、有限的float64端点池")


class FEAT(torch.nn.Module):
    def __init__(self, system, config=None):
        super().__init__()
        self.system, self.config = system, config or FEATConfig()
        c = self.config
        if min(c.iterations, c.batch_size, c.hidden, c.layers, c.path_steps, c.checkpoint_every) <= 0 or min(c.learning_rate, c.noise) <= 0 or c.tsm_weight < 0:
            raise ValueError("FEAT配置中的次数、宽度、噪声和学习率必须合法")
        self.alpha, self.beta, self.gamma = Coef("1-t"), Coef("t"), Coef("sqrt(a*t*(1-t))+b", a=0.05, b=0.001)
        with torch.random.fork_rng():
            torch.manual_seed(c.seed)
            self.vector = score_VectorFieldNet(system.dim, "cpu", num_layers=c.layers, num_hid=c.hidden).double()
            self.score = score_PotentialNet(system.dim, "cpu", self.alpha, self.beta, self.gamma, num_layers=c.layers, num_hid=c.hidden).double()

    def endpoint_log_prob(self, points, endpoint):
        if hasattr(self.system, "positions") and not torch.isfinite(self.system.positions(points)).all():
            raise FloatingPointError("FEAT端点坐标重建非有限，不能当成零支撑")
        values = -torch.vmap(lambda z: self.system.effective_u(z, endpoint))(points)
        legal = self.system.in_domain(points) if hasattr(self.system, "in_domain") else torch.ones(len(points), dtype=torch.bool, device=points.device)
        if not torch.isfinite(values[legal]).all() or torch.isnan(values).any() or torch.isposinf(values).any():
            raise FloatingPointError("FEAT合法端点物理密度非有限")
        return values

    def endpoint_score(self, points, endpoint):
        return torch.vmap(torch.func.grad(lambda z: -self.system.effective_u(z, endpoint)))(points)

    def fit(self, source, target, checkpoint=None, resume=False, stop_after=None, check_budget=None):

        _pool(source, self.system.dim)
        _pool(target, self.system.dim)
        if source.device.type != "cpu" or target.device.type != "cpu":
            raise ValueError("当前FEAT持久化入口要求CPU端点池")
        c, first, history = self.config, 0, []
        optimizer = torch.optim.Adam(self.parameters(), lr=c.learning_rate)
        initial = None
        if resume:
            if checkpoint is None or not Path(checkpoint).exists():
                raise FileNotFoundError("FEAT恢复需要现有检查点")
            initial = torch.load(checkpoint, weights_only=False)
            if initial["config"] != asdict(c) or not torch.equal(initial["source"], source) or not torch.equal(initial["target"], target):
                raise ValueError("FEAT恢复配置或完整训练池已改变")
            self.load_state_dict(initial["model"])
            optimizer.load_state_dict(initial["optimizer"])
            first, history = initial["iteration"], initial["history"]
        elif checkpoint is not None and Path(checkpoint).exists():
            raise FileExistsError("FEAT检查点已存在，请显式恢复")
        if stop_after is not None and stop_after < first:
            raise ValueError("停止迭代早于现有进度")
        last = c.iterations if stop_after is None else min(stop_after, c.iterations)
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with torch.random.fork_rng():
                torch.manual_seed(c.seed + 1)
                if initial is not None:
                    torch.set_rng_state(initial["rng"])
                self.train()
                for iteration in range(first, last):
                    if check_budget is not None:
                        check_budget()

                    x = source[torch.randint(len(source), (c.batch_size,))]
                    y = target[torch.randint(len(target), (c.batch_size,))]
                    loss = b_loss(x, y, self.alpha, self.beta, self.gamma, self.vector) + eta_loss(x, y, self.alpha, self.beta, self.gamma, self.score)
                    if c.tsm_weight:
                        loss = loss + c.tsm_weight * tsm_loss(x, y, self.endpoint_score(x, 0), self.endpoint_score(y, 1), self.alpha, self.beta, self.gamma, self.score)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("FEAT原生训练损失非有限")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
                    history.append({"iteration": iteration + 1, "loss": float(loss.detach())})
                    if checkpoint is not None and ((iteration + 1) % c.checkpoint_every == 0 or iteration + 1 == last):
                        if check_budget is not None:
                            check_budget()
                        _atomic_save(checkpoint, {"config": asdict(c), "model": self.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(), "iteration": iteration + 1, "history": history, "source": source, "target": target})
        finally:
            torch.set_default_dtype(old_dtype)
        self.eval()
        return {"iterations": last, "complete": last == c.iterations, "coupling": "independent", "history": history}

    def _drift(self, x, t, forward):
        return (1 if forward else -1) * self.vector(x, t) + self.config.noise * self.score.score(x, t)

    @torch.no_grad()
    def sample_paths(self, initial, forward=True, generator=None, times=None, check_budget=None):

        _pool(initial, self.system.dim)
        times = torch.linspace(0, 1, self.config.path_steps + 1, dtype=initial.dtype, device=initial.device) if times is None else torch.as_tensor(times, dtype=initial.dtype, device=initial.device)
        if times.ndim != 1 or len(times) < 2 or times[0] != 0 or times[-1] != 1 or not (torch.diff(times) > 0).all():
            raise ValueError("FEAT需要从0到1的严格递增有限时间网格")
        integration_times = times if forward else times.flip(0)
        x = initial
        path = [x]
        for t0, t1 in zip(integration_times[:-1], integration_times[1:]):
            if check_budget is not None:
                check_budget()
            dt = abs(t1 - t0)
            t = t0.expand(len(x))
            noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
            x = x + dt * self._drift(x, t, forward) + torch.sqrt(2 * dt * self.config.noise) * noise
            if not torch.isfinite(x).all():
                raise FloatingPointError("FEAT实际随机路径非有限，整批失败")
            path.append(x)
        result = torch.stack(path, 1)
        ratio = self.replay_ratio(result, forward=forward, times=times, check_budget=check_budget)
        return result, ratio

    @torch.no_grad()
    def replay_ratio(self, path, forward=True, times=None, check_budget=None):
        if path.ndim != 3 or path.shape[2] != self.system.dim or not torch.isfinite(path).all():
            raise ValueError("FEAT重放需要完整有限随机路径")
        times = torch.linspace(0, 1, path.shape[1], dtype=path.dtype, device=path.device) if times is None else torch.as_tensor(times, dtype=path.dtype, device=path.device)
        if len(times) != path.shape[1] or times[0] != 0 or times[-1] != 1 or not (torch.diff(times) > 0).all():
            raise ValueError("路径与时间网格不符")
        oriented = path if forward else path.flip(1)
        logp0 = self.endpoint_log_prob(oriented[:, 0], 0)
        logp1 = self.endpoint_log_prob(oriented[:, -1], 1)

        initial_logp = logp0 if forward else logp1
        if not torch.isfinite(initial_logp).all():
            raise ValueError("FEAT初端不是有限合法物理端点样本")
        result = logp1 - logp0
        for k, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
            if check_budget is not None:
                check_budget()
            dt = t1 - t0
            x, y = oriented[:, k], oriented[:, k + 1]
            mean_f = x + dt * self._drift(x, t0.expand(len(x)), True)
            mean_r = y + dt * self._drift(y, t1.expand(len(y)), False)
            variance = 2 * dt * self.config.noise
            log_f = -0.5 * (((y - mean_f).square() / variance) + torch.log(2 * math.pi * variance)).sum(1)
            log_r = -0.5 * (((x - mean_r).square() / variance) + torch.log(2 * math.pi * variance)).sum(1)
            result = result + log_r - log_f
        ratio = result if forward else -result
        if torch.isnan(ratio).any() or torch.isposinf(ratio).any():
            raise FloatingPointError("FEAT密度比无效；正常的零目标支撑应为负无穷")
        return ratio

    @torch.no_grad()
    def evaluate(self, source, target, generator=None, eta=1e-4, min_overlap=1e-8,
                 check_budget=None, batch_size=32, store_paths=False, path_sink=None):
        if type(batch_size) is not int or batch_size<1:
            raise ValueError("生产计算批量必须为正整数")
        _pool(source,self.system.dim); _pool(target,self.system.dim)
        paths=[];ratios=[]
        for initial,forward in ((source,True),(target,False)):
            stored=[];weights=[]
            for start in range(0,len(initial),batch_size):
                path,ratio=self.sample_paths(initial[start:start+batch_size],forward,generator,check_budget=check_budget)
                if path_sink is not None:
                    path_sink(forward,start,path,ratio)
                stored.append(path if store_paths else path[:,[0,-1]])
                weights.append(ratio)
            paths.append(torch.cat(stored));ratios.append(torch.cat(weights))
        fp,rp=paths;a,b=ratios
        if check_budget is not None:
            check_budget()

        solved = nebar(a.cpu().numpy(), b.cpu().numpy(), min_overlap)
        readout = bridge_readout(fp.cpu().numpy(), rp.cpu().numpy(), a.cpu().numpy(), b.cpu().numpy(), solved["log_partition_ratio"], lambda points: _classify(self.system, points), eta)
        if getattr(self.system, "name", None) == "ALDP":
            readout["task_log_odds_difference"] = -readout["reported_dimensionless_preference"]
        return {"native_nebar": solved, "readout": readout, "forward_path": fp, "reverse_path": rp,
                "forward_log_ratio": a, "reverse_log_ratio": b, "production_batch_size":batch_size,
                "path_storage":"all_nodes" if store_paths else "endpoints_only_after_actual_kernel_replay"}
