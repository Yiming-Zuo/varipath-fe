from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .centers import DiskCenters


class _ProtectionDerivative(torch.autograd.Function):

    @staticmethod
    def forward(ctx,z,protection):
        ctx.protection=protection
        ctx.save_for_backward(z)

        with torch.enable_grad(),torch.autograd.graph.saved_tensors_hooks(lambda t:t.detach(),lambda t:t):
            point=z.detach().requires_grad_(True)
            value=protection(point)
            derivative=torch.autograd.grad(value.sum(),point)[0] if value.requires_grad else torch.zeros_like(point)
        return value.detach(),derivative.detach()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,value_gradient,derivative_gradient):
        z,=ctx.saved_tensors
        with torch.enable_grad(),torch.autograd.graph.saved_tensors_hooks(lambda t:t.detach(),lambda t:t):
            point=z.detach().requires_grad_(True)
            value=ctx.protection(point)
            if value.requires_grad:
                derivative=torch.autograd.grad(value.sum(),point,create_graph=True)[0]
                scalar=(value*value_gradient).sum()
                if derivative.requires_grad:
                    scalar=scalar+(derivative*derivative_gradient).sum()
                result=torch.autograd.grad(scalar,point)[0]
            else:
                result=torch.zeros_like(point)
        return result,None


def _spectral_matrix(raw: torch.Tensor, upper: float = 2.0) -> torch.Tensor:
    return raw / (torch.linalg.matrix_norm(raw, ord=2) / upper).clamp(min=1.0)


class Transport(nn.Module):


    def __init__(
        self,
        dim: int,
        protection: nn.Module,
        spline_count: int = 50,
        steps: int | None = None,
        centers=None,
        bandwidths=None,
        amplitudes=None,
        time_centers=None,
        seed: int = 1024,
        inverse_tolerance: float = 1e-11,
        inverse_iterations: int = 64,
    ):
        super().__init__()
        if dim < 1 or spline_count < 4:
            raise ValueError("维数须为正；三次样条至少需要四个基函数")
        if not getattr(protection, "compact_support", False):
            raise ValueError("保护函数必须提供紧支撑证书")
        if not math.isfinite(float(protection.lipschitz)) or protection.lipschitz < 0:
            raise ValueError("保护函数缺少有限全局 Lipschitz 上界")
        if any(parameter.requires_grad for parameter in protection.parameters()):
            raise ValueError("共同保护函数必须在训练前冻结，不能包含可训练参数")
        if inverse_tolerance <= 0 or inverse_iterations < 1:
            raise ValueError("逆解容差及迭代数须为正")
        self.dim, self.spline_count = int(dim), int(spline_count)
        self.protection = protection
        self.inverse_tolerance = float(inverse_tolerance)
        self.inverse_iterations = int(inverse_iterations)
        generator = torch.Generator().manual_seed(seed)

        def initial_matrix(out, inp):
            m = torch.randn(out, inp, generator=generator, dtype=torch.float64)
            return m / torch.linalg.matrix_norm(m, ord=2)

        self.raw_U = nn.Parameter(initial_matrix(64, dim))
        self.raw_V = nn.Parameter(initial_matrix(64, 64))
        self.raw_W = nn.Parameter(initial_matrix(dim, 65))
        self.C = nn.Parameter(torch.zeros(spline_count, 65, dtype=torch.float64))
        if centers is None:
            centers = torch.empty((0, dim), dtype=torch.float64)
        centers = torch.as_tensor(centers, dtype=torch.float64)
        if centers.ndim != 2 or centers.shape[1] != dim or not torch.isfinite(centers).all():
            raise ValueError("锚点目录须为有限 [n_anchor,dim] 数组")
        n = len(centers)
        bandwidths = torch.ones(n, dtype=torch.float64) if bandwidths is None else torch.as_tensor(bandwidths, dtype=torch.float64)
        amplitudes = torch.zeros(n, dtype=torch.float64) if amplitudes is None else torch.as_tensor(amplitudes, dtype=torch.float64)
        if bandwidths.shape != (n,) or amplitudes.shape != (n,):
            raise ValueError("锚点、带宽和势垒幅度数量不符")
        if not torch.isfinite(bandwidths).all() or not (bandwidths > 0).all():
            raise ValueError("带宽须为有限正数")
        if not torch.isfinite(amplitudes).all() or not ((amplitudes >= 0) & (amplitudes <= 0.5)).all():
            raise ValueError("势垒幅度须位于 [0,1/2]")
        self.register_buffer("centers", centers.clone())
        self.register_buffer("bandwidths", bandwidths.clone())
        self.register_buffer("amplitudes", amplitudes.clone())
        self.alpha = nn.Parameter(torch.zeros(n, dtype=torch.float64))
        self.register_buffer("eye", torch.eye(dim, dtype=torch.float64))
        self.global_lipschitz = self.bound_certificate["total_lipschitz_upper"]
        required = max(1, math.ceil(4 * self.global_lipschitz))
        if steps is None:
            steps = required
        if not isinstance(steps, int) or steps < required:
            raise ValueError(f"固定网格不满足 δ Lip(f) ≤ 0.25；至少需要 {required} 步")
        self.n_steps = self.steps = steps
        self._knots = [0.0] * 4 + [j / (spline_count - 3) for j in range(1, spline_count - 3)] + [1.0] * 4
        self._center_provider=None
        if isinstance(time_centers,DiskCenters):
            if time_centers.shape!=(steps,n,dim):
                raise ValueError("磁盘移动中心 shape 与实际网格、锚点数或维数不符")
            self._center_provider=time_centers
            time_centers=None
        elif time_centers is not None:
            time_centers = torch.as_tensor(time_centers, dtype=torch.float64)
            if time_centers.shape != (steps, n, dim) or not torch.isfinite(time_centers).all():
                raise ValueError("传播中心须为有限 [steps,n_anchor,dim] 数组")
            time_centers=time_centers.detach()
        self.register_buffer("time_centers", time_centers)

    def center_provider_descriptor(self):
        return None if self._center_provider is None else self._center_provider.descriptor()

    def center_provider_source_state(self):
        return None if self._center_provider is None else self._center_provider.source_state()

    def validate_center_provider(self):
        if self._center_provider is not None:
            self._center_provider.validate()

    def centers_at(self,k):
        if self._center_provider is not None:
            return self._center_provider.at(k,device=self.centers.device)
        return self.centers if self.time_centers is None else self.time_centers[k]

    @property
    def U(self):
        return _spectral_matrix(self.raw_U)

    @property
    def V(self):
        return _spectral_matrix(self.raw_V)

    @property
    def W(self):
        return _spectral_matrix(self.raw_W)

    @property
    def bound_certificate(self) -> dict:
        base_speed = 2 * math.sqrt(65)
        base_lip = 8.0
        field_speed = float((self.amplitudes * math.exp(-0.5) / self.bandwidths).sum().detach()) / 4
        field_lip = float((self.amplitudes / self.bandwidths.square()).sum().detach()) / 4
        total = base_lip + field_lip + float(self.protection.lipschitz) * (base_speed + field_speed)
        return {
            "base_speed_upper": base_speed,
            "base_lipschitz_upper": base_lip,
            "field_speed_upper": field_speed,
            "field_lipschitz_upper": field_lip,
            "protection_lipschitz_upper": float(self.protection.lipschitz),
            "total_lipschitz_upper": total,
            "derivation": "Lip(bv) <= Lip(v)+sup|v| Lip(b); Gaussian grad <= exp(-1/2)/sigma, Hessian <= 1/sigma^2; t(1-t)<=1/4",
            "parameter_domain": "spectral U,V,W<=2; |C|<=1; 0<=alpha<=1",
        }

    def parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        lows, highs = [], []
        for name, p in self.named_parameters():
            if name == "alpha":
                low, high = 0.0, 1.0
            elif name == "C":
                low, high = -1.0, 1.0
            else:
                low, high = -2.0, 2.0
            lows.append(torch.full_like(p, low).reshape(-1))
            highs.append(torch.full_like(p, high).reshape(-1))
        return torch.cat(lows), torch.cat(highs)

    def validate_parameters(self):
        low, high = self.parameter_bounds()
        flat = torch.cat([p.reshape(-1) for p in self.parameters()])
        if not torch.isfinite(flat).all() or not ((flat >= low) & (flat <= high)).all():
            raise ValueError("模型参数超出已认证紧参数域")
        if not torch.isfinite(self.centers).all() or not torch.isfinite(self.bandwidths).all() or not (self.bandwidths > 0).all():
            raise ValueError("冻结锚点目录或带宽无效")
        if not torch.isfinite(self.amplitudes).all() or not ((self.amplitudes >= 0) & (self.amplitudes <= 0.5)).all():
            raise ValueError("冻结势垒幅度超出认证范围")
        if self.bound_certificate["total_lipschitz_upper"] / self.n_steps > 0.25:
            raise ValueError("当前冻结字段与固定网格的全局步界不符")

    def _single_velocity(self, z, k, U, V, W):
        hidden = torch.tanh(V @ torch.tanh(U @ z))
        features = torch.cat([hidden, z.new_ones(1)])
        velocity = W @ (self.spline_coefficients(k) * features)
        if len(self.centers):
            centers = self.centers_at(k)
            delta = z[None] - centers
            psi = torch.exp(-delta.square().sum(-1) / (2 * self.bandwidths.square()))
            field = (-delta * (self.alpha * self.amplitudes * psi / self.bandwidths.square())[:, None]).sum(0)
            t = k / self.n_steps
            velocity = velocity + (t * (1 - t)) * field
        return self.protection(z) * velocity

    def spline_coefficients(self, k: int) -> torch.Tensor:

        t = k / self.n_steps
        span = min(self.spline_count - 1, max(3, bisect_right(self._knots, t) - 1))
        values = [1.0, 0.0, 0.0, 0.0]
        left, right = [0.0] * 4, [0.0] * 4
        for j in range(1, 4):
            left[j] = t - self._knots[span + 1 - j]
            right[j] = self._knots[span + j] - t
            saved = 0.0
            for r in range(j):
                temporary = values[r] / (right[r + 1] + left[j - r])
                values[r] = saved + right[r + 1] * temporary
                saved = left[j - r] * temporary
            values[j] = saved
        return self.C.new_tensor(values) @ self.C[span - 3 : span + 1]

    def single_velocity(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return self._single_velocity(z, k, self.U, self.V, self.W)

    velocity_single = single_velocity

    def single_step(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.single_velocity(z, k) / self.n_steps

    def single_step_with_logdet(self, z: torch.Tensor, k: int):
        value = self.single_step(z, k)
        matrix = torch.func.jacrev(lambda x: self.single_step(x, k))(z)
        sign, logdet = torch.linalg.slogdet(matrix)
        return value, torch.where(sign > 0, logdet, z.new_tensor(float("nan")))

    def _check_batch(self, z):
        if z.ndim != 2 or z.shape[1] != self.dim or z.dtype != torch.float64:
            raise ValueError("输运输入须为 float64 [batch,dim]")
        if not torch.isfinite(z).all():
            raise FloatingPointError("输运输入非有限；整批失败")
        self.validate_parameters()

    def step(self, z: torch.Tensor, k: int, *, check: bool = True):
        if not 0 <= k < self.n_steps:
            raise IndexError("离散步编号超出固定网格")
        if check:
            self._check_batch(z)

        U, V, W = self.U, self.V, self.W
        single = lambda x: x + self._single_velocity(x, k, U, V, W) / self.n_steps
        y = torch.vmap(single)(z)
        matrix = torch.vmap(torch.func.jacrev(single))(z)
        sign, ld = torch.linalg.slogdet(matrix)
        ld = torch.where(sign > 0, ld, torch.full_like(ld, float("nan")))
        if check and (not torch.isfinite(y).all() or not torch.isfinite(ld).all()):
            raise FloatingPointError("实际有限步 Jacobian 或终点无效；整批失败")
        return y, ld

    def _checkpoint_step(self,z,k,matrices):

        track=torch.is_grad_enabled()
        with torch.enable_grad():
            point=z
            U,V,W=matrices
            first=torch.tanh(point@U.T)
            hidden=torch.tanh(first@V.T)
            coefficients=self.spline_coefficients(k)
            features=torch.cat((hidden,point.new_ones((len(point),1))),-1)
            velocity=(features*coefficients)@W.T
            hidden_derivative=(1-hidden.square())[:,:,None]*((V[None]*(1-first.square())[:,None,:])@U)
            derivative=(W[:,:64]*coefficients[:64])@hidden_derivative
            if len(self.centers):
                centers=self.centers_at(k)
                delta=point[:,None]-centers
                width2=self.bandwidths.square()
                psi=torch.exp(-delta.square().sum(-1)/(2*width2))
                amplitude=self.alpha*self.amplitudes*psi
                velocity=velocity+(k/self.n_steps)*(1-k/self.n_steps)*(-delta*(amplitude/width2)[:,:,None]).sum(1)
                hessian=delta[:,:,:,None]*delta[:,:,None,:]/width2[None,:,None,None].square()-self.eye/width2[None,:,None,None]
                derivative=derivative+(k/self.n_steps)*(1-k/self.n_steps)*(amplitude[:,:,None,None]*hessian).sum(1)
            b,grad_b=_ProtectionDerivative.apply(point,self.protection)
            matrix=self.eye+(b[:,None,None]*derivative+velocity[:,:,None]*grad_b[:,None,:])/self.n_steps
            sign,logdet=torch.linalg.slogdet(matrix)
            logdet=torch.where(sign>0,logdet,torch.full_like(logdet,float("nan")))
            result=point+b[:,None]*velocity/self.n_steps
        return (result,logdet) if track else (result.detach(),logdet.detach())

    def forward(self, z: torch.Tensor, *, check: bool = True, return_nodes: bool = False,
                checkpoint_steps: int = 0,check_budget=None):
        if check:
            self._check_batch(z)
        if type(checkpoint_steps) is not int or checkpoint_steps<0:
            raise ValueError("checkpoint_steps 必须为非负整数")
        if checkpoint_steps and return_nodes:
            raise ValueError("保存全部节点与节省节点内存的 checkpoint 模式不能同时使用")
        y, ld = z, z.new_zeros(len(z))
        nodes = [z] if return_nodes else None
        if checkpoint_steps:
            matrices=(self.U,self.V,self.W)
            for start in range(0,self.n_steps,checkpoint_steps):
                stop=min(start+checkpoint_steps,self.n_steps)
                def segment(point,first=start,last=stop):
                    if check_budget is not None:
                        check_budget()
                    subtotal=point.new_zeros(len(point))
                    for k in range(first,last):
                        point,term=self._checkpoint_step(point,k,matrices)
                        subtotal=subtotal+term
                    return point,subtotal
                if torch.is_grad_enabled():

                    y,term=checkpoint(segment,y,use_reentrant=False,preserve_rng_state=False)
                else:
                    y,term=segment(y)
                ld=ld+term
        else:
            for k in range(self.n_steps):
                if check_budget is not None:
                    check_budget()
                y, term = self.step(y, k, check=False)
                ld = ld + term
                if return_nodes:
                    nodes.append(y)
        if check and (not torch.isfinite(y).all() or not torch.isfinite(ld).all()):
            raise FloatingPointError("输运终点或完整有限步 Jacobian 非有限；整批失败")
        return (y, ld, nodes) if return_nodes else (y, ld)

    def single_inverse_step(self, y: torch.Tensor, k: int) -> torch.Tensor:

        U, V, W = self.U, self.V, self.W
        x = y
        for _ in range(self.inverse_iterations):
            x = y - self._single_velocity(x, k, U, V, W) / self.n_steps
        return x

    def inverse_step(self, y: torch.Tensor, k: int, *, check: bool = True):
        if check:
            self._check_batch(y)
        x = torch.vmap(lambda point: self.single_inverse_step(point, k))(y)
        restored, ld = self.step(x, k, check=False)
        if check:
            residual = (restored - y).abs().amax()
            if not torch.isfinite(x).all() or float(residual.detach()) > self.inverse_tolerance:
                raise RuntimeError("逆步压缩迭代未达到指定残差；整批失败")
        return x, -ld

    def inverse(self, y: torch.Tensor, *, check: bool = True):
        if check:
            self._check_batch(y)
        x, ld = y, y.new_zeros(len(y))
        for k in reversed(range(self.n_steps)):
            x, term = self.inverse_step(x, k, check=check)
            ld = ld + term
        return x, ld

    def log_weights(self, z: torch.Tensor, u0: Callable, u1: Callable, *,checkpoint_steps=0,check_budget=None):
        y, ld = self(z,checkpoint_steps=checkpoint_steps,check_budget=check_budget)
        result = torch.vmap(u0)(z) - torch.vmap(u1)(y) + ld
        if not torch.isfinite(result).all():
            raise FloatingPointError("精确权重非有限；不得截断或丢弃高权重点")
        return result, y

    def objective(self, z: torch.Tensor, u0: Callable, u1: Callable, *,checkpoint_steps=0,check_budget=None):
        weights, _ = self.log_weights(z,u0,u1,checkpoint_steps=checkpoint_steps,check_budget=check_budget)
        return torch.logsumexp(2 * weights, 0) - math.log(len(z))
