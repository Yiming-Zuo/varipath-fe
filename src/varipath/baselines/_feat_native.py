import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

UPSTREAM_COMMIT = "0fb77e1acee9c3806e6c9aa3fce6ae3e1745ba84"


class Coef:
    def __init__(self, coef, a=1.0, b=0.001):
        self.coef = coef
        self.a = a
        self.b = b

    def t(self, t):
        if self.coef == "t":
            return t
        elif self.coef == "1-t":
            return 1 - t
        elif self.coef == "cos(pi*t/2)":
            return torch.cos(1 / 2 * np.pi * t)
        elif self.coef == "sin(pi*t/2)":
            return torch.sin(1 / 2 * np.pi * t)
        elif self.coef == "sqrt(a*t*(1-t))":
            return torch.sqrt(self.a * t * (1 - t))
        elif self.coef == "sqrt(a*t*(1-t))+b":
            return torch.sqrt(self.a * t * (1 - t)) + self.b
        elif self.coef == "a*t*(1-t)":
            return self.a * t * (1 - t)
        elif self.coef == "a*sin(pi*t)":
            return self.a * torch.sin(np.pi * t)
        else:
            return torch.zeros_like(t) + float(self.coef)

    def dt(self, t):
        if self.coef == "t":
            return torch.ones_like(t)
        elif self.coef == "1-t":
            return -torch.ones_like(t)
        elif self.coef == "cos(pi*t/2)":
            return -torch.sin(1 / 2 * np.pi * t) * 1 / 2 * np.pi
        elif self.coef == "sin(pi*t/2)":
            return torch.cos(1 / 2 * np.pi * t) * 1 / 2 * np.pi
        elif self.coef == "sqrt(a*t*(1-t))":
            return 1 / 2 / torch.sqrt(t * (1 - t)) * self.a**0.5 * (1 - 2 * t)
        elif self.coef == "sqrt(a*t*(1-t))+b":
            return 1 / 2 / torch.sqrt(t * (1 - t)) * self.a**0.5 * (1 - 2 * t)
        elif self.coef == "a*t*(1-t)":
            return self.a - 2 * t * self.a
        elif self.coef == "a*sin(pi*t)":
            return self.a * np.pi * torch.cos(np.pi * t)
        else:
            return torch.zeros_like(t)

    def tdt(self, t):
        if self.coef == "sqrt(a*t*(1-t))":
            return 1 / 2 * self.a * (1 - 2 * t)
        else:
            return self.t(t) * self.dt(t)


def remove_mean(samples, n_particles, n_dimensions):

    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        x = (x + 0.1).log() / 4
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class score_PotentialNet(nn.Module):
    def __init__(
        self,
        dim,
        device,
        alpha,
        beta,
        gamma,
        num_layers: int = 4,
        num_hid: int = 256,
        clip: float = 10000.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_hid = num_hid
        self.clip = clip
        self.time_embed = PositionalEmbedding(self.num_hid)
        self.time_coder_state = nn.Sequential(
            *[
                nn.Linear(self.num_hid, self.num_hid),
                nn.GELU(),
                nn.Linear(self.num_hid, self.num_hid),
            ]
        )
        self.state_time_net = (
            [
                nn.Sequential(
                    *[nn.Linear(self.num_hid + self.dim, self.num_hid), nn.GELU()]
                )
            ]
            + [
                nn.Sequential(*[nn.Linear(self.num_hid, self.num_hid), nn.GELU()])
                for _ in range(self.num_layers - 1)
            ]
            + [nn.Linear(self.num_hid, self.dim)]
        )
        self.state_time_net = nn.Sequential(*self.state_time_net)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _forward(self, input_array, time_array, *args, **kwargs):
        time_array_emb = self.time_embed(time_array)
        t_net1 = self.time_coder_state(time_array_emb)
        extended_input = torch.cat((input_array, t_net1), -1)
        out_state = self.state_time_net(extended_input)
        return out_state

    def score(self, input_array, time_array, clip=False, *args, **kwargs):
        return self._forward(input_array, time_array)

    def forward(self, input_array, time_array, *args, **kwargs):
        raise NotImplementedError(
            "MW_PotentialNet does not support forward method, use score method instead."
        )


class score_VectorFieldNet(nn.Module):
    def __init__(
        self,
        dim,
        device,
        num_layers: int = 4,
        num_hid: int = 256,
        clip: float = 10000.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_hid = num_hid
        self.clip = clip
        self.time_embed = PositionalEmbedding(self.num_hid)
        self.time_coder_state = nn.Sequential(
            *[
                nn.Linear(self.num_hid, self.num_hid),
                nn.GELU(),
                nn.Linear(self.num_hid, self.num_hid),
            ]
        )
        self.state_time_net = (
            [
                nn.Sequential(
                    *[nn.Linear(self.num_hid + self.dim, self.num_hid), nn.GELU()]
                )
            ]
            + [
                nn.Sequential(*[nn.Linear(self.num_hid, self.num_hid), nn.GELU()])
                for _ in range(self.num_layers - 1)
            ]
            + [nn.Linear(self.num_hid, self.dim)]
        )
        self.state_time_net = nn.Sequential(*self.state_time_net)

    def forward(self, input_array, time_array, *args, **kwargs):
        time_array_emb = self.time_embed(time_array)
        t_net1 = self.time_coder_state(time_array_emb)
        extended_input = torch.cat((input_array, t_net1), -1)
        out_state = self.state_time_net(extended_input)
        return torch.clip(out_state, -self.clip, self.clip)

    def div(self, input_array, time_array, *args, **kwargs):
        assert len(input_array.shape) == 2
        with torch.enable_grad():
            input_array = input_array.requires_grad_(True)
            out = torch.autograd.functional.jacobian(
                lambda x: self.forward(x, time_array).sum(0),
                input_array,
                create_graph=True,
                vectorize=True,
            )
            div = out.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
            return div


def b_loss(
    x1: torch.Tensor,
    x2: torch.Tensor,
    alpha: Coef,
    beta: Coef,
    gamma: Coef,
    vector_field: callable,
    center_noise: bool = False,
    n_particles: int = None,
    n_dim: int = None,
):
    device = x1.device
    batch_size = x1.shape[0]
    dim_size = [1 for _ in x1.shape[1:]]
    t = torch.clip(torch.rand(batch_size).to(device), 0, 1)
    z = torch.randn_like(x1)
    if center_noise:
        z = remove_mean(z, n_particles, n_dim)
    x_int = (
        alpha.t(t).reshape(-1, *dim_size) * x1
        + beta.t(t).reshape(-1, *dim_size) * x2
        + gamma.t(t).reshape(-1, *dim_size) * z
    )
    dt_x_int = (
        alpha.dt(t).reshape(-1, *dim_size) * x1 + beta.dt(t).reshape(-1, *dim_size) * x2
    )
    b_t = vector_field(x_int, t)
    target = dt_x_int + gamma.dt(t).reshape(-1, *dim_size) * z
    loss = 1 / 2 * (b_t**2).reshape(batch_size, -1).sum(-1) - (target * b_t).reshape(
        batch_size, -1
    ).sum(-1)
    x_int = (
        alpha.t(t).reshape(-1, *dim_size) * x1
        + beta.t(t).reshape(-1, *dim_size) * x2
        - gamma.t(t).reshape(-1, *dim_size) * z
    )
    b_t = vector_field(x_int, t)
    target = dt_x_int - gamma.dt(t).reshape(-1, *dim_size) * z
    loss += 1 / 2 * (b_t**2).reshape(batch_size, -1).sum(-1) - (target * b_t).reshape(
        batch_size, -1
    ).sum(-1)
    return loss.mean()


def eta_loss(
    x1,
    x2,
    alpha: Coef,
    beta: Coef,
    gamma: Coef,
    score_net: callable,
    center_noise: bool = False,
    n_particles: int = None,
    n_dim: int = None,
):
    device = x1.device
    batch_size = x1.shape[0]
    dim_size = [1 for _ in x1.shape[1:]]
    t = torch.rand(batch_size).to(device)
    z = torch.randn_like(x1)
    if center_noise:
        z = remove_mean(z, n_particles, n_dim)
    x_int = (
        alpha.t(t).reshape(-1, *dim_size) * x1
        + beta.t(t).reshape(-1, *dim_size) * x2
        + gamma.t(t).reshape(-1, *dim_size) * z
    )
    score = score_net.score(x_int, t)
    eta = -score
    loss = gamma.t(t).flatten() * 1 / 2 * (eta**2).reshape(batch_size, -1).sum(-1) - (
        z * eta
    ).reshape(batch_size, -1).sum(-1)
    x_int = (
        alpha.t(t).reshape(-1, *dim_size) * x1
        + beta.t(t).reshape(-1, *dim_size) * x2
        - gamma.t(t).reshape(-1, *dim_size) * z
    )
    score = score_net.score(x_int, t)
    eta = -score
    loss += gamma.t(t).flatten() * 1 / 2 * (eta**2).reshape(batch_size, -1).sum(-1) + (
        z * eta
    ).reshape(batch_size, -1).sum(-1)
    return loss.mean()


def tsm_loss(
    x1: torch.Tensor,
    x2: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    alpha: Coef,
    beta: Coef,
    gamma: Coef,
    score_net: callable,
    center_noise: bool = False,
    n_particles: int = None,
    n_dim: int = None,
):
    device = x1.device
    batch_size = x1.shape[0]
    dim_size = [1 for _ in x1.shape[1:]]
    t = torch.rand(batch_size).to(device) * 0.5
    z = torch.randn_like(x1)
    if center_noise:
        z = remove_mean(z, n_particles, n_dim)
    x_int = (
        alpha.t(t).reshape(-1, *dim_size) * x1
        + beta.t(t).reshape(-1, *dim_size) * x2
        + gamma.t(t).reshape(-1, *dim_size) * z
    )
    score = score_net.score(x_int, t)
    target = s1
    loss = alpha.t(t).flatten() * 1 / 2 * (score**2).reshape(batch_size, -1).sum(-1) - (
        target * score
    ).reshape(batch_size, -1).sum(-1)
    x_int = (
        alpha.t(t).reshape(-1, *dim_size) * x1
        + beta.t(t).reshape(-1, *dim_size) * x2
        - gamma.t(t).reshape(-1, *dim_size) * z
    )
    score = score_net.score(x_int, t)
    target = s1
    loss += alpha.t(t).flatten() * 1 / 2 * (score**2).reshape(batch_size, -1).sum(
        -1
    ) - (target * score).reshape(batch_size, -1).sum(-1)
    t = torch.rand(batch_size).to(device) * 0.5 + 0.5
    z = torch.randn_like(x1)
    if center_noise:
        z = remove_mean(z, n_particles, n_dim)
    x_int = (
        alpha.t(t).reshape(-1, *dim_size) * x1
        + beta.t(t).reshape(-1, *dim_size) * x2
        + gamma.t(t).reshape(-1, *dim_size) * z
    )
    score = score_net.score(x_int, t)
    target = s2
    loss += beta.t(t).flatten() * 1 / 2 * (score**2).reshape(batch_size, -1).sum(-1) - (
        target * score
    ).reshape(batch_size, -1).sum(-1)
    x_int = (
        alpha.t(t).reshape(-1, *dim_size) * x1
        + beta.t(t).reshape(-1, *dim_size) * x2
        - gamma.t(t).reshape(-1, *dim_size) * z
    )
    score = score_net.score(x_int, t)
    target = s2
    loss += beta.t(t).flatten() * 1 / 2 * (score**2).reshape(batch_size, -1).sum(-1) - (
        target * score
    ).reshape(batch_size, -1).sum(-1)
    return loss.mean()


def Jarzynski_integrate(
    x_init: torch.Tensor,
    init_logprob: torch.Tensor,
    eps: Coef,
    alpha: Coef,
    beta: Coef,
    gamma: Coef,
    vector_field: callable,
    score_net: callable,
    collect_interval: int = 10,
    learn_b: bool = True,
    times: torch.Tensor = None,
    forward: bool = True,
    return_A: bool = False,
    target_logp: callable = None,
    center_noise: bool = False,
    n_particles: int = None,
    n_dim: int = None,
):
    t_start = times[0]
    t_end = times[-1]
    if forward:
        assert t_start < t_end
    else:
        assert t_start > t_end
    device = x_init.device
    batch_size = x_init.shape[0]
    dim_size = [1 for _ in x_init.shape[1:]]
    A = torch.zeros(batch_size).to(device)
    t = torch.zeros(batch_size).to(device) + t_start
    x = x_init
    Xs = [x]
    weighted_Xs = []
    ESS = []
    A -= init_logprob
    for i in range(len(times) - 1):
        step_size = np.abs((times[i + 1] - times[i]).item())
        if i == 0:
            print(x.shape)
            vf = vector_field(x, t)
            s = score_net.score(x, t)
            if not learn_b:
                vf = vf - gamma.tdt(t).reshape(-1, *dim_size) * s
        std = torch.sqrt(2 * step_size * eps.t(t).reshape(-1, *dim_size))
        if forward:
            noise = torch.randn_like(x)
            if center_noise:
                noise = remove_mean(noise, n_particles, n_dim)
            x_new = (
                x
                + step_size * vf
                + step_size * eps.t(t).reshape(-1, *dim_size) * s
                + std * noise
            )
            t_new = t * 0 + times[i + 1]
            A -= (
                Normal(
                    x
                    + step_size * vf
                    + step_size * eps.t(t).reshape(-1, *dim_size) * s,
                    std,
                    validate_args=False,
                )
                .log_prob(x_new)
                .sum(-1)
            )
            vf_new = vector_field(x_new, t_new)
            if not learn_b:
                vf_new = vf_new - gamma.tdt(t_new).reshape(-1, *dim_size) * s
            s_new = score_net.score(x_new, t_new)
            std_new = torch.sqrt(2 * step_size * eps.t(t_new).reshape(-1, *dim_size))
            A += (
                Normal(
                    x_new
                    - step_size * vf_new
                    + step_size * eps.t(t_new).reshape(-1, *dim_size) * s_new,
                    std_new,
                    validate_args=False,
                )
                .log_prob(x)
                .sum(-1)
            )
        else:
            noise = torch.randn_like(x)
            if center_noise:
                noise = remove_mean(noise, n_particles, n_dim)
            x_new = (
                x
                - step_size * vf
                + step_size * eps.t(t).reshape(-1, *dim_size) * s
                + std * noise
            )
            t_new = t * 0 + times[i + 1]
            A -= (
                Normal(
                    x
                    - step_size * vf
                    + step_size * eps.t(t).reshape(-1, *dim_size) * s,
                    std,
                    validate_args=False,
                )
                .log_prob(x_new)
                .sum(-1)
            )
            vf_new = vector_field(x_new, t_new)
            if not learn_b:
                vf_new = vf_new - gamma.tdt(t_new).reshape(-1, *dim_size) * s
            s_new = score_net.score(x_new, t_new)
            std_new = torch.sqrt(2 * step_size * eps.t(t_new).reshape(-1, *dim_size))
            A += (
                Normal(
                    x_new
                    + step_size * vf_new
                    + step_size * eps.t(t_new).reshape(-1, *dim_size) * s_new,
                    std_new,
                    validate_args=False,
                )
                .log_prob(x)
                .sum(-1)
            )
        x = x_new
        t = t_new
        vf = vf_new
        s = s_new
        if (i + 1) % collect_interval == 0 or i == len(times) - 2:
            weighted_Xs.append((x, t.clone()))
        Xs.append(x)
    if return_A:
        A_final = A + target_logp(x).flatten()
        return (x, Xs, A_final)
    return (x, Xs)
