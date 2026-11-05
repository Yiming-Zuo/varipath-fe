from __future__ import annotations

import math

import torch
from torch import nn


class RadialProtection(nn.Module):


    def __init__(self, center, inner_radius: float, outer_radius: float):
        super().__init__()
        center = torch.as_tensor(center, dtype=torch.float64)
        if center.ndim != 1 or not torch.isfinite(center).all():
            raise ValueError("保护中心须为有限的一维向量")
        if not (0 < inner_radius < outer_radius < math.inf):
            raise ValueError("保护半径须满足 0 < inner < outer < inf")
        self.register_buffer("center", center.clone())
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.lipschitz = 1.875 / (self.outer_radius - self.inner_radius)
        self.compact_support = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        r = ((z - self.center).square().sum(-1) + 1e-30).sqrt()
        t = ((r - self.inner_radius) / (self.outer_radius - self.inner_radius)).clamp(0, 1)
        return 1 - t.pow(3) * (10 + t * (6 * t - 15))

    @property
    def certificate(self) -> dict:
        return {
            "kind": "radial_quintic",
            "center": self.center.detach().cpu().tolist(),
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "gradient_norm_upper": self.lipschitz,
            "derivation": "max S'(t)=15/8; norm is 1-Lipschitz",
            "physical_domain_certified": False,
        }
