from __future__ import annotations

import math
import torch

R_KJ = 0.00831446261815324


def logmeanexp(values):
    if values.ndim != 1 or len(values) == 0 or not torch.isfinite(values).all():
        raise FloatingPointError("A complete finite nonempty weight batch is required")
    return torch.logsumexp(values, 0) - math.log(len(values))


def log_population(log_weights, indicator):
    if indicator.dtype != torch.bool or indicator.shape != log_weights.shape:
        raise ValueError("Population indicators and weights must have the same shape")
    if not torch.isfinite(log_weights).all():
        raise FloatingPointError("Nonfinite weights invalidate the complete batch")
    if not indicator.any():
        return log_weights.new_tensor(-math.inf)
    return torch.logsumexp(log_weights[indicator], 0) - torch.logsumexp(log_weights, 0)


def estimate(log_weights, source_regions, target_regions, system, floor=1e-4):
    if not 0 < floor < 1:
        raise ValueError("The reporting floor must be between zero and one")
    logK = logmeanexp(log_weights)
    zero = torch.zeros_like(log_weights)
    lp = {f"{endpoint}_{region}": log_population(weights, regions[region])
          for endpoint, regions, weights in ((0, source_regions, zero),
                                             (1, target_regions, log_weights))
          for region in ("a", "b")}
    native_defined = all(bool(torch.isfinite(value)) for value in lp.values())
    def odds(values):
        return values["1_b"] - values["1_a"] - values["0_b"] + values["0_a"]
    stabilized = {key: value.clamp(min=math.log(floor)) for key, value in lp.items()}
    units = R_KJ * 300 if system == "ALA4" else 1.0
    sign = -units if system == "ALA4" else 1.0
    probabilities = {key: float(value.exp()) for key, value in lp.items()}
    result = dict(delta_f=float(-logK), delta_free_energy=float(-units * logK),
                  preference=float(sign * odds(stabilized)),
                  native_preference=float(sign * odds(lp)) if native_defined else None,
                  native_defined=native_defined, reporting_floor=floor,
                  floor_triggered=any(bool(v < math.log(floor)) for v in lp.values()),
                  endpoint_populations=probabilities,
                  log_second_moment=float(logmeanexp(2 * log_weights)),
                  log_relative_second_moment=float(logmeanexp(2 * log_weights) - 2 * logK),
                  ess=float(torch.exp(2 * torch.logsumexp(log_weights, 0)
                                      - torch.logsumexp(2 * log_weights, 0))),
                  sample_count=len(log_weights),
                  units="kJ/mol" if system == "ALA4" else "dimensionless")
    if "fine" in source_regions and "fine" in target_regions:
        cells = 25 if system == "ALA4" else 5
        result["fine_populations"] = {
            str(endpoint): [float(log_population(weights, regions["fine"] == k).exp())
                            for k in range(cells)]
            for endpoint, regions, weights in ((0, source_regions, zero),
                                               (1, target_regions, log_weights))}
    return result


def mismatch(proposal, reference):
    return 0.5 * sum(abs(float(proposal[key]) - float(reference[key]))
                     for key in ("a", "b", "remainder"))


def quota_shape_decomposition(pi, rho, conditional_chi2):
    pi, rho, conditional_chi2 = map(torch.as_tensor, (pi, rho, conditional_chi2))
    if (rho <= 0).any() or (pi < 0).any() or (conditional_chi2 < 0).any():
        raise ValueError("Risk decomposition requires positive proposal support")
    quota = ((pi - rho).square() / rho).sum()
    shape = (pi.square() / rho * conditional_chi2).sum()
    return quota, shape
