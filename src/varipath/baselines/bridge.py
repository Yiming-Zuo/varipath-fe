import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, log_expit, logsumexp


def _ratios(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or np.isnan(result).any() or np.isposinf(result).any():
        raise ValueError("密度比必须是非空向量；零目标支撑可用负无穷表示")
    return result


def nebar(forward_log_ratio, reverse_log_ratio, min_overlap=1e-8):

    a, b = _ratios(forward_log_ratio), _ratios(reverse_log_ratio)
    finite_a, finite_b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(finite_a) or not len(finite_b):
        raise ValueError("两向路径支撑无可观测重叠，NEBAR无合法解")
    if not 0 <= min_overlap < 1:
        raise ValueError("重叠阈值必须在[0,1)内")
    log_count_ratio = math.log(len(b) / len(a))

    def equation(kappa):
        return float(logsumexp(log_expit(a - kappa + log_count_ratio))
                     - logsumexp(log_expit(b + kappa - log_count_ratio)))

    lower = min(finite_a.min(), -finite_b.max()) - 100
    upper = max(finite_a.max(), -finite_b.min()) + 100
    kappa = brentq(equation, lower, upper, xtol=1e-12, rtol=1e-14)
    overlap_f = float(expit(a - kappa + log_count_ratio).mean())
    overlap_r = float(expit(b + kappa - log_count_ratio).mean())
    if min(overlap_f, overlap_r) <= min_overlap:
        raise ValueError("双向路径重叠不足，禁止用FEP补作BAR成绩")
    return {
        "log_partition_ratio": float(kappa), "dimensionless_free_energy": float(-kappa),
        "solver_log_equation_residual": abs(equation(kappa)),
        "forward_log_fep": float(logsumexp(a) - math.log(len(a))),
        "reverse_log_fep": float(-logsumexp(b) + math.log(len(b))),
        "forward_reverse_membership": overlap_f,
        "reverse_forward_membership": overlap_r,
        "forward_paths": len(a), "reverse_paths": len(b),
    }


def bar(forward_work, reverse_work, min_overlap=1e-8):

    a, b = np.asarray(forward_work), np.asarray(reverse_work)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("确定性双向功必须有限")
    result = nebar(-a, -b, min_overlap)
    result["free_energy"] = result["dimensionless_free_energy"]
    return result


def _classify(system, points):
    import torch

    result = system.regions(torch.as_tensor(points, dtype=torch.float64))
    if isinstance(result, dict):
        if "valid" in result and not bool(result["valid"].all()):
            raise ValueError("区域几何不可判定，不能默认归入remainder")
        result = torch.stack([result["a"], result["b"]], -1)
    if isinstance(result, torch.Tensor):
        result = result.detach().cpu().numpy()
    result = np.asarray(result)
    if result.dtype != bool or result.shape != (len(points), 2) or (result.sum(1) > 1).any():
        raise ValueError("两端读出需要预定的不交硬区域")
    return result


def bridge_readout(forward_path, reverse_path, a, b, kappa, classify, eta=1e-4):

    fp, rp = np.asarray(forward_path), np.asarray(reverse_path)
    a, b = _ratios(a), _ratios(b)
    if (fp.ndim != 3 or rp.ndim != 3 or fp.shape[1:] != rp.shape[1:]
            or len(a) != len(fp) or len(b) != len(rp) or fp.shape[1] < 2
            or not np.isfinite(fp).all() or not np.isfinite(rp).all()
            or not math.isfinite(kappa) or not 0 < eta < 1):
        raise ValueError("桥接需要完整路径、对应计数和合法地板")
    common_ratio = np.r_[a, -b]
    fraction = len(a) / (len(a) + len(b))
    target_lw = -np.logaddexp(math.log(fraction) + kappa - common_ratio, math.log1p(-fraction))
    source_lw = -np.logaddexp(math.log(fraction), math.log1p(-fraction) + common_ratio - kappa)
    source_points = np.concatenate([fp[:, 0], rp[:, -1]])
    target_points = np.concatenate([fp[:, -1], rp[:, 0]])
    log_probabilities, ess = [], []
    for points, weights in ((source_points, source_lw), (target_points, target_lw)):
        flags = np.asarray(classify(points))
        if flags.shape != (len(points), 2) or flags.dtype != bool or (flags.sum(1) > 1).any():
            raise ValueError("硬区域必须不交且可判定")
        denominator = logsumexp(weights)
        if not math.isfinite(denominator):
            raise ValueError("桥接权重没有正质量")
        log_probabilities.append([logsumexp(weights[flags[:, k]]) - denominator for k in range(2)])
        ess.append(float(np.exp(2 * denominator - logsumexp(2 * weights))))
    values = np.asarray(log_probabilities)
    zero = np.isneginf(values)
    stable = np.maximum(values, math.log(eta))
    raw = None if zero.any() else float(-np.diff(values[:, 1] - values[:, 0])[0])
    return {
        "endpoint_region_probabilities": np.exp(values).tolist(),
        "endpoint_region_log_probabilities": [[None if not np.isfinite(v) else float(v) for v in row] for row in values],
        "raw_dimensionless_preference": raw,
        "reported_dimensionless_preference": float(-np.diff(stable[:, 1] - stable[:, 0])[0]),
        "probability_floor": eta, "floor_triggered": bool((values < math.log(eta)).any()),
        "zero_regions_by_endpoint": zero.tolist(), "endpoint_bridge_ess": ess,
    }
