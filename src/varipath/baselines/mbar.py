from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.special import logsumexp
import torch


class OverlapError(ValueError):
    pass


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _connected(matrix, threshold):
    edges = np.minimum(matrix, matrix.T) > threshold
    reached = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for other in np.flatnonzero(edges[current]):
            if int(other) not in reached:
                reached.add(int(other))
                pending.append(int(other))
    return len(reached) == len(matrix)


def fit_mbar(
    u_kn,
    counts,
    *,
    endpoint_u=None,
    region_flags=None,
    min_overlap=1e-8,
    sample_ids=None,
):

    from pymbar import MBAR

    u = np.array(_array(u_kn), dtype=np.float64, copy=True)
    raw_counts = np.asarray(_array(counts))
    if (
        u.ndim != 2
        or u.shape[0] < 2
        or raw_counts.shape != (u.shape[0],)
        or not np.isfinite(u).all()
        or not np.isfinite(raw_counts).all()
        or np.any(raw_counts < 0)
        or np.any(raw_counts != np.floor(raw_counts))
        or raw_counts.sum() != u.shape[1]
        or np.count_nonzero(raw_counts) < 2
        or not np.isfinite(min_overlap)
        or min_overlap < 0
        or min_overlap >= 1
    ):
        raise ValueError("MBAR requires finite cross energies and exact counts for at least two sampled windows")
    n_k = raw_counts.astype(np.int64)
    if sample_ids is not None:
        identifiers = list(sample_ids)
        if len(identifiers) != u.shape[1]:
            raise ValueError("Every MBAR sample must have exactly one record ID")
        try:
            unique = len(set(identifiers))
        except TypeError as error:
            raise ValueError("MBAR record IDs must be hashable") from error
        if unique != len(identifiers):
            raise ValueError("A saved trajectory record cannot be counted twice")
    physical = u[[0, -1]].copy() if endpoint_u is None else np.asarray(_array(endpoint_u), dtype=np.float64)
    if physical.shape != (2, u.shape[1]) or not np.isfinite(physical).all():
        raise ValueError("Unbiased endpoint energies must be finite with shape (2,N)")
    flags = None
    if region_flags is not None:
        flags = np.asarray(_array(region_flags))
        if flags.ndim != 2 or flags.shape[0] != u.shape[1] or not np.isin(flags, (0, 1)).all():
            raise ValueError("Region observables must be hard indicators with shape (N,R)")

    model = MBAR(
        u,
        n_k,
        maximum_iterations=10000,
        relative_tolerance=1e-11,
        solver_protocol="robust",
        verbose=False,
    )
    if not np.isfinite(model.f_k).all():
        raise FloatingPointError("MBAR returned nonfinite free energies")
    overlap = model.compute_overlap()
    matrix = np.asarray(overlap["matrix"], dtype=np.float64)
    sampled = np.flatnonzero(n_k)
    sampled_matrix = matrix[np.ix_(sampled, sampled)]
    if not np.isfinite(matrix).all() or not _connected(sampled_matrix, min_overlap):
        raise OverlapError("Sampled MBAR windows have a disconnected overlap network")

    log_counts = np.full(len(n_k), -np.inf)
    log_counts[sampled] = np.log(n_k[sampled])
    log_denom = logsumexp(log_counts[:, None] + model.f_k[:, None] - u, axis=0)
    fixed_point_f = -logsumexp(-u - log_denom[None], axis=1)
    residual = float(np.max(np.abs((fixed_point_f - fixed_point_f[0]) - (model.f_k - model.f_k[0]))))
    if not np.isfinite(residual) or residual > 1e-7:
        raise FloatingPointError(f"MBAR fixed-point residual too large: {residual:g}")

    perturbed = model.compute_perturbed_free_energies(physical, compute_uncertainty=False)
    delta_f = float(perturbed["Delta_f"][0, 1])
    log_weights = -physical - log_denom[None]
    log_weights -= logsumexp(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights)
    result = {
        "endpoint_delta_f": delta_f,
        "physical_log_partition_ratio": -delta_f,
        "window_free_energies": (model.f_k - model.f_k[0]).tolist(),
        "counts": n_k.tolist(),
        "overlap_matrix": matrix.tolist(),
        "overlap_scalar": float(np.real(overlap["scalar"])),
        "fixed_point_residual": residual,
        "endpoint_weight_ess": (1 / np.square(weights).sum(axis=1)).tolist(),
        "endpoint_max_weight_share": weights.max(axis=1).tolist(),
        "uncertainty_scope": "No independence or equilibrium certification; use independent-ladder resampling in the workflow",
    }
    if flags is not None:
        logs = np.full((2, flags.shape[1]), -np.inf)
        for index in range(flags.shape[1]):
            selected = flags[:, index].astype(bool)
            if np.any(selected):
                logs[:, index] = logsumexp(log_weights[:, selected], axis=1)
        result["endpoint_log_probabilities"] = logs.tolist()
        result["endpoint_probabilities"] = np.exp(logs).tolist()
        result["zero_regions_by_endpoint"] = np.isneginf(logs).tolist()
    return result


@dataclass(frozen=True)
class EnergyBias:


    coordinate: Callable
    center: float
    stiffness: float

    def __post_init__(self):
        if not np.isfinite((self.center, self.stiffness)).all() or self.stiffness < 0:
            raise ValueError("Energy-bias parameters must be finite with nonnegative stiffness")

    def __call__(self, z):
        return 0.5 * self.stiffness * (self.coordinate(z) - self.center) ** 2


@dataclass(frozen=True)
class PeriodicCoreBias:


    angles: Callable
    centers: Sequence[Sequence[float]]
    width: float
    strength: float

    def __post_init__(self):
        centers = np.asarray(self.centers)
        if centers.ndim != 2 or not centers.size or not np.isfinite(centers).all() or not np.isfinite((self.width, self.strength)).all() or self.width <= 0:
            raise ValueError("Core bias requires finite periodic centers, positive width, and finite strength")

    def __call__(self, z):
        angles = self.angles(z)
        centers = torch.as_tensor(self.centers, dtype=angles.dtype, device=angles.device)
        if angles.shape[-1] != centers.shape[1]:
            raise ValueError("Core-bias angle dimension differs from its centers")
        kernels = torch.exp(-((1 - torch.cos(angles[..., None, :] - centers)) / self.width**2).sum(-1))
        return -self.strength * (1 - (1 - kernels).prod(-1))


def window_potentials(samples, states, potential, biases=None):

    if samples.ndim != 2 or not torch.isfinite(samples).all() or not len(states):
        raise ValueError("Cross-energy evaluation requires a nonempty finite coordinate batch and states")
    if biases is None:
        biases = [None] * len(states)
    if len(biases) != len(states):
        raise ValueError("Each window must have its explicit bias or None")
    rows = []
    for state, bias in zip(states, biases):
        energy = torch.stack([potential(z, state) for z in samples])
        if bias is not None:
            energy = energy + torch.stack([bias(z) for z in samples])
        if energy.shape != (len(samples),) or not torch.isfinite(energy).all():
            raise FloatingPointError("Actual window cross energy is invalid")
        rows.append(energy)
    return torch.stack(rows)
