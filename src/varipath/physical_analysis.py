import math
import numpy as np
import torch

from .catalog import physical_potential
from .readouts import R_KJ
from .records import save_checkpoint
from .transport import Transport


def risk_partition(log_weights, labels, n_cells, reference_probabilities=None):

    log_weights = torch.as_tensor(log_weights, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long, device=log_weights.device)
    if labels.shape != log_weights.shape or not torch.isfinite(log_weights).all() or ((labels < 0)|(labels >= n_cells)).any():
        raise ValueError("Risk partitions need finite complete weights and exhaustive labels")
    r = torch.exp(log_weights-torch.logsumexp(log_weights, 0)+math.log(len(log_weights)))
    rho, pi, conditional, shape = [], [], [], []
    for k in range(n_cells):
        selected = labels == k
        q = float(selected.double().mean())
        p = float((r*selected).mean())
        rho.append(q)
        pi.append(p)
        if not selected.any():
            conditional.append(None)
            shape.append(0.)
        else:
            second = float((r.square()*selected).mean())
            shape.append(second-p*p/q)
            conditional.append(second*q/(p*p)-1 if p > 0 else None)
    unsupported = [k for k in range(n_cells) if rho[k] == 0 and
                   (reference_probabilities is not None and reference_probabilities[k] > 0)]
    quota = sum((p-q)**2/q for p, q in zip(pi, rho) if q > 0)
    total = float(r.square().mean()-1)
    optimal = (sum(p*math.sqrt(1+c) for p, c in zip(pi, conditional) if c is not None))**2-1
    return dict(status="unestimable" if unsupported else "diagnostic", proposal=rho,
                reweighted_target=pi, conditional_chi2=conditional,
                quota=quota, shape=sum(shape), total=total,
                identity_residual=abs(quota+sum(shape)-total),
                optimal_quota_risk_at_fixed_conditionals=optimal,
                unsupported_reference_cells=unsupported,
                scope="self_normalized_empirical_decomposition")


def freeze_energy_bins(source_polar, bins=16):
    source_polar = np.asarray(source_polar, dtype=float)
    if source_polar.ndim != 1 or not len(source_polar) or not np.isfinite(source_polar).all():
        raise ValueError("Energy bins must be frozen from ordinary D source energies")
    edges = np.unique(np.quantile(source_polar, np.linspace(0, 1, bins+1)[1:-1]))
    return np.r_[-np.inf, edges, np.inf]


def energy_shape_diagnostic(system, endpoints, log_weights, frozen_edges):
    if system.name != "ALA4":
        raise ValueError("Polar energy decomposition is defined for ALA4")
    edges = np.asarray(frozen_edges, dtype=float)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0) or edges[0] != -np.inf or edges[-1] != np.inf:
        raise ValueError("Frozen energy edges must cover the full real line")
    with torch.no_grad():
        polar = torch.stack([physical_potential(system, z, 1)-physical_potential(system, z, 0) for z in endpoints])
        energy_labels = torch.from_numpy(np.searchsorted(edges[1:-1], polar.cpu().numpy(), side="right")).to(endpoints.device)
        fine = system.regions(endpoints)["fine"]
        energy = risk_partition(log_weights, energy_labels, len(edges)-1)
        joint = risk_partition(log_weights, energy_labels*25+fine, (len(edges)-1)*25)
    return dict(energy=energy, energy_by_conformation=joint)


def pii_beta_preference(source_fine, target_fine):
    populations = np.asarray([source_fine, target_fine], dtype=float)
    if populations.shape != (2, 25) or not np.isfinite(populations).all() or (populations < 0).any():
        raise ValueError("PII/beta requires the same 25-cell physical populations")
    grid = populations.reshape(2, 5, 5)
    marginals = (grid.sum(1)+grid.sum(2))/2
    beta, pii = marginals[:, 1], marginals[:, 2]
    defined = bool((beta > 0).all() and (pii > 0).all())
    return dict(beta=beta.tolist(), PII=pii.tolist(), native_defined=defined,
                log_odds_difference=float(np.log(pii[1]/beta[1])-np.log(pii[0]/beta[0])) if defined else None)


def polar_closure(system, reference_bank):
    if system.name != "ALA4" or reference_bank["role"] != "R":
        raise ValueError("The polar closure uses only the independent ALA4 reference")
    states = np.asarray(reference_bank["states"])
    points = reference_bank["positions"]
    differences, conditional = [], []
    with torch.no_grad():
        for index, s in enumerate(states):
            q = points[:, :, index].flatten(0, 1)
            regions = system.regions(q)
            polar = torch.stack([physical_potential(system, z, 1)-physical_potential(system, z, 0) for z in q])
            if not regions["a"].any() or not regions["b"].any():
                return dict(status="unestimable", reason="Reference conditional core has no observations", state=float(s))
            a, b = float(polar[regions["a"]].mean()), float(polar[regions["b"]].mean())
            conditional.append(dict(s=float(s), a=a, b=b))
            differences.append(b-a)
    integral = float(np.trapezoid(differences, states))
    return dict(status="diagnostic", trapezoidal_integral_kjmol=integral,
                dimensionless_integral=integral/(R_KJ*300), conditional_polar=conditional,
                scope="Finite window quadrature; ladder/block and quadrature uncertainty must be propagated")


@torch.no_grad()
def anchor_exposure(model, source, check_budget=None):
    point = source
    exposure = source.new_zeros(len(model.centers))
    fractions = source.new_zeros(3)
    for k in range(model.n_steps):
        if check_budget:
            check_budget()
        b = model.protection(point)
        fractions += torch.stack(((b == 0).double().mean(), ((b > 0)&(b < 1)).double().mean(), (b == 1).double().mean()))
        centers = model.centers_at(k)
        kernel = torch.exp(-(point[:, None]-centers).square().sum(-1)/(2*model.bandwidths.square()))
        exposure += (b[:, None]*kernel).mean(0)
        point = torch.vmap(lambda z: model.single_step(z, k))(point)
    return dict(exposure=(exposure/model.n_steps).tolist(), protection_fractions=(fractions/model.n_steps).tolist(),
                strengths=model.alpha.detach().tolist(), scope="additional frozen-trajectory diagnostic cost")


def frozen_cross_diagnostic(v, v0, source, u0, u1, check_budget=None):

    models = {}
    for route_name, backbone in (("V", v), ("V0", v0)):
        for field in (False, True):
            try:
                model = Transport(backbone.dim, backbone.protection,
                    spline_count=backbone.spline_count, steps=backbone.n_steps,
                    centers=v.centers, bandwidths=v.bandwidths, amplitudes=v.amplitudes,
                    time_centers=v.time_centers)
                with torch.no_grad():
                    for name in ("raw_U", "raw_V", "raw_W", "C"):
                        getattr(model, name).copy_(getattr(backbone, name))
                    model.alpha.copy_(v.alpha if field else torch.zeros_like(v.alpha))
                models[route_name, field] = model
            except ValueError as exc:
                return dict(status="missing", reason=str(exc), route=route_name, field=field)
    values = {}
    with torch.no_grad():
        for key, model in models.items():
            if check_budget:
                check_budget()
            loga, _ = model.log_weights(source, u0, u1)
            values[key] = float(torch.logsumexp(2*loga, 0)-math.log(len(loga)))
    dv, dv0 = values["V", True]-values["V", False], values["V0", True]-values["V0", False]
    return dict(status="diagnostic", d_V=dv, d_V0=dv0, interaction=dv-dv0,
                moments={f"{r}_{'on' if f else 'off'}":value for (r, f), value in values.items()})
