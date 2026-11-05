from __future__ import annotations

import math
import copy
import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from .readouts import R_KJ


def physical_potential(system, z, surface=1.):
    raw = getattr(system, "system", system)
    return raw.energy(system.positions(z), surface if system.name == "ALA4" else 1.)


def shape_metric(system, z):

    xyz = system.positions(z)
    A = torch.func.jacfwd(system.positions)(z).reshape(xyz.numel(), system.dim)
    eye = torch.eye(3, dtype=z.dtype, device=z.device)
    C = torch.linalg.cross(eye[:, None, :], xyz[None, :, :]).reshape(3, -1).T
    mass = system.chart.masses.repeat_interleave(3)
    H, I, J = A.T@(mass[:, None]*A), C.T@(mass[:, None]*C), A.T@(mass[:, None]*C)
    metric = H-J@torch.linalg.solve(I, J.T)
    return .5*(metric+metric.T)


def aligned_rmsd(first, second, masses=None):
    first, second = np.asarray(first), np.asarray(second)
    weights = np.ones(len(first)) if masses is None else np.asarray(masses)
    weights = weights/weights.sum()
    a = first-(first*weights[:, None]).sum(0)
    b = second-(second*weights[:, None]).sum(0)
    u, _, vh = np.linalg.svd(a.T@(weights[:, None]*b))
    u[:, -1] *= np.linalg.det(u@vh)
    return float(np.sqrt((weights[:, None]*(a@u@vh-b)**2).sum()))


def physical_regularity(system, z, protection):

    raw = getattr(system, "system", system)
    point = torch.as_tensor(z, dtype=torch.float64)
    support_radius = float(protection.normalized_margins(point).min())
    if not math.isfinite(support_radius) or support_radius <= 0:
        return dict(certified=False, reason="No open protected all-L neighborhood", radius=0.)
    x = system.positions(point)
    n = len(x)
    pair_lipschitz = protection.position_lipschitz[:, None]+protection.position_lipschitz[None, :]
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    distances = (x[:, None]-x[None, :]).norm(dim=-1)
    radii, screened = raw.energy.gb_or, raw.energy.gb_sr
    branches = {
        "screening_zero": distances+screened[None, :]-radii[:, None],
        "absolute_value": distances-screened[None, :],
        "max_lower_limit": radii[:, None]-(distances-screened[None, :]).abs(),
    }
    if raw.energy.gb_kind == "GBn2":
        radius = radii+raw.energy.offset
        branches["neck_step"] = radius[:, None]+radius[None, :]+.68-distances
    margins = {}
    for name, branch in branches.items():
        distance_to_branch = (branch.abs()/pair_lipschitz)[mask]
        margins[name] = float(distance_to_branch.min())
    radius = .5*min(support_radius,*margins.values())
    return dict(certified=math.isfinite(radius) and radius>0, radius=radius,
                support_radius=support_radius, branch_distance_lower_bounds=margins,
                method="1-Lipschitz normalized geometry margins and pair-distance bounds",
                scope="local_gb_chart")


def _whiten(matrix, cholesky):
    left = solve_triangular(cholesky,matrix,lower=True)
    return solve_triangular(cholesky,left.T,lower=True).T


def certify_stationary(potential, z, metric, gradient_tolerance=1e-3,
                       curvature_tolerance=1e-6, negative_modes=1,
                       regularity=None):

    if not all(math.isfinite(v) and v > 0 for v in (gradient_tolerance, curvature_tolerance)):
        raise ValueError("Stationary-point tolerances must be finite and positive")
    z = torch.as_tensor(z, dtype=torch.float64)
    grad_tensor = torch.func.grad(potential)(z)
    H_raw = torch.func.hessian(potential)(z).detach().numpy()
    G_tensor = metric(z)
    G_raw = G_tensor.detach().numpy()
    grad = grad_tensor.detach().numpy()
    if not all(np.isfinite(v).all() for v in (grad,H_raw,G_raw)):
        raise FloatingPointError("Nonfinite constrained stationary-point certificate")
    G = .5*(G_raw+G_raw.T)
    cholesky = np.linalg.cholesky(G)
    norm = float(np.sqrt(grad@np.linalg.solve(G, grad)))
    contravariant = torch.linalg.solve(G_tensor,grad_tensor).detach()
    A = torch.func.jacfwd(lambda point:metric(point)@contravariant)(z)
    DG = torch.func.jvp(metric,(z,),(contravariant,))[1]
    connection = (.5*(A+A.T-DG)).detach().numpy()
    white_raw = _whiten(H_raw-connection,cholesky)
    white = .5*(white_raw+white_raw.T)
    eigenvalues,Q = np.linalg.eigh(white)
    eigenvectors = solve_triangular(cholesky.T,Q,lower=False)
    connection_bound = float(np.linalg.norm(_whiten(connection,cholesky),ord=2))
    symmetry_bound = float(np.linalg.norm(.5*(white_raw-white_raw.T),ord=2))
    eig_residual = float(np.linalg.norm(white@Q-Q*eigenvalues[None,:],ord=2))
    orthogonality = float(np.linalg.norm(Q.T@Q-np.eye(len(Q)),ord=2))
    scale = max(1.,float(np.linalg.norm(white,ord=2)))
    floating_scale = 64*np.finfo(float).eps*max(1,len(z))*scale
    metric_asymmetry = float(np.linalg.norm(_whiten(.5*(G_raw-G_raw.T),cholesky),ord=2))
    envelope = connection_bound+symmetry_bound+eig_residual+orthogonality*scale+metric_asymmetry*scale+floating_scale
    if not all(math.isfinite(v) for v in (norm,envelope,connection_bound)):
        raise FloatingPointError("Nonfinite covariant curvature error envelope")
    negative = int((eigenvalues < -curvature_tolerance).sum())
    unresolved = int((np.abs(eigenvalues) <= curvature_tolerance+envelope).sum())
    local_regular = regularity is None or regularity.get("certified") is True
    return dict(certified=local_regular and norm <= gradient_tolerance and negative == negative_modes and unresolved == 0,
                gradient_mass_norm=norm, negative_modes=negative, unresolved_modes=unresolved,
                eigenvalues=eigenvalues.tolist(), mode=eigenvectors[:, 0].tolist(),
                method="mass-covariant-numerical-v2", gradient_tolerance=gradient_tolerance,
                curvature_tolerance=curvature_tolerance, curvature_error_envelope=envelope,
                residual_connection_bound=connection_bound, symmetry_error_bound=symmetry_bound,
                eigenpair_residual=eig_residual, metric_symmetry_error=metric_asymmetry,
                floating_scale_allowance=floating_scale, local_regularity=regularity,
                scope="numerical_stationary_point")


def dimer_search(potential, start, seed, force_limit=500, fmax=1e-4, max_steps=250, cost=None):

    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

    start = torch.as_tensor(start, dtype=torch.float64)
    d = len(start)
    carrier_dim = 3*math.ceil(d/3)
    calls = 0
    grad_value = torch.func.grad_and_value(potential)

    class Surface(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            nonlocal calls
            super().calculate(atoms, properties, system_changes)
            if calls >= force_limit:
                raise RuntimeError("Dimer force-call limit exhausted")
            calls += 1
            if cost is not None:
                cost["search_force_calls"] = calls
            q = torch.tensor(atoms.positions.ravel(), dtype=torch.float64)
            g, u = grad_value(q[:d])
            if not torch.isfinite(g).all() or not torch.isfinite(u):
                raise FloatingPointError("Nonfinite dimer energy/force")

            forces = np.zeros(carrier_dim)
            forces[:d] = -g.detach().numpy()
            forces[d:] = -q[d:].numpy()
            self.results = dict(energy=float(u.detach())+.5*float(q[d:].square().sum()),
                                forces=forces.reshape(-1, 3))

    positions = np.zeros(carrier_dim)
    positions[:d] = start.numpy()
    atoms = Atoms("H"*(carrier_dim//3), positions=positions.reshape(-1, 3), calculator=Surface())
    with DimerControl(initial_eigenmode_method="gauss", logfile=None,
                      eigenmode_logfile=None, maximum_translation=.05) as control:
        modes = MinModeAtoms(atoms, control, random_seed=int(seed))
        with MinModeTranslate(modes, logfile=None) as optimizer:
            optimizer.run(fmax=fmax, steps=max_steps)
        point = torch.tensor(modes.get_positions().ravel()[:d], dtype=torch.float64)
    return point, calls


def search_catalog(system, starts, fitting_source, protection, config, checkpoint_callback=None, check_budget=None,
                   initial_records=None):
    if fitting_source.ndim != 2 or fitting_source.shape[1] != system.dim:
        raise ValueError("Catalog bandwidths require this deployment's fitting source pool")
    surfaces = config.get("surfaces", [0., .5, 1.] if system.name == "ALA4" else [1.])
    records, accepted = [], []
    completed = {}
    for row in initial_records or []:
        key = (float(row["surface"]),int(row["start"]))
        if key in completed or row.get("status") not in ("failed","certified") or key[0] not in surfaces or not 0<=key[1]<len(starts):
            raise ValueError("Catalog recovery contains duplicated, incomplete, or incompatible starts")
        if row["status"]=="certified" and row.get("certificate",{}).get("method")!="mass-covariant-numerical-v2":
            raise ValueError("Recovered saddle certificate uses an incompatible curvature definition")
        completed[key] = copy.deepcopy(row)
    heavy = [r["index"] for r in system.metadata["atoms"] if r["element"] != "H"]
    metric = lambda z: shape_metric(system, z)
    for surface in surfaces:
        def potential(z):
            if check_budget is not None:
                check_budget()
            return physical_potential(system, z, surface)
        gv = torch.func.grad_and_value(potential)
        for index, start in enumerate(starts):
            if (float(surface),index) in completed:
                record = completed[(float(surface),index)]
                records.append(record)
                if record["status"]=="certified":
                    accepted.append(record)
                continue
            record = dict(surface=float(surface), start=index, status="failed")
            try:
                z, calls = dimer_search(potential, start, config["seed"]+index,
                                       force_limit=config.get("force_limit", 500),
                                       max_steps=config.get("max_steps", 250), cost=record)
                record["search_force_calls"] = calls
                if not bool(system.in_domain(z)):
                    raise ValueError("Dimer candidate outside all-L support")
                certificate = certify_stationary(potential, z, metric,
                    config.get("gradient_tolerance", 1e-3), config.get("curvature_tolerance", 1e-6),
                    regularity=physical_regularity(system,z,protection))
                record["certificate"] = certificate
                if not certificate["certified"]:
                    raise ValueError("Dimer candidate failed index-one certification")
                if float(protection(z)) < 1-1e-12:
                    raise ValueError("Certified saddle is outside the common protection plateau")
                mode = torch.tensor(certificate["mode"], dtype=torch.float64)
                mode = mode/mode.norm()
                minima = []
                for sign in (-1, 1):
                    evaluations = 0
                    clearance = float(protection.normalized_margins(z).min())
                    if not clearance > 0:
                        raise ValueError("No certified all-L descent neighborhood at saddle")
                    displacement = min(config.get("descent_displacement", .01), .25*clearance)
                    previous = z+sign*displacement*mode
                    descent_segments = []
                    def check_segment(q):
                        nonlocal previous
                        point = torch.tensor(q, dtype=torch.float64)
                        safe_radius = float(protection.normalized_margins(previous).min())
                        distance = float((point-previous).norm())
                        if not safe_radius > distance:
                            raise ValueError("Downhill segment lacks a continuous all-L support certificate")
                        descent_segments.append(dict(length=distance, safe_radius=safe_radius))
                        previous = point
                    def objective(q):
                        nonlocal evaluations
                        evaluations += 1
                        point = torch.tensor(q, dtype=torch.float64)
                        if not bool(system.in_domain(point)):
                            return 1e100, np.zeros_like(q)
                        g, value = gv(point)
                        if not torch.isfinite(g).all() or not torch.isfinite(value):
                            return 1e100, np.zeros_like(q)
                        return float(value.detach()), g.detach().numpy()
                    result = minimize(objective, previous.numpy(), callback=check_segment,
                                      jac=True, method="L-BFGS-B", options={"maxiter": config.get("descent_steps", 1000), "gtol": 1e-7, "ftol": 1e-14})
                    minimum = torch.tensor(result.x, dtype=torch.float64)
                    cert = certify_stationary(potential, minimum, metric,
                        config.get("gradient_tolerance", 1e-3), config.get("curvature_tolerance", 1e-6), negative_modes=0,
                        regularity=physical_regularity(system,minimum,protection))
                    if not bool(system.in_domain(minimum)) or not cert["certified"] or float(potential(minimum)) >= float(potential(z)):
                        raise ValueError("A downhill endpoint failed all-L minimum certification")
                    minima.append(dict(z=minimum.tolist(), energy=float(potential(minimum)),
                                       a=bool(system.regions(minimum)["a"]), certificate=cert,
                                       force_calls=evaluations, descent_segments=descent_segments))
                if aligned_rmsd(system.positions(torch.tensor(minima[0]["z"], dtype=torch.float64)).detach(),
                                system.positions(torch.tensor(minima[1]["z"], dtype=torch.float64)).detach()) < config.get("dedup_rmsd_nm", .001):
                    raise ValueError("The two downhill minima are geometrically indistinguishable")
                sigma = float(torch.quantile((fitting_source-z).norm(dim=-1), .1))
                barrier = float(potential(z))-min(m["energy"] for m in minima)
                amplitude = float(np.exp(-np.logaddexp(barrier/(2*R_KJ*300), -barrier/(2*R_KJ*300))))
                if not math.isfinite(sigma) or sigma <= 0 or barrier < 0:
                    raise ValueError("Undefined bandwidth or physical barrier")
                record.update(status="certified", z=z.tolist(), sigma=sigma, barrier_kjmol=barrier,
                              amplitude=amplitude, minima=minima,
                              target_connected=sum(m["a"] for m in minima) == 1)
                accepted.append(record)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                record["reason"] = str(exc)
            records.append(record)
            if checkpoint_callback:
                checkpoint_callback(records)
    ordered = sorted(accepted, key=lambda r: (-r["amplitude"], r["surface"], r["start"]))
    unique = []
    for row in ordered:
        positions = system.positions(torch.tensor(row["z"], dtype=torch.float64)).detach().numpy()[heavy]
        if all(aligned_rmsd(positions, system.positions(torch.tensor(other["z"], dtype=torch.float64)).detach().numpy()[heavy]) >= config.get("dedup_rmsd_nm", .001) for other in unique):
            unique.append(row)
    count = config.get("centers", 4)
    return dict(records=records, selected=unique[:count], complete=len(unique) >= count,
                selection="amplitude descending; surface then start; no conformational labels used")
