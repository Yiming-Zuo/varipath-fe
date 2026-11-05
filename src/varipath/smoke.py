from pathlib import Path
import math
import time

import numpy as np
import torch

from .baselines.feat import FEAT, FEATConfig
from .baselines.mnf import PeriodicMNF, MNFConfig
from .baselines.mbar import fit_mbar
from .benchmarks import HarmonicSystem, PeriodicSystem
from .physics import MolecularSystem
from .protection import RadialProtection
from .readouts import estimate
from .records import read_json, run_lock, save_checkpoint, write_json
from .sampling import HMCSampler, ReplicaExchangeSampler
from .training import train_transport, TrainingConfig
from .transport import Transport


def _mbar_control(system, biased, seed):
    rng = np.random.default_rng(seed)
    states = [0., .5, 1.]
    pools, biases = [], []
    precision1, mean1 = np.array([1.21, .81]), np.array([.35, -.2])
    for state in states:
        k = .8 if biased and state == .5 else 0.
        center = -.4
        precision = (1-state)+state*precision1+np.array([k, 0.])
        linear = state*precision1*mean1+np.array([k*center, 0.])
        pools.append(torch.tensor(rng.normal(size=(128, 2))/np.sqrt(precision)+linear/precision, dtype=torch.float64))
        biases.append(k)
    points = torch.cat(pools)
    u0, u1 = system.effective_u(points, 0), system.effective_u(points, 1)
    cross = torch.stack([(1-s)*u0+s*u1+.5*k*(points[:, 0]+.4).square()
                         for s, k in zip(states, biases)])
    regions = system.regions(points)
    return fit_mbar(cross, [128]*3, endpoint_u=torch.stack((u0, u1)),
                    region_flags=torch.stack((regions["a"], regions["b"]), -1),
                    sample_ids=[f"state-{s}:draw-{i}" for s in range(3) for i in range(128)])


def run_smoke(output, resume=False, seed=1024):
    output = Path(output)
    settings = dict(seed=seed, schema=1, scope="analytical_example")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    with run_lock(output):
        if (output/"config.json").exists():
            if not resume:
                raise FileExistsError("Smoke output exists; use --resume")
            if read_json(output/"config.json") != settings:
                raise ValueError("Smoke resume configuration changed")
            if (output/"summary.json").exists():
                return read_json(output/"summary.json")
        else:
            write_json(output/"config.json", settings)
        started = time.monotonic()
        system = HarmonicSystem()
        source, validation = system.sample(16, 0, seed), system.sample(16, 0, seed+1)
        production, target_production = system.sample(256, 0, seed+2), system.sample(256, 1, seed+3)
        target_fit = system.sample(16, 1, seed+4)
        model = Transport(2, RadialProtection([0., 0.], 5., 10.), spline_count=50,
                          centers=[[0., 0.]], bandwidths=[1.], amplitudes=[.4], seed=seed)
        u0 = lambda z: system.effective_u(z, 0)
        u1 = lambda z: system.effective_u(z, 1)
        checkpoint = output/"transport.pt"
        fit = train_transport(model, source, validation, u0, u1,
                              TrainingConfig(max_steps=2, patience=2), checkpoint,
                              resume=resume and checkpoint.exists(),
                              train_chain_ids=torch.zeros(len(source), dtype=torch.long),
                              validation_chain_ids=torch.ones(len(validation), dtype=torch.long),
                              scale_potential=[u0, u1])
        log_weights, endpoint = model.log_weights(production, u0, u1)
        inverse, _ = model.inverse(endpoint.detach())
        result = estimate(log_weights.detach(), system.regions(production), system.regions(endpoint.detach()), system.name)
        inverse_error = float((production-inverse).abs().max().detach())
        if inverse_error > 1e-9:
            raise AssertionError("Complete finite-map inverse failed")
        fallback = system.sample(128, 0, seed+5)
        fallback_readout = estimate(u0(fallback)-u1(fallback), system.regions(fallback), system.regions(fallback), system.name)
        feat = FEAT(system, FEATConfig(iterations=1, batch_size=8, path_steps=10, seed=seed+6))
        feat_checkpoint = output/"feat.pt"
        feat_fit = feat.fit(source, target_fit, feat_checkpoint, resume=resume and feat_checkpoint.exists())
        feat_result = feat.evaluate(production, target_production,
                                    generator=torch.Generator().manual_seed(seed+7))
        save_checkpoint(output/"feat-paths.pt", feat_result)
        periodic = PeriodicSystem()
        mnf = PeriodicMNF(periodic, MNFConfig(iterations=2, batch_size=8, seed=seed+8))
        mnf_checkpoint = output/"mnf.pt"
        mnf_fit = mnf.fit(periodic.sample(16, 0, seed+9), periodic.sample(16, 1, seed+10),
                          mnf_checkpoint, resume=resume and mnf_checkpoint.exists())
        mnf_result = mnf.evaluate(periodic.sample(256, 0, seed+11), periodic.sample(256, 1, seed+12))
        save_checkpoint(output/"mnf-paths.pt", mnf_result)
        physical = {}
        for name in ("ALDP", "ALA4"):
            molecule = MolecularSystem(name)
            point = molecule.initial_coordinates()
            xyz = molecule.positions(point)
            oracle = molecule.oracle()
            native, force = oracle.evaluate(xyz.detach().numpy(), 1.)
            energy = molecule.energy(xyz, 1.)
            oracle.close()
            if abs(float(energy)-native) > 1e-7 or molecule.chart.constraint_error(xyz) > 1e-10:
                raise AssertionError("Native molecular energy or exact bond constraints failed")
            hmc = HMCSampler(molecule, molecule.state(0), seed+13,
                             step_size=.01, leapfrog_steps=1)
            hmc.step()
            hmc.save(output/f"{name}-hmc.pt")
            continued = HMCSampler.load(molecule, output/f"{name}-hmc.pt")
            hmc.step()
            continued.step()
            if not torch.equal(hmc.z, continued.z):
                raise AssertionError("Physical HMC recovery changed the trajectory")
            states = sorted([molecule.state(0), molecule.state(1)])
            exchange = ReplicaExchangeSampler(molecule, states, seed+14,
                                               step_size=.01, leapfrog_steps=1)
            swapped = exchange.step()
            physical[name] = dict(atoms=len(xyz), shape_dimension=molecule.dim,
                                  energy_error_kjmol=abs(float(energy)-native),
                                  constraint_error_nm=float(molecule.chart.constraint_error(xyz)),
                                  all_L=bool(molecule.in_domain(point)), hmc_resume_exact=True,
                                  actual_cross_energies=swapped["u_kn"].tolist(),
                                  eligible_for_formal=False)
        em, eb = _mbar_control(system, False, seed+15), _mbar_control(system, True, seed+16)
        summary = dict(settings, status="passed", eligible_for_formal=False,
                       elapsed_seconds=time.monotonic()-started,
                       finite_transport=dict(steps=model.n_steps, certificate=model.bound_certificate,
                           training=fit, readout=result, exact_delta_f=system.exact_delta_f(),
                           inverse_error=inverse_error),
                       paid_identity_fallback=fallback_readout,
                       feat=dict(training=feat_fit, nebar=feat_result["native_nebar"], readout=feat_result["readout"]),
                       mnf=dict(training=mnf_fit, bar=mnf_result["native_bar"], readout=mnf_result["readout"],
                                exact_delta_f=periodic.exact_delta_f()),
                       energy_mbar=em, energy_biased_mbar=eb, molecular=physical)
        write_json(output/"summary.json", summary)
        return summary
