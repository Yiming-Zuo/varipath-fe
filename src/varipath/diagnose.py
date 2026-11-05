import argparse
from pathlib import Path

import numpy as np
import torch

from .geometry import MolecularProtection
from .physics import AffineSystem, MolecularSystem
from .physical_analysis import (energy_shape_diagnostic, freeze_energy_bins,
    frozen_cross_diagnostic, pii_beta_preference, polar_closure, risk_partition)
from .catalog import physical_potential
from .records import read_json, write_json
from .readouts import mismatch
from .transport import Transport


def _load(preparation, deployment):
    prepared = torch.load(Path(preparation)/"prepared.pt", weights_only=False)
    system = AffineSystem(MolecularSystem(prepared["molecule"]), prepared["location"], prepared["scale"])
    protection = MolecularProtection(system, **prepared["protection"])
    saved = torch.load(Path(deployment)/"model.pt", weights_only=False)
    time_centers = saved["time_centers"]
    if saved.get("center_provider") is not None:
        from .centers import DiskCenters
        time_centers = DiskCenters.from_descriptor(saved["center_provider"], deployment,
            expected_source_state=saved["center_provider_source_state"])
    count = saved["state_dict"]["C"].shape[0]
    model = Transport(system.dim, protection, spline_count=count, steps=saved["steps"],
        centers=saved["centers"], bandwidths=saved["bandwidths"], amplitudes=saved["amplitudes"],
        time_centers=time_centers)
    current = model.state_dict()
    for key, value in saved["state_dict"].items():
        if key.startswith("protection.") and (key not in current or not torch.equal(value, current[key])):
            raise ValueError("The frozen physical system, affine chart or protection differs from preparation")
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return system, model


def run_diagnostics(preparation, deployment, output, *, v0=None, reference=None, registry=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Diagnostic output already exists")
    result = read_json(Path(deployment)/"result.json")
    if result["status"] != "completed" or result["method"] not in ("V", "V0", "G", "MG"):
        raise ValueError("Frozen transport diagnostics require a completed native transport")
    system, model = _load(preparation, deployment)
    production = torch.load(Path(deployment)/"production-output.pt", weights_only=False)
    source, target, log_weights = production["source"], production["target"], production["log_weights"]
    regions = system.regions(target)
    labels = torch.where(regions["a"], 0, torch.where(regions["b"], 1, 2))
    native = result["native"]
    report = dict(status="diagnostic", molecule=system.name,
        method=result["method"], replica=result.get("replica"),
        data_unit_ids=result.get("data_unit_ids", {}),
        eligible_for_formal=result.get("eligible_for_formal", False),
        independent_validation=False, reused_production=str(Path(deployment).resolve()),
        main_partition=risk_partition(log_weights, labels, 3),
        fine_partition=risk_partition(log_weights, regions["fine"], 25 if system.name == "ALA4" else 5),
        anchor_exposure=native["anchor_exposure"], protection_fractions=native["protection_fractions"],
        strengths=native["frozen_strengths"])
    if system.name == "ALA4":
        bank = torch.load(Path(preparation)/"fit/bank.pt", weights_only=False)
        state_index = bank["states"].index(0.)
        d_source = (bank["positions"][:, :, state_index].flatten(0, 1)-system.location)/system.scale
        with torch.no_grad():
            polar = torch.stack([physical_potential(system, z, 1)-physical_potential(system, z, 0) for z in d_source])
        edges = freeze_energy_bins(polar.numpy())
        report["energy_partition"] = energy_shape_diagnostic(system, target, log_weights, edges)
        fine = native["fine_populations"]
        report["pii_beta"] = pii_beta_preference(fine["0"], fine["1"])
    if v0 is not None:
        other_system, other_model = _load(preparation, v0)
        if other_system.name != system.name:
            raise ValueError("Cross diagnostic molecular tasks differ")
        report["frozen_2x2"] = frozen_cross_diagnostic(model, other_model, source,
            lambda z: system.effective_u(z, 0), lambda z: system.effective_u(z, 1))
    if reference is not None:
        from .inference import RoleSeal
        seal = RoleSeal(registry) if registry is not None else None
        roles = seal.state["roles"] if seal is not None else {}
        if "R" not in roles or any(roles.get(r, {}).get("status") != "sealed" for r in ("P", "I", "F")):
            raise ValueError("Read the independent reference only after its explicit reveal following P/I/F seals")
        rbank = torch.load(reference, weights_only=False)
        if not set(rbank.get("source_unit_ids", [])).issubset(set(roles["R"]["unit_ids"])):
            raise ValueError("The reference bank is not in the revealed independent R set")
        rbank = dict(rbank)
        rbank["positions"] = (rbank["positions"]-system.location)/system.scale
        from .workflow import _window_readout
        reference_readout = _window_readout(system, rbank, read_json(Path(deployment)/"run-config.json")["config"], lambda: None)
        target_populations = {key: reference_readout["endpoint_populations"]["1_"+key] for key in ("a", "b")}
        target_populations["remainder"] = 1-sum(target_populations.values())
        report["proposal_mismatch"] = mismatch(native["proposal_populations"], target_populations)
        if system.name == "ALA4":
            report["polar_closure"] = polar_closure(system, rbank)
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute diagnostics from frozen transport outputs.")
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v0", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args(argv)
    run_diagnostics(args.preparation, args.deployment, args.output,
                    v0=args.v0, reference=args.reference, registry=args.registry)


if __name__ == "__main__":
    main()
