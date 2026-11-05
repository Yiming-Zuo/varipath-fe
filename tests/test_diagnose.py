import pytest
import torch

from varipath.diagnose import run_diagnostics
from varipath.geometry import MolecularProtection
from varipath.physics import AffineSystem, MolecularSystem
from varipath.readouts import estimate
from varipath.records import save_checkpoint, write_json
from varipath.sampling import HMCSampler
from varipath.transport import Transport
from varipath.workflow import prepare


def test_e2_reads_actual_frozen_identity_and_refuses_unrevealed_reference(tmp_path):
    config = dict(molecule="ALDP", mode="development", seed=1024, budget_seconds=120,
        sampling=dict(count=2, warmup=0, ladders=2, states=[300., 550.], leapfrog_steps=1, checkpoint_every=1))
    preparation = tmp_path/"prepared"
    prepare(config, preparation)
    fixed = torch.load(preparation/"prepared.pt", weights_only=False)
    system = AffineSystem(MolecularSystem("ALDP"), fixed["location"], fixed["scale"])
    guard = MolecularProtection(system, **fixed["protection"])
    model = Transport(48, guard)
    source = HMCSampler(system, 550., 885, leapfrog_steps=1).sample(4)


    loga = system.effective_u(source, 0)-system.effective_u(source, 1)
    regions = system.regions(source)
    native = estimate(loga, regions, regions, "ALDP")
    b = guard(source)
    fractions = [float((b == 0).double().mean()), float(((b > 0)&(b < 1)).double().mean()), float((b == 1).double().mean())]
    native.update(anchor_exposure=[], protection_fractions=fractions, frozen_strengths=[],
                  proposal_populations={key:float(regions[key].double().mean()) for key in ("a", "b", "remainder")})
    deployment = tmp_path/"identity"
    save_checkpoint(deployment/"model.pt", dict(state_dict=model.state_dict(), steps=model.n_steps,
        centers=model.centers, bandwidths=model.bandwidths, amplitudes=model.amplitudes, time_centers=None))
    save_checkpoint(deployment/"production-output.pt", dict(source=source, target=source, log_weights=loga))
    write_json(deployment/"result.json", dict(status="completed", method="V0", native=native))
    write_json(deployment/"run-config.json", dict(config=config))
    result = run_diagnostics(preparation, deployment, tmp_path/"diagnostic.json")
    assert result["status"] == "diagnostic" and not result["independent_validation"]
    assert result["main_partition"]["identity_residual"] < 1e-12
    with pytest.raises(ValueError, match="explicit reveal"):
        run_diagnostics(preparation, deployment, tmp_path/"unrevealed.json", reference=tmp_path/"never-read-R.pt")
    saved = torch.load(deployment/"model.pt", weights_only=False)
    saved["state_dict"]["protection.system.location"] = system.location+1
    save_checkpoint(deployment/"model.pt", saved)
    with pytest.raises(ValueError, match="differs from preparation"):
        run_diagnostics(preparation, deployment, tmp_path/"wrong-preparation.json")
