import numpy as np
import pytest
import torch

from varipath.mechanism import (SourceBatch, prepare_panel, run_interventions,
                                 run_predictions, sources_from_bank)
from varipath.interventions import GeometryQualification
from varipath.protection import RadialProtection


class IdentityBase:
    n_steps = 6

    def single_step(self, z, k):
        return z

    def single_inverse_step(self, z, k):
        return z

    def forward(self, z):
        return z, z.new_zeros(len(z))


class DoubleWell:
    name = "ALDP"
    dim = 1

    def in_domain(self, z):
        return torch.isfinite(z).all(-1)

    def effective_u(self, z, endpoint):
        value = (z[..., 0].square() - 1).square()
        return value - .05*torch.tanh(z[..., 0]) if endpoint else value

    def regions(self, z):
        return {"a": z[..., 0] > 0, "b": z[..., 0] <= 0,
                "remainder": torch.zeros(z.shape[:-1], dtype=torch.bool),
                "valid": torch.isfinite(z).all(-1)}

    def mechanism_boundary_levels(self, z):
        return -z[:, :1]

    def state(self, endpoint):
        return (550., 300.)[endpoint]


def panel():
    torch.set_num_threads(1)
    system = DoubleWell()
    guard = RadialProtection(torch.zeros(1), 8., 10.)
    anchor = {"status": "certified", "certificate": {"certified": True},
              "z": [0.], "sigma": .5, "amplitude": .5,
              "surface": 1., "start": 0, "minima": [{"a": False}, {"a": True}]}
    source = torch.linspace(-2, 2, 17, dtype=torch.float64)[:, None]
    return prepare_panel(IdentityBase(), system, guard, [anchor], source)


def source(role):
    values = torch.tensor([[-1.5], [-.5], [.3], [1.2]], dtype=torch.float64).repeat(4, 1)
    units = [role+f"-ladder-{i}" for i in range(4) for _ in range(4)]
    blocks = [i//2 for i in range(4) for _ in range(4)]
    return SourceBatch(values, units, blocks, role)


def test_builder_tracks_actual_boundary_and_preserves_missing_anchor_slots():
    built = panel()
    assert len(built.cells) == 48
    ready = [cell for cell in built.cells if cell["status"] == "ready"]
    assert len(ready) == 12
    assert len({cell["epsilon"] for cell in ready}) == 1
    for cell in ready:
        assert cell["boundary_record"]["endpoint_level_residual"] < 1e-12
        active = cell["eta"] > 0
        displacement = ((cell["boundary"] - cell["anchor_center"])*cell["normals"]).sum(-1)
        assert torch.allclose(displacement[active], displacement.new_full((int(active.sum()),), cell["offset"]*cell["sigma"]))
        assert cell["dose_diagnostics"]["p95_maximum_path_displacement"] <= .1*cell["sigma"]
        assert cell["continuous_tube_certificates"]
        assert not cell["geometry"].qualified
    assert built.development["primary_pair"] is not None
    failed = [cell for cell in built.cells if cell["status"] == "construction_failed"]
    assert len(failed) == 36


def test_no_real_boundary_or_no_continuous_tube_keeps_failures():
    built = panel()
    anchor = {"status": "certified", "certificate": {"certified": True},
              "z": [0.], "sigma": .5, "amplitude": .5,
              "minima": [{"a": False}, {"a": True}]}
    positive = torch.linspace(.1, 2, 17, dtype=torch.float64)[:, None]
    no_boundary = prepare_panel(IdentityBase(), DoubleWell(), built.protection, [anchor], positive)
    assert all(cell["status"] == "construction_failed" for cell in no_boundary.cells)
    assert any("没有找到" in cell["reason"] for cell in no_boundary.cells)
    tiny_guard = RadialProtection(torch.zeros(1), .1, .2)
    no_tube = prepare_panel(IdentityBase(), DoubleWell(), tiny_guard, [anchor], torch.linspace(-2, 2, 17)[:, None])
    assert all(cell["status"] == "construction_failed" for cell in no_tube.cells)


def test_predictions_and_independent_full_half_run_keep_full_96_418_manifest(tmp_path):
    built = panel()
    for cell in built.cells:
        if cell["status"] == "ready":
            cell["geometry"] = GeometryQualification(True, 0., 0., cell["sigma"], True,
                certified=True, reason="此测试的 -x=0 是全局唯一线性一维分界，法向和曲率严格为常数")
    prediction = run_predictions(built, source("P"), tmp_path, {"numerical_error_bound": 0.})
    assert prediction["sealed"] and prediction["prediction_events"] == 552
    assert len(prediction["manifest"]) == 96
    assert (tmp_path/"predictions.pt").is_file()
    intervention = run_interventions(prediction, source("I"), tmp_path)
    assert intervention["sealed"] and intervention["family_events"] == 418
    assert intervention["matrix"].shape == (2, 418)
    assert len(intervention["manifest"]) == 96
    assert sum(cell["status"] == "computed" for cell in intervention["manifest"]) == 12
    assert np.isnan(intervention["matrix"]).any()
    assert intervention["agreement"]["main"]["status"] != "supported"
    assert not intervention["eligible_for_formal"]
    assert intervention["intervals"]["ALDP/geometric_squared_error_difference"].finite
    resumed = run_interventions(prediction, source("I"), tmp_path, resume=True)
    assert resumed["sealed"]
    with pytest.raises(FileExistsError):
        run_predictions(built, source("P"), tmp_path)


def test_source_blocks_group_whole_ladder_pairs_and_aldp_source_endpoint():
    positions = torch.zeros(4, 2, 2, 1, dtype=torch.float64)
    positions[:, :, 1] = 7.
    bank = {"positions": positions, "states": [300., 550.],
            "source_unit_ids": [f"P-unit-{i}" for i in range(4)], "role": "P"}
    batch = sources_from_bank(DoubleWell(), bank, inference_blocks=2)
    assert torch.all(batch.samples == 7.)
    assert batch.block_ids == [0, 0, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="拆成多个"):
        SourceBatch(torch.zeros(2, 1), ["same-ladder"]*2, [0, 1], "P")
