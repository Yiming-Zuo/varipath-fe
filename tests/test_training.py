import pytest
import torch

from varipath.protection import RadialProtection
from varipath.transport import Transport
from varipath.training import TrainingConfig, sampled_gradient_scale, train_transport


torch.set_num_threads(1)


def setup():
    g = torch.Generator().manual_seed(987)
    train = torch.randn(6, 1, generator=g, dtype=torch.float64)
    validation = torch.randn(6, 1, generator=g, dtype=torch.float64)
    model = Transport(1, RadialProtection([0], 4, 12), spline_count=4)
    u0 = lambda z: z.square().sum() / 2
    u1 = lambda z: (z - 0.2).square().sum() / 2
    return model, train, validation, u0, u1


def test_full_pool_training_reduces_true_objective_and_restores_checkpoint(tmp_path):
    model, train, validation, u0, u1 = setup()
    path = tmp_path / "checkpoint.pt"
    report = train_transport(model, train, validation, u0, u1,
                             TrainingConfig(max_steps=1), path)
    assert report["accepted_steps"] == 1
    assert report["last_training_objective"] < report["initial_objective"]
    fresh, *_ = setup()
    resumed = train_transport(fresh, train, validation, u0, u1,
                              TrainingConfig(max_steps=2), path, resume=True)
    uninterrupted, *_ = setup()
    direct = train_transport(uninterrupted, train, validation, u0, u1,
                             TrainingConfig(max_steps=2))
    assert resumed["history"] == direct["history"]
    for a, b in zip(fresh.parameters(), uninterrupted.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        train_transport(fresh, train, validation, u0, u1, checkpoint=path)
    with pytest.raises(ValueError, match="池发生变化"):
        train_transport(fresh, train + 0.01, validation, u0, u1, checkpoint=path, resume=True)


def test_scale_uses_chain_balanced_actual_gradients():
    points = torch.tensor([[0.0], [0.4], [0.9], [1.2]], dtype=torch.float64)
    assert sampled_gradient_scale(points, lambda z: 3 * z.square().sum(), [0, 0, 1, 1]) == pytest.approx(6)


def test_independence_and_deadline_failures(tmp_path):
    model, train, validation, u0, u1 = setup()
    with pytest.raises(ValueError, match="角色相交"):
        train_transport(model, train, validation, u0, u1,
                        train_chain_ids=[0] * len(train), validation_chain_ids=[0] * len(validation))
    with pytest.raises(TimeoutError, match="预算"):
        train_transport(model, train, validation, u0, u1,
                        TrainingConfig(deadline_monotonic=0), checkpoint=tmp_path / "stop.pt")


def test_timeout_during_line_search_resumes_last_accepted_iteration(tmp_path):
    model, train, validation, u0, u1 = setup()
    checkpoint = tmp_path / "interrupted.pt"
    def interrupted():

        return bool(model.C.detach().abs().max()>0)

    with pytest.raises(TimeoutError):
        train_transport(model, train, validation, u0, u1,
                        TrainingConfig(max_steps=2), checkpoint, should_stop=interrupted)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["accepted"] == 0
    restored, *_ = setup()
    resumed = train_transport(restored, train, validation, u0, u1,
                              TrainingConfig(max_steps=2), checkpoint, resume=True)
    direct, *_ = setup()
    expected = train_transport(direct, train, validation, u0, u1,
                               TrainingConfig(max_steps=2))
    assert resumed["history"] == expected["history"]
