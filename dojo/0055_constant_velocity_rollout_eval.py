"""Constant-velocity rollout reference, adapted from Sureal's dynamic.py."""

# %%
import torch


# %%
def constant_velocity_rollout(history: torch.Tensor, times: torch.Tensor, future_times: torch.Tensor) -> torch.Tensor:
    if (history.ndim != 2 or history.shape[1] != 2 or history.shape[0] < 2
            or times.shape != (history.shape[0],) or future_times.ndim != 1
            or future_times.numel() == 0 or not history.is_floating_point()
            or times.dtype != history.dtype or future_times.dtype != history.dtype
            or times.device != history.device or future_times.device != history.device
            or not torch.isfinite(history).all() or not torch.isfinite(times).all()
            or not torch.isfinite(future_times).all()
            or not torch.all(torch.diff(times) > 0)
            or not torch.all(future_times > times[-1])
            or not torch.all(torch.diff(future_times) > 0)):
        raise ValueError("need finite, matching tensors and strictly increasing times")
    velocity = (history[-1] - history[-2]) / (times[-1] - times[-2])
    return history[-1] + (future_times - times[-1])[:, None] * velocity


# %% [verifiers]
def _rejects(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("invalid trajectory contract must raise ValueError")


def verify_baseline_elapsed_seconds(constant_velocity_rollout):
    h = torch.tensor([[0., 0.], [2., -1.]], dtype=torch.float64)
    out = constant_velocity_rollout(h, torch.tensor([1., 1.5], dtype=h.dtype), torch.tensor([2.], dtype=h.dtype))
    assert out.shape == (1, 2) and out.dtype == h.dtype
    assert torch.allclose(out, torch.tensor([[4., -2.]], dtype=h.dtype))


def verify_followup_irregular_observations_and_queries(constant_velocity_rollout):
    h = torch.tensor([[99., 99.], [1., 2.], [3., 2.]], dtype=torch.float64)
    t = torch.tensor([0., 1., 3.], dtype=h.dtype)
    out = constant_velocity_rollout(h, t, torch.tensor([3.5, 5.], dtype=h.dtype))
    assert torch.allclose(out, torch.tensor([[3.5, 2.], [5., 2.]], dtype=h.dtype))


def verify_followup_timeline_order(constant_velocity_rollout):
    h = torch.zeros(2, 2)
    for t, future in [([1., 1.], [2.]), ([1., 0.], [2.]), ([0., 1.], [1.]), ([0., 1.], [3., 2.])]:
        _rejects(constant_velocity_rollout, h, torch.tensor(t), torch.tensor(future))


def verify_production_finite_shapes_and_no_mutation(constant_velocity_rollout):
    h = torch.tensor([[0., 0.], [1., -2.]], dtype=torch.float64)
    t = torch.tensor([0., 2.], dtype=h.dtype)
    f = torch.tensor([3., 7.], dtype=h.dtype)
    before = (h.clone(), t.clone(), f.clone())
    out = constant_velocity_rollout(h, t, f)
    assert torch.allclose(out, torch.tensor([[1.5, -3.], [3.5, -7.]], dtype=h.dtype))
    assert all(torch.equal(a, b) for a, b in zip((h, t, f), before))
    _rejects(constant_velocity_rollout, h, t, torch.empty(0, dtype=h.dtype))
    _rejects(constant_velocity_rollout, h, t, torch.tensor([float('nan')], dtype=h.dtype))
    _rejects(constant_velocity_rollout, h[:1], t[:1], f)
    _rejects(constant_velocity_rollout, h, t.float(), f)


# %%
print("Constant Velocity Rollout: challenge the judge!")
from torch_judge import evaluate
if not evaluate("0055_constant_velocity_rollout_eval.py", constant_velocity_rollout):
    raise SystemExit(1)
