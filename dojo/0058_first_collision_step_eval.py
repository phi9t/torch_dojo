"""Circular-footprint sampled collision reference, inspired by Sureal Motion."""

# %%
import math
import torch


# %%
def first_collision_step(ego: torch.Tensor, others: torch.Tensor, ego_radius: float, other_radii: torch.Tensor) -> int:
    if (ego.ndim != 2 or ego.shape[1] != 2 or others.ndim != 3
            or others.shape[1:] != ego.shape or other_radii.shape != (others.shape[0],)
            or not ego.is_floating_point() or others.dtype != ego.dtype
            or other_radii.dtype != ego.dtype or others.device != ego.device
            or other_radii.device != ego.device or not math.isfinite(ego_radius)
            or ego_radius < 0 or not torch.isfinite(ego).all()
            or not torch.isfinite(others).all() or not torch.isfinite(other_radii).all()
            or torch.any(other_radii < 0)):
        raise ValueError("need finite, synchronized paths and nonnegative radii")
    if ego.shape[0] == 0 or others.shape[0] == 0:
        return -1
    distances = torch.linalg.vector_norm(others - ego[None], dim=-1)
    hit = distances <= ego_radius + other_radii[:, None]
    steps = torch.nonzero(hit.any(dim=0)).flatten()
    return int(steps[0].item()) if steps.numel() else -1


# %% [verifiers]
def _rejects(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("invalid collision contract must raise ValueError")


def verify_baseline_first_sampled_contact(first_collision_step):
    e = torch.tensor([[0., 0.], [1., 0.], [2., 0.]])
    o = torch.tensor([[[5., 0.], [4., 0.], [2.9, 0.]], [[5., 0.], [1.9, 0.], [5., 0.]]])
    assert first_collision_step(e, o, .5, torch.tensor([.5, .5])) == 1


def verify_followup_synchrony_and_actor_radii(first_collision_step):
    e = torch.tensor([[0., 0.], [1., 0.], [2., 0.]])
    o = torch.tensor([[[2., 0.], [0., 2.], [0., 0.]]])
    assert first_collision_step(e, o, .4, torch.tensor([.4])) == -1
    e = torch.zeros(2, 2)
    o = torch.tensor([[[3., 0.], [3., 0.]], [[2., 0.], [1.5, 0.]]])
    assert first_collision_step(e, o, 1., torch.tensor([.5, 1.])) == 0


def verify_production_exact_tangency_empty_and_invalid(first_collision_step):
    e = torch.zeros(1, 2, dtype=torch.float64)
    o = torch.tensor([[[1.5, 0.]]], dtype=e.dtype)
    radii = torch.tensor([1.], dtype=e.dtype)
    old = (e.clone(), o.clone(), radii.clone())
    assert first_collision_step(e, o, .5, radii) == 0
    assert all(torch.equal(a, b) for a, b in zip((e, o, radii), old))
    assert first_collision_step(e, o + .001, .5, torch.tensor([1.], dtype=e.dtype)) == -1
    assert first_collision_step(e, o[:0], .5, torch.empty(0, dtype=e.dtype)) == -1
    assert first_collision_step(e[:0], o[:, :0], .5, torch.tensor([1.], dtype=e.dtype)) == -1
    _rejects(first_collision_step, e, o, -.1, torch.tensor([1.], dtype=e.dtype))
    _rejects(first_collision_step, e, o, .5, torch.tensor([float('nan')], dtype=e.dtype))
    _rejects(first_collision_step, e, o.float(), .5, torch.tensor([1.], dtype=e.dtype))
    _rejects(first_collision_step, e, o, float('inf'), torch.tensor([1.], dtype=e.dtype))


# %%
print("First Collision Step: challenge the judge!")
from torch_judge import evaluate
if not evaluate("0058_first_collision_step_eval.py", first_collision_step):
    raise SystemExit(1)
