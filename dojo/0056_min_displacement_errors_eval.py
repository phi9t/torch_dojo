"""Multimodal displacement-error reference, adapted from Sureal Motion fixtures."""

# %%
import torch


# %%
def min_displacement_errors(predictions: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (predictions.ndim != 3 or predictions.shape[-1] != 2 or predictions.shape[0] == 0
            or target.shape != predictions.shape[1:] or valid.shape != target.shape[:1]
            or target.shape[0] == 0 or not predictions.is_floating_point()
            or target.dtype != predictions.dtype or valid.dtype != torch.bool
            or predictions.device != target.device or valid.device != target.device
            or not torch.isfinite(predictions).all() or not torch.isfinite(target).all()
            or not torch.any(valid)):
        raise ValueError("need finite, matching trajectories and a valid target step")
    distances = torch.linalg.vector_norm(predictions - target[None], dim=-1)
    ade = distances[:, valid].mean(dim=1).min()
    final = torch.nonzero(valid)[-1, 0]
    fde = distances[:, final].min()
    return ade, fde


# %% [verifiers]
def _rejects(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("invalid metric contract must raise ValueError")


def verify_baseline_independent_best_modes(min_displacement_errors):
    p = torch.tensor([[[0., 0.], [0., 0.], [3., 0.]], [[2., 0.], [2., 0.], [0., 0.]]])
    ade, fde = min_displacement_errors(p, torch.zeros(3, 2), torch.ones(3, dtype=torch.bool))
    assert ade.ndim == fde.ndim == 0
    assert torch.allclose(ade, torch.tensor(1.)) and torch.allclose(fde, torch.tensor(0.))


def verify_followup_non_prefix_mask_and_last_valid(min_displacement_errors):
    p = torch.tensor([[[1., 0.], [100., 0.], [2., 0.], [100., 0.]]], dtype=torch.float64)
    y = torch.zeros(4, 2, dtype=p.dtype)
    ade, fde = min_displacement_errors(p, y, torch.tensor([True, False, True, False]))
    assert torch.allclose(ade, torch.tensor(1.5, dtype=p.dtype))
    assert torch.allclose(fde, torch.tensor(2., dtype=p.dtype))


def verify_production_all_invalid_and_nonfinite_policy(min_displacement_errors):
    p = torch.zeros(1, 2, 2)
    y = torch.zeros(2, 2)
    _rejects(min_displacement_errors, p, y, torch.zeros(2, dtype=torch.bool))
    _rejects(min_displacement_errors, p, y, torch.ones(2))
    _rejects(min_displacement_errors, p, y.double(), torch.ones(2, dtype=torch.bool))
    bad = p.clone(); bad[0, 0, 0] = float('inf')
    _rejects(min_displacement_errors, bad, y, torch.ones(2, dtype=torch.bool))
    _rejects(min_displacement_errors, p[:, :0], y[:0], torch.zeros(0, dtype=torch.bool))


def verify_production_inputs_remain_intact(min_displacement_errors):
    p = torch.tensor([[[3., 4.], [5., 12.]]])
    y = torch.zeros(2, 2)
    v = torch.tensor([True, False])
    old = (p.clone(), y.clone(), v.clone())
    ade, fde = min_displacement_errors(p, y, v)
    assert ade.dtype == fde.dtype == p.dtype
    assert torch.allclose(ade, torch.tensor(5.)) and torch.allclose(fde, torch.tensor(5.))
    assert all(torch.equal(a, b) for a, b in zip((p, y, v), old))


# %%
print("Minimum Displacement Errors: challenge the judge!")
from torch_judge import evaluate
if not evaluate("0056_min_displacement_errors_eval.py", min_displacement_errors):
    raise SystemExit(1)
