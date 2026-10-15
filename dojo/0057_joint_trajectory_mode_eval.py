"""Shared trajectory-mode reference, adapted from Sureal's joint Motion fixture."""

# %%
import torch


# %%
def joint_trajectory_mode(predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> tuple[int, torch.Tensor]:
    if (predictions.ndim != 4 or predictions.shape[-1] != 2 or predictions.shape[0] == 0
            or targets.shape != predictions.shape[1:] or targets.shape[0] == 0
            or targets.shape[1] == 0 or valid.shape != targets.shape[:2]
            or valid.dtype != torch.bool or not predictions.is_floating_point()
            or targets.dtype != predictions.dtype or targets.device != predictions.device
            or valid.device != predictions.device or not torch.isfinite(predictions).all()
            or not torch.isfinite(targets).all() or not torch.all(valid.any(dim=1))):
        raise ValueError("need finite trajectories and at least one valid step per actor")
    distances = torch.linalg.vector_norm(predictions - targets[None], dim=-1)
    per_actor = distances.masked_fill(~valid[None], 0).sum(dim=-1) / valid.sum(dim=-1)[None]
    scores = per_actor.mean(dim=-1)
    index = int(torch.argmin(scores).item())
    return index, scores[index]


# %% [verifiers]
def _rejects(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("invalid joint metric contract must raise ValueError")


def verify_baseline_one_shared_mode(joint_trajectory_mode):
    p = torch.tensor([[[[0., 0.]], [[4., 0.]]], [[[1., 0.]], [[1., 0.]]]])
    index, score = joint_trajectory_mode(p, torch.zeros(2, 1, 2), torch.ones(2, 1, dtype=torch.bool))
    assert index == 1 and score.ndim == 0 and torch.allclose(score, torch.tensor(1.))


def verify_followup_equal_actor_weight_and_nonprefix_masks(joint_trajectory_mode):
    p = torch.tensor([[[[0., 0.], [100., 0.], [0., 0.]], [[3., 0.], [3., 0.], [3., 0.]]],
                      [[[4., 0.], [0., 0.], [4., 0.]], [[0., 0.], [0., 0.], [0., 0.]]]], dtype=torch.float64)
    y = torch.zeros(2, 3, 2, dtype=p.dtype)
    index, score = joint_trajectory_mode(p, y, torch.tensor([[True, False, True], [True, True, True]]))
    assert index == 0 and torch.allclose(score, torch.tensor(1.5, dtype=p.dtype))


def verify_followup_first_mode_wins_tie(joint_trajectory_mode):
    p = torch.ones(3, 1, 1, 2)
    index, score = joint_trajectory_mode(p, torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool))
    assert index == 0 and torch.allclose(score, torch.tensor(2. ** .5))


def verify_production_all_invalid_actor_and_nonfinite_policy(joint_trajectory_mode):
    p = torch.zeros(2, 2, 1, 2)
    y = torch.zeros(2, 1, 2)
    _rejects(joint_trajectory_mode, p, y, torch.tensor([[True], [False]]))
    _rejects(joint_trajectory_mode, p, y, torch.zeros(2, 1, dtype=torch.bool))
    _rejects(joint_trajectory_mode, p, y.double(), torch.ones(2, 1, dtype=torch.bool))
    bad = p.clone(); bad[0, 0, 0, 0] = float('nan')
    _rejects(joint_trajectory_mode, bad, y, torch.ones(2, 1, dtype=torch.bool))
    _rejects(joint_trajectory_mode, p[:, :0], y[:0], torch.zeros(0, 1, dtype=torch.bool))


# %%
print("Joint Trajectory Mode: challenge the judge!")
from torch_judge import evaluate
if not evaluate("0057_joint_trajectory_mode_eval.py", joint_trajectory_mode):
    raise SystemExit(1)
