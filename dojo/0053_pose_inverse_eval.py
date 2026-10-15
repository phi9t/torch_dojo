"""Rigid pose inverse reference and cumulative checks."""

# %%
import torch

# %%
def pose_inverse(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape != (4, 4) or pose.dtype not in (torch.float32, torch.float64) or not torch.isfinite(pose).all():
        raise ValueError('finite floating [4,4] pose required')
    R = pose[:3, :3]
    I = torch.eye(3, dtype=pose.dtype, device=pose.device)
    last = pose.new_tensor([0., 0., 0., 1.])
    tol = 1e-5 if pose.dtype == torch.float32 else 1e-8
    if (not torch.allclose(R.T @ R, I, atol=tol, rtol=0)
            or not torch.allclose(torch.det(R), pose.new_tensor(1.), atol=tol, rtol=0)
            or not torch.allclose(pose[3], last, atol=tol, rtol=0)):
        raise ValueError('pose must be a proper rigid transform')
    inverse = torch.eye(4, dtype=pose.dtype, device=pose.device)
    inverse[:3, :3] = R.T
    inverse[:3, 3] = -(R.T @ pose[:3, 3])
    return inverse

# %% [verifiers]
def verify_baseline_rotated_translation(pose_inverse):
    T = torch.tensor([[0., -1., 0., 2.], [1., 0., 0., 3.],
                      [0., 0., 1., 4.], [0., 0., 0., 1.]], dtype=torch.float64)
    expected = torch.tensor([[0., 1., 0., -3.], [-1., 0., 0., 2.],
                             [0., 0., 1., -4.], [0., 0., 0., 1.]], dtype=T.dtype)
    assert torch.allclose(pose_inverse(T), expected, atol=1e-12)


def verify_followup_both_round_trips_and_dtype(pose_inverse):
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    T[:3, 3] = torch.tensor([2., 3., 4.])
    inverse = pose_inverse(T)
    assert inverse.dtype == T.dtype and inverse.device == T.device
    assert torch.allclose(inverse @ T, torch.eye(4))
    assert torch.allclose(T @ inverse, torch.eye(4))


def verify_production_reject_invalid_poses(pose_inverse):
    cases = [torch.eye(3), torch.diag(torch.tensor([2., 1., 1., 1.])),
             torch.diag(torch.tensor([-1., 1., 1., 1.])),
             torch.diag(torch.tensor([1., 1., 1., 2.])),
             torch.full((4, 4), float('nan'))]
    for T in cases:
        try:
            pose_inverse(T)
        except ValueError:
            continue
        raise AssertionError('invalid pose accepted')


def verify_production_strides_no_mutation_and_gradient(pose_inverse):
    base = torch.eye(4, dtype=torch.float64, requires_grad=False).T
    T = base.clone().T
    T[:3, 3] = torch.tensor([2., -3., 4.], dtype=T.dtype)
    T.requires_grad_()
    before = T.detach().clone()
    inverse = pose_inverse(T)
    assert torch.equal(T, before) and inverse.dtype == T.dtype
    inverse[:3, 3].sum().backward()
    assert T.grad is not None and torch.allclose(T.grad[:3, 3], -torch.ones(3, dtype=T.dtype))


# %%
from torch_judge import evaluate
if not evaluate('0053_pose_inverse_eval.py', pose_inverse):
    raise SystemExit(1)
