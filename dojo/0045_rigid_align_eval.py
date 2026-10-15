"""Sureal-inspired Rigid Point Alignment reference and judge run.

After installing from the repository root, run inside dojo/: python 0045_rigid_align_eval.py
"""

# %%
import torch

# %%
def rigid_align(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (source.ndim != 2 or source.shape != target.shape or source.shape[1] != 3 or len(source) < 3
            or not source.is_floating_point() or source.dtype != target.dtype or source.device != target.device
            or not torch.isfinite(source).all() or not torch.isfinite(target).all()):
        raise ValueError("paired (N, 3) points with N >= 3 required")
    a = source.mean(dim=0)
    b = target.mean(dim=0)
    if torch.linalg.matrix_rank(source - a) < 2 or torch.linalg.matrix_rank(target - b) < 2:
        raise ValueError("point pairs must be noncollinear")
    u, _, vh = torch.linalg.svd((source - a).T @ (target - b))
    correction = torch.eye(3, dtype=source.dtype, device=source.device)
    correction[-1, -1] = torch.sign(torch.det(vh.T @ u.T))
    rotation = vh.T @ correction @ u.T
    return rotation, b - rotation @ a

# %%


# %% [verifiers]
def verify_baseline_known_rotation_and_translation(rigid_align):
    s = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    expected_r = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    expected_t = torch.tensor([2.0, -3.0, 1.0], dtype=torch.float64)
    tgt = s @ expected_r.T + expected_t
    r, t = rigid_align(s, tgt)
    assert torch.allclose(r, expected_r, atol=1e-10)
    assert torch.allclose(t, expected_t, atol=1e-10)


def verify_followup_reflection_must_be_corrected(rigid_align):
    s = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float64)
    tgt = s * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    r, t = rigid_align(s, tgt)
    assert torch.det(r) > 0.999999 and torch.allclose(r.T @ r, torch.eye(3, dtype=s.dtype), atol=1e-10)


def verify_followup_invalid_pair_size(rigid_align):
    try:
        rigid_align(torch.zeros(2, 3), torch.zeros(2, 3))
    except ValueError:
        pass
    else:
        raise AssertionError('two points must fail')


def verify_production_degenerate_geometry(rigid_align):
    line = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], dtype=torch.float64)
    for source, target in ((line, line + 1), (torch.full_like(line, 2.), torch.full_like(line, 3.))):
        try:
            rigid_align(source, target)
        except ValueError:
            pass
        else:
            raise AssertionError("degenerate geometry accepted")
    good = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    for bad in (good.double(), torch.full_like(good, float('nan'))):
        try:
            rigid_align(good, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched or nonfinite pairs accepted")


def verify_production_noisy_fit_and_no_mutation(rigid_align):
    source = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [1., 2., 3.]], dtype=torch.float64)
    target = source + torch.tensor([2., -3., 1.], dtype=source.dtype)
    target[4, 0] += .01
    original = source.clone(), target.clone()
    r, t = rigid_align(source, target)
    fitted_error = torch.sum((source @ r.T + t - target) ** 2)
    identity_error = torch.sum((source - target) ** 2)
    assert fitted_error < identity_error * .001
    assert torch.allclose(r.T @ r, torch.eye(3, dtype=source.dtype), atol=1e-10)
    assert torch.det(r) > 0.999999
    assert torch.equal(source, original[0]) and torch.equal(target, original[1])


# %%
print("Rigid Point Alignment: challenge the judge!")
from torch_judge import evaluate
if not evaluate('0045_rigid_align_eval.py', rigid_align):
    raise SystemExit(1)
