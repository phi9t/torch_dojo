"""LiDAR polar conversion reference and cumulative checks."""

# %%
import math
import torch

# %%
def polar_to_cartesian(polar: torch.Tensor) -> torch.Tensor:
    if (polar.ndim < 1 or polar.shape[-1] != 3
            or polar.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(polar).all() or torch.any(polar[..., 0] < 0)):
        raise ValueError('polar must have finite nonnegative range and shape (...,3)')
    radius, azimuth, elevation = polar.unbind(dim=-1)
    horizontal = radius * torch.cos(elevation)
    return torch.stack((horizontal * torch.cos(azimuth),
                        horizontal * torch.sin(azimuth), radius * torch.sin(elevation)), dim=-1)

# %% [verifiers]
def verify_baseline_three_axes(polar_to_cartesian):
    p = torch.tensor([[2., 0., 0.], [2., math.pi / 2, 0.],
                      [2., 0., math.pi / 2]], dtype=torch.float64)
    out = polar_to_cartesian(p)
    assert torch.allclose(out, 2 * torch.eye(3, dtype=p.dtype), atol=1e-12)


def verify_followup_batch_negative_elevation_and_zero_range(polar_to_cartesian):
    p = torch.tensor([[[2., math.pi / 4, -math.pi / 4], [0., 1., -.2]]], dtype=torch.float64)
    out = polar_to_cartesian(p)
    expected = torch.tensor([[[1., 1., -math.sqrt(2.)], [0., 0., 0.]]], dtype=p.dtype)
    assert out.shape == p.shape and torch.allclose(out, expected, atol=1e-12)
    empty = polar_to_cartesian(torch.empty((2, 0, 3), dtype=p.dtype))
    assert empty.shape == (2, 0, 3)


def verify_followup_gradient(polar_to_cartesian):
    p = torch.tensor([[2., 0., 0.]], dtype=torch.float64, requires_grad=True)
    polar_to_cartesian(p).sum().backward()
    assert torch.allclose(p.grad, torch.tensor([[1., 2., 2.]], dtype=p.dtype), atol=1e-12)


def verify_production_invalid_inputs(polar_to_cartesian):
    cases = [torch.tensor([[-1., 0., 0.]]), torch.tensor([[float('nan'), 0., 0.]]),
             torch.ones(2, 2), torch.ones(2, 3, dtype=torch.int64)]
    for p in cases:
        try:
            polar_to_cartesian(p)
        except ValueError:
            continue
        raise AssertionError('invalid polar input accepted')


def verify_production_strided_dtype_and_no_mutation(polar_to_cartesian):
    base = torch.tensor([[2., 0., 0., 99.], [3., 0., 0., 99.]], dtype=torch.float32)
    p = base[:, :3]
    assert not p.is_contiguous()
    before = p.clone()
    out = polar_to_cartesian(p)
    assert out.dtype == p.dtype and out.device == p.device
    assert torch.allclose(out, torch.tensor([[2., 0., 0.], [3., 0., 0.]]), atol=1e-6)
    assert torch.equal(p, before)


# %%
from torch_judge import evaluate
if not evaluate('0054_polar_to_cartesian_eval.py', polar_to_cartesian):
    raise SystemExit(1)
