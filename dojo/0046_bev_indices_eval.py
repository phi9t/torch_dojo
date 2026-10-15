"""Sureal-inspired BEV Grid Indices reference and judge run.

After installing from the repository root, run inside dojo/: python 0046_bev_indices_eval.py
"""

# %%
import torch

# %%
def bev_indices(points_xy: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor, resolution: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (points_xy.ndim != 2 or points_xy.shape[1] != 2
            or any(x.shape != (2,) for x in (minimum, maximum, resolution))
            or not points_xy.is_floating_point()
            or any(x.dtype != points_xy.dtype or x.device != points_xy.device for x in (minimum, maximum, resolution))
            or any(not torch.isfinite(x).all() for x in (points_xy, minimum, maximum, resolution))
            or torch.any(resolution <= 0) or torch.any(maximum <= minimum)):
        raise ValueError("invalid BEV grid")
    valid = ((points_xy >= minimum) & (points_xy < maximum)).all(dim=1)
    indices = torch.full(points_xy.shape, -1, dtype=torch.int64, device=points_xy.device)
    dimensions = torch.ceil((maximum - minimum) / resolution).to(torch.int64)
    cells = torch.floor((points_xy[valid] - minimum) / resolution).to(torch.int64)
    indices[valid] = torch.minimum(cells, dimensions - 1)
    return indices, valid

# %%


# %% [verifiers]
def verify_baseline_half_open_grid_and_invalid_sentinel(bev_indices):
    xy = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [0.999, 0.999], [1.0, 0.0], [-1.01, 0.0]])
    idx, valid = bev_indices(xy, torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.5]))
    assert idx.dtype == torch.int64 and valid.dtype == torch.bool
    assert torch.equal(idx, torch.tensor([[0, 0], [2, 2], [3, 3], [-1, -1], [-1, -1]]))
    assert torch.equal(valid, torch.tensor([True, True, True, False, False]))


def verify_followup_anisotropic_cells(bev_indices):
    xy = torch.tensor([[1.9, 1.9], [2.0, 0.0]])
    idx, valid = bev_indices(xy, torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0]), torch.tensor([1.0, 0.5]))
    assert torch.equal(idx, torch.tensor([[1, 3], [-1, -1]])) and torch.equal(valid, torch.tensor([True, False]))


def verify_followup_partial_last_cell(bev_indices):
    xy = torch.tensor([[2.4, .5], [2.5, .5]])
    idx, valid = bev_indices(xy, torch.zeros(2), torch.tensor([2.5, 1.]), torch.ones(2))
    assert torch.equal(idx, torch.tensor([[2, 0], [-1, -1]]))
    assert torch.equal(valid, torch.tensor([True, False]))


def verify_production_reject_invalid_resolution(bev_indices):
    try:
        bev_indices(torch.zeros(1, 2), torch.zeros(2), torch.ones(2), torch.tensor([0.0, 1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError('zero resolution must fail')


def verify_production_empty_and_malformed(bev_indices):
    xy = torch.empty((0, 2), dtype=torch.float64)
    idx, valid = bev_indices(xy, torch.zeros(2, dtype=xy.dtype), torch.ones(2, dtype=xy.dtype), torch.ones(2, dtype=xy.dtype))
    assert idx.shape == (0, 2) and valid.shape == (0,)
    for bad in (torch.full((1, 2), float('nan')), torch.zeros(1, 3)):
        try:
            bev_indices(bad, torch.zeros(2), torch.ones(2), torch.ones(2))
        except ValueError:
            pass
        else:
            raise AssertionError("malformed points accepted")


def verify_production_noncontiguous_and_immutability(bev_indices):
    xy = torch.tensor([[.25, .75, 9.], [1.25, .25, 8.]])[:, :2]
    before = xy.clone()
    idx, valid = bev_indices(xy, torch.zeros(2), torch.tensor([2., 1.]), torch.tensor([.5, .5]))
    assert torch.equal(idx, torch.tensor([[0, 1], [2, 0]])) and bool(valid.all())
    assert torch.equal(xy, before)


# %%
print("BEV Grid Indices: challenge the judge!")
from torch_judge import evaluate
if not evaluate('0046_bev_indices_eval.py', bev_indices):
    raise SystemExit(1)
