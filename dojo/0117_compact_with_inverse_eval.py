"""Reference and verifiers for mask, compact, and inverse."""

# %%
import torch

# Drill 1: It prints `[[0, 0], [0, 2], [1, 1], [1, 2]]`. `nonzero` lists the
# True entries in row-major order: batch first, then slot. Four rows, M = 4.

# Drill 2: It prints `[0.0, 2.0, 4.0, 5.0]`. Boolean-mask indexing walks the same
# row-major order, so row m of `x[mask]` belongs to address m of `mask.nonzero()`.
# That alignment is the whole contract: both must iterate identically.

# Drill 3: `[[0, -1, 1], [-1, 2, 3]]`. Start from -1 everywhere, then write the
# compact row id at each kept address. Excluded slots keep -1, which is how
# "missing" is represented instead of inspecting values.

# %%
def compact_with_inverse(x: torch.Tensor, mask: torch.Tensor):
    if not isinstance(x, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("tensors required")
    if x.ndim != 3 or mask.dtype != torch.bool or mask.shape != x.shape[:2]:
        raise ValueError("x must be [B,N,C] and mask bool [B,N]")
    if x.device != mask.device:
        raise ValueError("common device required")
    addr = mask.nonzero(as_tuple=False)
    compact = x[mask]
    inverse = torch.full(mask.shape, -1, dtype=torch.int64, device=x.device)
    inverse[addr[:, 0], addr[:, 1]] = torch.arange(addr.shape[0], device=x.device)
    return compact, addr, inverse


# %% [verifiers]
def verify_baseline_hand_example(compact_with_inverse):
    x = torch.arange(6.).reshape(2, 3, 1)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    compact, addr, inverse = compact_with_inverse(x, mask)
    assert compact[:, 0].tolist() == [0., 2., 4., 5.]
    assert addr.tolist() == [[0, 0], [0, 2], [1, 1], [1, 2]]
    assert inverse.tolist() == [[0, -1, 1], [-1, 2, 3]]


def verify_followup_scatter_back_rebuilds_kept_rows(compact_with_inverse):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(3, 5, 2, generator=generator)
    mask = torch.rand(3, 5, generator=generator) > .4
    compact, addr, inverse = compact_with_inverse(x, mask)
    rebuilt = torch.zeros_like(x); rebuilt[addr[:, 0], addr[:, 1]] = compact
    assert torch.equal(rebuilt[mask], x[mask]) and not rebuilt[~mask].any()
    assert torch.equal(compact[inverse[mask]], x[mask]) and (inverse[~mask] == -1).all()
    assert inverse.dtype == addr.dtype == torch.int64 and compact.shape == (int(mask.sum()), 2)


def verify_followup_all_false_and_nan_padding(compact_with_inverse):
    x = torch.randn(2, 3, 4)
    compact, addr, inverse = compact_with_inverse(x, torch.zeros(2, 3, dtype=torch.bool))
    assert compact.shape == (0, 4) and addr.shape == (0, 2) and (inverse == -1).all() and inverse.shape == (2, 3)
    x[0, 1] = float('nan')
    mask = torch.tensor([[True, False, True], [True, True, True]])
    compact, _, _ = compact_with_inverse(x, mask)
    assert torch.isfinite(compact).all()


def verify_production_validation(compact_with_inverse):
    x = torch.zeros(2, 3, 1)
    for bad_x, bad_mask in [(x, torch.ones(2, 3)), (x, torch.ones(2, 2, dtype=torch.bool)),
                            (torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool)),
                            (x, torch.ones(2, 3, 1, dtype=torch.bool))]:
        try:
            compact_with_inverse(bad_x, bad_mask)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input accepted')


def verify_production_gradient_strides_and_no_mutation(compact_with_inverse):
    x = torch.randn(4, 3, 2, dtype=torch.float64).transpose(0, 1).detach().requires_grad_()   # non-contiguous view
    assert not x.is_contiguous()
    mask = torch.tensor([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    before, mask_before = x.detach().clone(), mask.clone()
    compact, _, _ = compact_with_inverse(x, mask)
    (compact * torch.arange(1., compact.shape[0] + 1, dtype=torch.float64)[:, None]).sum().backward()
    assert (x.grad[~mask] == 0).all() and (x.grad[mask][:, 0] == torch.arange(1., int(mask.sum()) + 1, dtype=torch.float64)).all()
    assert torch.equal(x.detach(), before) and torch.equal(mask, mask_before)

# %%
from torch_judge import evaluate
if not evaluate("0117_compact_with_inverse_eval.py", compact_with_inverse):
    raise SystemExit(1)
