"""Reference and verifiers for row-major addresses and their traps."""

# %%
import math
import torch
from hypothesis import given, settings, strategies as st

# Drill 1: It prints `tensor(8)`. Row-major means the last axis moves fastest:
# (1*2 + 0)*3 + 2 = 8, which is also what `lin[1, 0, 2]` reads.

# Drill 2: It prints `3 3`. x = 3 is one past the last column, so (0,0,3) wraps
# to the first cell of the next row and gets the same address as (0,1,0).

# Drill 3: No. The flat address is identical, so nothing downstream can recover
# which coordinate was meant. Check each axis against its own bound BEFORE
# forming the address; the same hazard reaches across batches when b is wrong.

# %%
def _validate_shape(shape):
    if not isinstance(shape, (tuple, list)) or not shape or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError("shape must be a nonempty tuple of positive ints")
    if math.prod(shape) > 2 ** 63 - 1:
        raise ValueError("prod(shape) does not fit in int64")


def ravel_index(coords: torch.Tensor, shape: tuple) -> torch.Tensor:
    _validate_shape(shape)
    if not isinstance(coords, torch.Tensor) or coords.dtype != torch.int64 or coords.ndim != 2 or coords.shape[1] != len(shape):
        raise ValueError("coords must be int64 [K,D] with D == len(shape)")
    bounds = coords.new_tensor(shape)
    if coords.numel() and ((coords < 0) | (coords >= bounds)).any():
        raise ValueError("coordinate outside its axis")
    linear = torch.zeros(coords.shape[0], dtype=torch.int64, device=coords.device)
    for axis, size in enumerate(shape):
        linear = linear * size + coords[:, axis]
    return linear


# %% [verifiers]
def _unravel(linear: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Inverse oracle: peel axes from the fastest (last) to the slowest."""
    columns, rest = [], linear
    for size in reversed(shape):
        columns.append(rest % size)
        rest = torch.div(rest, size, rounding_mode='floor')
    return torch.stack(columns[::-1], dim=1)


def verify_baseline_hand_addresses(ravel_index):
    coords = torch.tensor([[1, 0, 2], [0, 1, 0]])
    assert ravel_index(coords, (2, 2, 3)).tolist() == [8, 3]
    assert torch.equal(_unravel(torch.tensor([8, 3]), (2, 2, 3)), coords)


def verify_followup_stride_oracle_and_round_trip(ravel_index):
    @settings(max_examples=20, deadline=None, derandomize=True)
    @given(st.lists(st.integers(1, 5), min_size=1, max_size=4))
    def trial(dims):
        shape = tuple(dims); n = math.prod(shape)
        table = torch.arange(n).reshape(shape)
        coords = _unravel(torch.arange(n), shape)
        assert torch.equal(table[tuple(coords.T)], torch.arange(n))
        assert torch.equal(ravel_index(coords, shape), torch.arange(n))
    trial()


def verify_followup_empty_input(ravel_index):
    assert ravel_index(torch.zeros(0, 3, dtype=torch.int64), (2, 2, 3)).shape == (0,)


def verify_production_aliasing_and_range(ravel_index):
    for coords in ([[0, 0, 3]], [[0, 2, 0]], [[2, 0, 0]], [[-1, 0, 0]], [[0, 0, -1]]):
        try:
            ravel_index(torch.tensor(coords), (2, 2, 3))
        except ValueError:
            pass
        else:
            raise AssertionError('coordinate outside its axis accepted: ' + str(coords))


def verify_production_shape_dtype_and_overflow(ravel_index):
    cases = [(torch.zeros(1, 2, dtype=torch.int64), (2 ** 32, 2 ** 32)), (torch.zeros(1, 2, dtype=torch.int64), (0, 3)),
             (torch.zeros(1, 2, dtype=torch.int64), (3,)), (torch.zeros(1, 2), (3, 3)),
             (torch.zeros(1, 2, dtype=torch.int64), (3, 3.0)), (torch.zeros(2, dtype=torch.int64), (3, 3))]
    for coords, shape in cases:
        try:
            ravel_index(coords, shape)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid ravel input accepted')

# %%
from torch_judge import evaluate
if not evaluate("0116_flat_addresses_eval.py", ravel_index):
    raise SystemExit(1)
