"""Reference and verifiers for the collision-safe scatter-add drill."""

# %%
import torch
from hypothesis import given, settings, strategies as st

# Drill 1: It prints `tensor([3., 0., 5.])`. Both values target slot 0, so one
# write replaces the other; which one survives is unspecified, and the 2 is lost.

# Drill 2: It prints `tensor([3., 0., 5.])`, the same loss. `out[index] += src`
# is `out[index] = out[index] + src`: it reads slot 0 twice (both zero), adds,
# then writes twice. Duplicate indices never see each other's contribution.

# Drill 3: It prints `tensor([5., 0., 5.])`. `index_add` applies every
# contribution, so duplicates sum. Order may change float rounding, not the sum.

# %%
def scatter_sum(index: torch.Tensor, src: torch.Tensor, size: int) -> torch.Tensor:
    if not isinstance(index, torch.Tensor) or not isinstance(src, torch.Tensor):
        raise ValueError("index and src must be tensors")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("size must be a nonnegative int")
    if index.dtype != torch.int64 or index.ndim != 1:
        raise ValueError("index must be an int64 vector")
    if src.ndim < 1 or src.shape[0] != index.shape[0]:
        raise ValueError("src needs one row per index")
    if src.dtype == torch.bool or not (src.dtype.is_floating_point or src.dtype in (torch.int32, torch.int64)):
        raise ValueError("src must be a float or integer tensor")
    if index.device != src.device:
        raise ValueError("index and src must share a device")
    if index.numel() and (int(index.min()) < 0 or int(index.max()) >= size):
        raise ValueError("index outside [0, size)")
    return src.new_zeros((size,) + tuple(src.shape[1:])).index_add(0, index, src)


# %% [verifiers]
def verify_baseline_hand_example(scatter_sum):
    out = scatter_sum(torch.tensor([0, 0, 2]), torch.tensor([2., 3., 5.]), 4)
    assert torch.equal(out, torch.tensor([5., 0., 5., 0.])), out


def verify_followup_rows_empty_and_integer_exactness(scatter_sum):
    index = torch.tensor([1, 0, 1, 1])
    src = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    assert torch.equal(scatter_sum(index, src, 3), torch.tensor([[3., 4.], [13., 16.], [0., 0.]]))
    empty = scatter_sum(torch.zeros(0, dtype=torch.int64), torch.zeros(0, 2), 3)
    assert empty.shape == (3, 2) and not empty.any()
    big = scatter_sum(torch.tensor([0, 0]), torch.tensor([2 ** 40, 1], dtype=torch.int64), 1)
    assert big.dtype == torch.int64 and int(big[0]) == 2 ** 40 + 1


def verify_followup_one_hot_oracle_and_order_independence(scatter_sum):
    @settings(max_examples=25, deadline=None, derandomize=True)
    @given(st.lists(st.integers(0, 3), min_size=1, max_size=10), st.randoms(use_true_random=False))
    def trial(ids, rng):
        index = torch.tensor(ids)
        src = torch.arange(1., len(ids) + 1).double().repeat(2, 1).T
        one_hot = torch.nn.functional.one_hot(index, 4).double()
        assert torch.allclose(scatter_sum(index, src, 4), one_hot.T @ src)
        order = list(range(len(ids))); rng.shuffle(order)
        perm = torch.tensor(order)
        assert torch.allclose(scatter_sum(index[perm], src[perm], 4), scatter_sum(index, src, 4))
    trial()


def verify_production_validation(scatter_sum):
    ok_index, ok_src = torch.tensor([0, 1]), torch.ones(2)
    bad = [(ok_index.float(), ok_src, 2), (torch.tensor([0, 2]), ok_src, 2), (torch.tensor([-1, 0]), ok_src, 2),
           (ok_index, torch.ones(3), 2), (ok_index, ok_src, -1), (ok_index, ok_src, 2.0),
           (ok_index, ok_src.bool(), 2), (ok_index.reshape(1, 2), ok_src, 2)]
    for index, src, size in bad:
        try:
            scatter_sum(index, src, size)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input accepted')


def verify_production_gradient_and_no_mutation(scatter_sum):
    index = torch.tensor([0, 0, 2])
    src = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64, requires_grad=True)
    before_index, before_src = index.clone(), src.detach().clone()
    out = scatter_sum(index, src, 3)
    weights = torch.tensor([[1., 10.], [2., 20.], [3., 30.]], dtype=torch.float64)
    (out * weights).sum().backward()
    assert torch.equal(src.grad, weights[index])
    assert torch.equal(index, before_index) and torch.equal(src.detach(), before_src)
    assert torch.autograd.gradcheck(lambda s: scatter_sum(index, s, 3), (src,))

# %%
from torch_judge import evaluate
if not evaluate("0114_scatter_sum_eval.py", scatter_sum):
    raise SystemExit(1)
