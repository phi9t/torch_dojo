"""Reference and verifiers for a gather with an explicit scatter-add backward."""

# %%
import torch
from hypothesis import given, settings, strategies as st

# Drill 1: It prints `[[4.0, 5.0], [0.0, 1.0], [4.0, 5.0]]`. Each id picks a row
# of the table; id 2 appears twice, so row 2 is copied twice.

# Drill 2: It prints `[[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]]`. A gather reads row 2
# twice, so the table row contributes to the loss twice and its gradient is the
# sum of both upstream gradients.

# Drill 3: Rows 0 and 2 get gradient; row 1 was never read. Row 2 holds 2, not 1,
# because backward of a gather is a scatter-ADD: it accumulates every upstream
# gradient that read that row. `grad[ids] = grad_out` would keep only one.

# %%
class GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, ids):
        if not isinstance(table, torch.Tensor) or not isinstance(ids, torch.Tensor):
            raise ValueError("tensors required")
        if table.ndim != 2 or ids.dtype != torch.int64 or ids.ndim != 1 or table.device != ids.device:
            raise ValueError("table must be [R,C] and ids int64 [K] on one device")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= table.shape[0]):
            raise ValueError("ids outside [0, R)")
        ctx.save_for_backward(ids)
        ctx.num_rows = table.shape[0]
        return table.index_select(0, ids)

    @staticmethod
    def backward(ctx, grad_out):
        ids, = ctx.saved_tensors
        grad_table = grad_out.new_zeros((ctx.num_rows,) + tuple(grad_out.shape[1:])).index_add(0, ids, grad_out)
        return grad_table, None


# %% [verifiers]
def verify_baseline_forward_matches_indexing(GatherRows):
    table = torch.arange(6.).reshape(3, 2)
    assert torch.equal(GatherRows.apply(table, torch.tensor([2, 0, 2])), table[torch.tensor([2, 0, 2])])


def verify_followup_backward_accumulates_duplicates(GatherRows):
    table = torch.arange(6.).reshape(3, 2).requires_grad_()
    GatherRows.apply(table, torch.tensor([2, 0, 2])).sum().backward()
    assert table.grad.tolist() == [[1., 1.], [0., 0.], [2., 2.]], table.grad


def verify_followup_matches_autograd_on_generated_ids(GatherRows):
    @settings(max_examples=25, deadline=None, derandomize=True)
    @given(st.lists(st.integers(0, 4), max_size=10))
    def trial(ids):
        index = torch.tensor(ids, dtype=torch.int64)
        weights = torch.arange(1., len(ids) * 3 + 1).reshape(len(ids), 3).double()
        mine = torch.randn(5, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)).requires_grad_()
        theirs = mine.detach().clone().requires_grad_()
        (GatherRows.apply(mine, index) * weights).sum().backward()
        (theirs[index] * weights).sum().backward()
        assert torch.allclose(mine.grad, theirs.grad)
    trial()


def verify_production_gradcheck_and_noncontiguous_upstream(GatherRows):
    table = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([3, 1, 3, 0])
    assert torch.autograd.gradcheck(lambda t: GatherRows.apply(t, ids), (table,))
    out = GatherRows.apply(table, ids)
    out.backward(torch.ones(3, 4, dtype=torch.float64).T)       # a non-contiguous [4,3] gradient
    assert table.grad.tolist()[3] == [2., 2., 2.] and table.grad.tolist()[2] == [0., 0., 0.]


def verify_production_validation_empty_and_reuse(GatherRows):
    table = torch.zeros(3, 2)
    for t, ids in [(table, torch.tensor([3])), (table, torch.tensor([-1])), (table, torch.tensor([0.])),
                   (table[0], torch.tensor([0])), (table, torch.tensor([[0]]))]:
        try:
            GatherRows.apply(t, ids)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input accepted')
    t = torch.ones(3, 2, requires_grad=True)
    GatherRows.apply(t, torch.zeros(0, dtype=torch.int64)).sum().backward()
    assert t.grad.shape == (3, 2) and not t.grad.any()
    u = torch.ones(3, 2, requires_grad=True)
    (GatherRows.apply(u, torch.tensor([0])).sum() + GatherRows.apply(u, torch.tensor([0, 1])).sum()).backward()
    assert u.grad.tolist() == [[2., 2.], [1., 1.], [0., 0.]]

# %%
from torch_judge import evaluate
if not evaluate("0120_gather_rows_eval.py", GatherRows):
    raise SystemExit(1)
