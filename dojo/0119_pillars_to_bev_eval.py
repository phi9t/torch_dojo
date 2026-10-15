"""Reference and verifiers for the pillar-rows to BEV layout."""

# %%
import torch

# Drill 1: It prints `torch.Size([1, 2, 2, 3]) (12, 1, 6, 2) False`. The view
# gives [B,Y,X,C] with strides (12,6,2,1); permuting only reorders the shape and
# strides to [B,C,Y,X], so the data is not laid out channel-major: not contiguous.

# Drill 2: It prints `view failed` and then `False`. Merging the permuted axes back into [6,2] needs a stride pattern
# the layout can't express, so `view` raises. `reshape` falls back to a copy, so
# its pointer differs from `flat`'s.

# Drill 3: It prints `True tensor(99.)`. `view` and `permute` return views onto
# the same storage; writing through `flat` changes the image too.

# %%
def pillars_to_bev(flat: torch.Tensor, batch_size: int, grid_y: int, grid_x: int) -> torch.Tensor:
    if not isinstance(flat, torch.Tensor) or flat.ndim != 2:
        raise ValueError("flat must be a [P,C] tensor")
    for size in (batch_size, grid_y, grid_x):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("sizes must be positive ints")
    if flat.shape[0] != batch_size * grid_y * grid_x:
        raise ValueError("flat needs B*Y*X rows")
    return flat.reshape(batch_size, grid_y, grid_x, flat.shape[1]).permute(0, 3, 1, 2)


# %% [verifiers]
def verify_baseline_hand_example(pillars_to_bev):
    flat = torch.arange(12.).reshape(6, 2)
    bev = pillars_to_bev(flat, 1, 2, 3)
    assert bev.shape == (1, 2, 2, 3)
    assert bev[0, 0].tolist() == [[0., 2., 4.], [6., 8., 10.]]
    assert bev[0, 1].tolist() == [[1., 3., 5.], [7., 9., 11.]]


def verify_followup_batch_order_and_loop_oracle(pillars_to_bev):
    flat = torch.arange(2 * 2 * 3 * 4.).reshape(12, 4)
    bev = pillars_to_bev(flat, 2, 2, 3)
    for b in range(2):
        for y in range(2):
            for x in range(3):
                assert torch.equal(bev[b, :, y, x], flat[(b * 2 + y) * 3 + x]), (b, y, x)


def verify_followup_view_shares_storage(pillars_to_bev):
    flat = torch.zeros(6, 2)
    bev = pillars_to_bev(flat, 1, 2, 3)
    assert bev.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr(), 'contiguous input must not be copied'
    flat[4, 1] = 7.
    assert bev[0, 1, 1, 1] == 7.


def verify_production_validation(pillars_to_bev):
    flat = torch.zeros(6, 2)
    for args in [(flat, 1, 2, 2), (flat, 0, 2, 3), (flat, 1, 2.0, 3), (flat, True, 2, 3), (flat[0], 1, 2, 3), (flat, 2, 2, 3)]:
        try:
            pillars_to_bev(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input accepted')


def verify_production_noncontiguous_zero_channels_and_gradient(pillars_to_bev):
    dense = torch.arange(12.).reshape(2, 6)
    bev = pillars_to_bev(dense.T, 1, 2, 3)            # a transposed, non-contiguous [6,2]
    assert torch.equal(bev, pillars_to_bev(dense.T.contiguous(), 1, 2, 3))
    assert pillars_to_bev(torch.zeros(6, 0), 1, 2, 3).shape == (1, 0, 2, 3)
    flat = torch.arange(12., dtype=torch.float64).reshape(6, 2).requires_grad_()
    weights = torch.arange(1., 13., dtype=torch.float64).reshape(1, 2, 2, 3)
    (pillars_to_bev(flat, 1, 2, 3) * weights).sum().backward()
    assert torch.equal(flat.grad, weights[0].permute(1, 2, 0).reshape(6, 2))

# %%
from torch_judge import evaluate
if not evaluate("0119_pillars_to_bev_eval.py", pillars_to_bev):
    raise SystemExit(1)
