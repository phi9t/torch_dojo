"""Reference and cumulative checks for Kaiming normal initialization."""

# %%
import math
import torch

# %%
def kaiming_init(weight: torch.Tensor) -> torch.Tensor:
    if (weight.layout != torch.strided or not weight.is_floating_point()
            or weight.ndim < 1 or any(size == 0 for size in weight.shape)):
        raise ValueError("weight must be nonempty, dense, and floating")
    fan_in = math.prod(weight.shape[1:]) if weight.ndim > 1 else weight.shape[0]
    with torch.no_grad():
        weight.normal_(0, math.sqrt(2.0 / fan_in))
    return weight


# %% [verifiers]
def verify_baseline_dense_scale(kaiming_init):
    torch.manual_seed(120)
    w = torch.empty(4096, 64)
    assert kaiming_init(w) is w
    assert abs(w.mean().item()) < .01
    assert abs(w.std().item() - math.sqrt(2 / 64)) < .003


def verify_baseline_fan_in_changes_scale(kaiming_init):
    torch.manual_seed(121)
    narrow, wide = torch.empty(4096, 16), torch.empty(4096, 256)
    kaiming_init(narrow)
    kaiming_init(wide)
    assert 3.7 < narrow.std().item() / wide.std().item() < 4.3


def verify_followup_convolution_fan_in(kaiming_init):
    torch.manual_seed(122)
    w = torch.empty(256, 16, 3, 3)
    kaiming_init(w)
    assert abs(w.std().item() - math.sqrt(2 / 144)) < .002
    v = torch.empty(8192)
    kaiming_init(v)
    assert abs(v.std().item() - math.sqrt(2 / 8192)) < .001


def verify_followup_invalid_extent(kaiming_init):
    for shape in ((0, 4), (2, 0), (2, 3, 0)):
        try:
            kaiming_init(torch.empty(shape))
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted empty extent {shape}")


def verify_production_dtype_rng_and_grad(kaiming_init):
    for dtype in (torch.float32, torch.float64):
        torch.manual_seed(123)
        first = torch.empty(80, 20, dtype=dtype, requires_grad=True)
        kaiming_init(first)
        torch.manual_seed(123)
        second = torch.empty(80, 20, dtype=dtype, requires_grad=True)
        kaiming_init(second)
        assert first.dtype == second.dtype == dtype
        assert torch.equal(first, second) and first.requires_grad
        (first.square().sum()).backward()
        assert first.grad is not None


def verify_production_reject_nonfloating_or_sparse(kaiming_init):
    invalid = (torch.empty(4, 3, dtype=torch.int64),
               torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.]), (2, 3), check_invariants=True))
    for w in invalid:
        try:
            kaiming_init(w)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid weight")


# %%
from torch_judge import evaluate
if not evaluate("0020_weight_init_eval.py", kaiming_init):
    raise SystemExit(1)
