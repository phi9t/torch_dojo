"""Three-stage stable logits cross-entropy reference and checks."""
# %%
import torch
from torch.nn import functional as F

# %%
def cross_entropy_loss(logits, targets, ignore_index=-100):
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor) or logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits [B,C], C>=2, and targets [B] required")
    if not logits.is_floating_point() or not torch.isfinite(logits).all() or targets.ndim != 1 or targets.shape[0] != logits.shape[0] or targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) or targets.device != logits.device:
        raise ValueError("finite floating logits and matching integral targets required")
    valid = targets != ignore_index
    if torch.any((targets[valid] < 0) | (targets[valid] >= logits.shape[1])):
        raise ValueError("target outside class range")
    if not valid.any():
        return logits.sum() * 0
    selected = logits[valid]
    labels = targets[valid].long()
    return (torch.logsumexp(selected, dim=1) - selected.gather(1, labels[:, None]).squeeze(1)).mean()

# %% [verifiers]
def verify_baseline_known_and_mean(cross_entropy_loss):
    z = torch.tensor([[0., 0.], [2., 0.]], requires_grad=True)
    y = torch.tensor([0, 1])
    out = cross_entropy_loss(z, y)
    assert out.ndim == 0 and torch.allclose(out, F.cross_entropy(z, y), atol=1e-6)
    out.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def verify_followup_ignored_count_and_empty(cross_entropy_loss):
    z = torch.tensor([[0., 1.], [2000., -2000.], [2., 0.]], requires_grad=True)
    labels = torch.tensor([1, -100, 0])
    out = cross_entropy_loss(z, labels)
    assert torch.allclose(out, F.cross_entropy(z, labels), atol=1e-6)
    out.backward()
    assert torch.equal(z.grad[1], torch.zeros(2))
    all_ignored = torch.randn(3, 2, requires_grad=True)
    zero = cross_entropy_loss(all_ignored, torch.full((3,), -100))
    assert zero.shape == () and zero.item() == 0
    zero.backward()
    assert torch.equal(all_ignored.grad, torch.zeros_like(all_ignored))
    empty = torch.zeros(0, 2, requires_grad=True)
    assert cross_entropy_loss(empty, torch.empty(0, dtype=torch.long)).item() == 0


def verify_production_shifts_noncontiguous_and_gradient(cross_entropy_loss):
    z = torch.tensor([[1000., 1001.], [-1000., 1000.]], dtype=torch.float64).T.requires_grad_()
    labels = torch.tensor([1, 0])
    out = cross_entropy_loss(z, labels)
    shifted = cross_entropy_loss(z + torch.tensor([[500.], [-700.]], dtype=z.dtype), labels)
    assert torch.isfinite(out) and torch.allclose(out, shifted, atol=1e-12)
    assert torch.allclose(out, F.cross_entropy(z, labels), atol=1e-12)
    out.backward()
    assert torch.isfinite(z.grad).all() and torch.allclose(z.grad.sum(1), torch.zeros(2, dtype=z.dtype), atol=1e-12)


def verify_production_validate_targets(cross_entropy_loss):
    z = torch.zeros(1, 2)
    for labels in (torch.tensor([-1]), torch.tensor([2]), torch.tensor([.5]), torch.tensor([[0]])):
        try:
            cross_entropy_loss(z, labels)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid target accepted")
    try:
        cross_entropy_loss(torch.tensor([[float('nan'), 0.]]), torch.tensor([0]))
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite logits accepted")

# %%
from torch_judge import evaluate
if not evaluate('0016_cross_entropy_eval.py', cross_entropy_loss):
    raise SystemExit(1)
