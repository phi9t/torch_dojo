"""Reference and staged checks for the direct preference loss."""
# %%
import math
import torch
import torch.nn.functional as F

# %%
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    values = (policy_chosen_logps, policy_rejected_logps,
              ref_chosen_logps, ref_rejected_logps)
    if any(v.ndim != 1 or v.shape != values[0].shape or v.device != values[0].device
           or v.dtype != values[0].dtype or not torch.isfinite(v).all() for v in values):
        raise ValueError("all log-probabilities must be finite matching [B] tensors")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if values[0].numel() == 0:
        return values[0].sum() * 0
    delta = (values[0] - values[1]) - (values[2].detach() - values[3].detach())
    return F.softplus(-beta * delta).mean()

# %% [verifiers]
def verify_baseline_known_gap(dpo_loss):
    z = torch.tensor([0., 0.])
    loss = dpo_loss(torch.tensor([0., 2.]), z, z, z, beta=1.)
    expected = (math.log(2) + math.log1p(math.exp(-2))) / 2
    assert loss.ndim == 0 and torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def verify_baseline_mean_not_sum(dpo_loss):
    c, r, rc, rr = [torch.tensor(x) for x in ([1., -2.], [-1., 1.], [0., 0.], [0., 0.])]
    one = dpo_loss(c, r, rc, rr, beta=0.3)
    twice = dpo_loss(c.repeat(2), r.repeat(2), rc.repeat(2), rr.repeat(2), beta=0.3)
    assert torch.allclose(one, twice)


def verify_followup_reference_is_detached(dpo_loss):
    c = torch.tensor([.1, -.3], requires_grad=True)
    r = torch.tensor([-.4, .2], requires_grad=True)
    rc = torch.tensor([-.2, -.1], requires_grad=True)
    rr = torch.tensor([-.3, -.5], requires_grad=True)
    loss = dpo_loss(c, r, rc, rr, beta=.5)
    loss.backward()
    assert c.grad is not None and r.grad is not None
    assert torch.isfinite(c.grad).all() and torch.isfinite(r.grad).all()
    assert rc.grad is None and rr.grad is None
    assert torch.allclose(c.grad, -r.grad)


def verify_followup_reference_gap_matters(dpo_loss):
    c, r = torch.tensor([0.]), torch.tensor([-1.])
    same = dpo_loss(c, r, c, r, beta=1.)
    weaker = dpo_loss(c, r, r, r, beta=1.)
    assert torch.allclose(same, torch.tensor(math.log(2)), atol=1e-6)
    assert weaker < same


def verify_production_extreme_gaps(dpo_loss):
    c = torch.tensor([1000., -1000.], dtype=torch.float64, requires_grad=True)
    r = torch.zeros_like(c)
    z = torch.zeros_like(c)
    loss = dpo_loss(c, r, z, z, beta=1.)
    assert torch.isfinite(loss) and torch.allclose(loss, torch.tensor(500., dtype=c.dtype), atol=1e-8)
    loss.backward()
    assert torch.isfinite(c.grad).all()


def verify_production_empty_and_validation(dpo_loss):
    empty = torch.empty(0, requires_grad=True)
    loss = dpo_loss(empty, empty, empty, empty)
    assert loss.ndim == 0 and loss.item() == 0
    loss.backward()
    assert empty.grad is not None
    for bad in (0., -1., float('nan')):
        try:
            dpo_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), beta=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid beta must fail")

# %%
from torch_judge import evaluate
if not evaluate('0037_dpo_loss_eval.py', dpo_loss):
    raise SystemExit(1)
