"""Reference and checks for masked token-to-pair DPO."""

# %%
import math
from numbers import Real
import torch
import torch.nn.functional as F

# %%
def dpo_token_pairs(policy_chosen_logits, policy_rejected_logits,
                    reference_chosen_logits, reference_rejected_logits,
                    chosen_tokens, rejected_tokens, chosen_mask, rejected_mask,
                    beta=0.1):
    logits = (policy_chosen_logits, policy_rejected_logits,
              reference_chosen_logits, reference_rejected_logits)
    shape = policy_chosen_logits.shape
    if (len(shape) != 3 or shape[1] < 1 or shape[2] < 2
            or any(not isinstance(x, torch.Tensor) or x.shape != shape
                   or x.dtype != logits[0].dtype or x.device != logits[0].device
                   or not x.is_floating_point() or not torch.isfinite(x).all()
                   for x in logits)):
        raise ValueError('logits must be finite matching [B,T,V] tensors')
    batch, time, vocab = shape
    for ids, mask in ((chosen_tokens, chosen_mask), (rejected_tokens, rejected_mask)):
        if (not isinstance(ids, torch.Tensor) or ids.shape != (batch, time)
                or ids.dtype != torch.long or ids.device != logits[0].device
                or not isinstance(mask, torch.Tensor) or mask.shape != (batch, time)
                or mask.dtype != torch.bool or mask.device != logits[0].device
                or torch.any((ids[mask] < 0) | (ids[mask] >= vocab))
                or (batch > 0 and not torch.all(mask.any(dim=1)))):
            raise ValueError('each completion needs valid masked token indices')
    if not isinstance(beta, Real) or isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    if batch == 0:
        return policy_chosen_logits.sum() * 0

    def sequence_logps(values, ids, mask):
        safe_ids = torch.where(mask, ids, 0)
        selected = F.log_softmax(values, dim=-1).gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1)
        return (selected * mask).sum(dim=1)

    pc = sequence_logps(policy_chosen_logits, chosen_tokens, chosen_mask)
    pr = sequence_logps(policy_rejected_logits, rejected_tokens, rejected_mask)
    rc = sequence_logps(reference_chosen_logits.detach(), chosen_tokens, chosen_mask)
    rr = sequence_logps(reference_rejected_logits.detach(), rejected_tokens, rejected_mask)
    return F.softplus(-beta * ((pc - pr) - (rc - rr))).mean()


# %% [verifiers]
def verify_baseline_one_token_pair(dpo_token_pairs):
    chosen = torch.tensor([[[math.log(3), 0.]]], requires_grad=True)
    uniform = torch.zeros(1, 1, 2)
    ids = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = dpo_token_pairs(chosen, uniform, uniform, uniform,
                           ids, ids, mask, mask, beta=1.)
    assert torch.allclose(loss, torch.tensor(math.log(5 / 3)), atol=1e-6)
    loss.backward()
    assert chosen.grad is not None and torch.isfinite(chosen.grad).all()


def verify_baseline_batch_mean(dpo_token_pairs):
    c = torch.tensor([[[1., 0.]], [[0., 1.]]])
    z = torch.zeros_like(c)
    ids = torch.zeros(2, 1, dtype=torch.long)
    mask = torch.ones(2, 1, dtype=torch.bool)
    one = dpo_token_pairs(c, z, z, z, ids, ids, mask, mask)
    both = dpo_token_pairs(c.repeat(2, 1, 1), z.repeat(2, 1, 1),
                           z.repeat(2, 1, 1), z.repeat(2, 1, 1),
                           ids.repeat(2, 1), ids.repeat(2, 1),
                           mask.repeat(2, 1), mask.repeat(2, 1))
    assert torch.allclose(one, both)


def verify_followup_masks_exclude_prompt_and_padding(dpo_token_pairs):
    c = torch.tensor([[[10., -10.], [0., 0.], [-10., 10.]]], requires_grad=True)
    r = torch.zeros_like(c)
    ids = torch.tensor([[-100, 0, -100]])
    mask = torch.tensor([[False, True, False]])
    loss = dpo_token_pairs(c, r, r, r, ids, ids, mask, mask, beta=1.)
    assert torch.allclose(loss, torch.tensor(math.log(2.)), atol=1e-6)
    loss.backward()
    assert torch.count_nonzero(c.grad[:, [0, 2]]) == 0


def verify_followup_reference_is_frozen(dpo_token_pairs):
    c = torch.tensor([[[0.3, -0.2]]], requires_grad=True)
    r = torch.tensor([[[0.1, -0.1]]], requires_grad=True)
    rc = torch.tensor([[[0.4, 0.]]], requires_grad=True)
    rr = torch.tensor([[[-0.2, 0.2]]], requires_grad=True)
    ids = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)
    dpo_token_pairs(c, r, rc, rr, ids, ids, mask, mask).backward()
    assert c.grad is not None and r.grad is not None
    assert rc.grad is None and rr.grad is None


def verify_followup_reference_gap_changes_loss(dpo_token_pairs):
    uniform = torch.zeros(1, 1, 2)
    reference_chosen = torch.tensor([[[math.log(3.), 0.]]])
    ids = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    loss = dpo_token_pairs(uniform, uniform, reference_chosen, uniform,
                           ids, ids, mask, mask, beta=1.)
    assert torch.allclose(loss, torch.tensor(math.log(5. / 2.)), atol=1e-6)


def verify_production_invalid_contract(dpo_token_pairs):
    z = torch.zeros(1, 1, 2)
    ids = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)
    args = (z, z, z, z, ids, ids, mask, mask)
    invalid = [
        (args, {'beta': 0.}),
        ((z, z, z, z, torch.full_like(ids, 2), ids, mask, mask), {}),
        ((z, z, z, z, ids, ids, torch.zeros_like(mask), mask), {}),
        ((z, z, z, z, ids, ids, ids, mask), {}),
        ((z, torch.zeros(1, 2, 2), z, z, ids, ids, mask, mask), {}),
    ]
    for positional, kwargs in invalid:
        try:
            dpo_token_pairs(*positional, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid DPO pair accepted')


def verify_production_extreme_gap_and_empty_batch(dpo_token_pairs):
    c = torch.tensor([[[-1000., 1000.]]], dtype=torch.float64, requires_grad=True)
    z = torch.zeros_like(c)
    ids = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = dpo_token_pairs(c, z, z, z, ids, ids, mask, mask, beta=1.)
    assert torch.allclose(loss, torch.tensor(2000. - math.log(2.), dtype=torch.float64))
    loss.backward()
    assert torch.isfinite(c.grad).all()
    empty = torch.empty(0, 1, 2, requires_grad=True)
    empty_ids = torch.empty(0, 1, dtype=torch.long)
    empty_mask = torch.empty(0, 1, dtype=torch.bool)
    loss = dpo_token_pairs(empty, empty, empty, empty,
                           empty_ids, empty_ids, empty_mask, empty_mask)
    assert loss.item() == 0 and loss.requires_grad
    loss.backward()
    assert empty.grad is not None


# %%
from torch_judge import evaluate
if not evaluate('0085_dpo_token_pairs_eval.py', dpo_token_pairs):
    raise SystemExit(1)
