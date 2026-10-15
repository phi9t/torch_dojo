"""Reference and staged checks for on-policy distillation reverse KL."""
# %%
import math
import torch
import torch.nn.functional as F
from torch import Tensor

# %%
def opd_loss(student_logits: Tensor, teacher_logits: Tensor,
             teacher_weights: Tensor | None = None, mask: Tensor | None = None,
             temperature: float = 1.0) -> Tensor:
    if (student_logits.ndim < 2 or student_logits.shape[-1] < 2
            or not student_logits.is_floating_point() or not torch.isfinite(student_logits).all()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("expect finite student [...,V], V>=2 and positive temperature")
    if teacher_logits.shape == student_logits.shape:
        teachers = teacher_logits.unsqueeze(0)
    elif teacher_logits.ndim == student_logits.ndim + 1 and teacher_logits.shape[1:] == student_logits.shape:
        teachers = teacher_logits
    else:
        raise ValueError("teacher shape must be [...,V] or [T,...,V]")
    if (teachers.shape[0] == 0 or teachers.device != student_logits.device
            or teachers.dtype != student_logits.dtype or not torch.isfinite(teachers).all()):
        raise ValueError("teachers must be nonempty, finite and match dtype/device")
    n = teachers.shape[0]
    if teacher_weights is None:
        weights = student_logits.new_full((n,), 1 / n)
    else:
        if (teacher_weights.shape != (n,) or teacher_weights.device != student_logits.device
                or not torch.isfinite(teacher_weights).all() or (teacher_weights < 0).any()
                or teacher_weights.sum() <= 0):
            raise ValueError("weights must be nonnegative [T] with positive sum")
        weights = teacher_weights.detach().to(student_logits.dtype)
        weights = weights / weights.sum()
    if mask is not None:
        if mask.shape != student_logits.shape[:-1] or mask.device != student_logits.device or not torch.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("mask must be finite nonnegative token weights")
        valid = mask.to(student_logits.dtype)
    else:
        valid = student_logits.new_ones(student_logits.shape[:-1])
    if valid.sum().item() == 0:
        return student_logits.sum() * 0
    t = float(temperature)
    s = F.log_softmax(student_logits / t, dim=-1)
    q = F.log_softmax(teachers.detach() / t, dim=-1)
    kl = (s.exp().unsqueeze(0) * (s.unsqueeze(0) - q)).sum(-1)
    weighted = (kl * weights.view((n,) + (1,) * valid.ndim)).sum(0)
    return (weighted * valid).sum() / valid.sum() * (t * t)

# %% [verifiers]
def verify_baseline_single_teacher_known_kl(opd_loss):
    s = torch.tensor([[0., 0.]], requires_grad=True)
    q = torch.tensor([[math.log(3), 0.]])
    # KL((1/2,1/2) || (3/4,1/4)) = log(2/sqrt(3)).
    expected = math.log(2 / math.sqrt(3))
    assert torch.allclose(opd_loss(s, q), torch.tensor(expected), atol=1e-6)


def verify_baseline_matches_self(opd_loss):
    x = torch.tensor([[[1., 2., 3.], [2., 0., 0.]]], requires_grad=True)
    assert abs(opd_loss(x, x.detach()).item()) < 1e-6


def verify_followup_teacher_weights_and_mask_denominator(opd_loss):
    s = torch.tensor([[[0., 0.], [0., 0.], [0., 0.]]])
    q0 = torch.tensor([[[math.log(3), 0.], [0., 0.], [0., 0.]]])
    q1 = torch.tensor([[[0., 0.], [math.log(3), 0.], [0., 0.]]])
    loss = opd_loss(s, torch.stack([q0, q1]), torch.tensor([1., 3.]), mask=torch.tensor([[1., 2., 0.]]))
    k = math.log(2 / math.sqrt(3))
    expected = (k * .25 + k * 2 * .75) / 3
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def verify_followup_teacher_and_weight_detach(opd_loss):
    s = torch.tensor([[[.2, -.3]]], requires_grad=True)
    q = torch.tensor([[[.4, -.6]]], requires_grad=True)
    weights = torch.tensor([1.], requires_grad=True)
    loss = opd_loss(s, q.unsqueeze(0), weights)
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert q.grad is None and weights.grad is None


def verify_production_temperature_and_all_masked(opd_loss):
    s = torch.tensor([[[2., 0.]]], requires_grad=True)
    q = torch.tensor([[[0., 2.]]])
    t = 2.
    sp = F.log_softmax(s / t, -1)
    qp = F.log_softmax(q / t, -1)
    expected = (sp.exp() * (sp - qp)).sum(-1).mean() * t * t
    assert torch.allclose(opd_loss(s, q, temperature=t), expected)
    loss = opd_loss(s, q, mask=torch.zeros(1, 1))
    assert loss.item() == 0
    loss.backward()
    assert s.grad is not None and torch.equal(s.grad, torch.zeros_like(s))


def verify_production_invalid_contract(opd_loss):
    s = torch.zeros(1, 2)
    q = torch.zeros(1, 2)
    for kwargs in ({'temperature': 0.}, {'mask': torch.ones(2)},
                   {'teacher_weights': torch.tensor([0.])},
                   {'teacher_logits': torch.zeros(2, 2, 2)}):
        try:
            opd_loss(s, kwargs.pop('teacher_logits', q), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input must fail")

# %%
from torch_judge import evaluate
if not evaluate('0041_opd_loss_eval.py', opd_loss):
    raise SystemExit(1)
