"""Reference and staged checks for a post-batch router bias controller."""
# %%
import math
import torch
from torch import nn

# %%
class BiasFeedback(nn.Module):
    def __init__(self, num_experts, step_size):
        super().__init__()
        if (type(num_experts) is not int or num_experts < 1
                or not isinstance(step_size, (int, float)) or not math.isfinite(step_size)
                or step_size < 0):
            raise ValueError('need E >= 1 and a finite nonnegative step size')
        self.step_size = float(step_size)
        self.register_buffer('bias', torch.zeros(num_experts))
        self.register_buffer('counts', torch.zeros(num_experts, dtype=torch.long))

    def forward(self, logits, k=1):
        experts = self.bias.numel()
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
                or logits.shape[1] != experts or not logits.is_floating_point()
                or logits.device != self.bias.device or not torch.isfinite(logits).all()
                or type(k) is not int or not 1 <= k <= experts):
            raise ValueError('need finite floating logits [T,E] and 1 <= k <= E')
        affinity = logits.sigmoid()
        # Pre-sort logits so underflowed equal affinities retain their
        # mathematical ordering; exact logit ties keep lower expert IDs.
        by_logit = torch.argsort(logits, dim=1, descending=True, stable=True)
        ranked_scores = (affinity + self.bias).gather(1, by_logit)
        by_score = torch.argsort(ranked_scores, dim=1, descending=True, stable=True)
        chosen = by_logit.gather(1, by_score)[:, :k]
        # log-sigmoid keeps relative selected mass when sigmoid underflows.
        log_affinity = torch.nn.functional.logsigmoid(logits).gather(1, chosen)
        weights = log_affinity.softmax(dim=1)
        with torch.no_grad():
            self.counts.add_(torch.bincount(chosen.reshape(-1), minlength=experts))
        return chosen, weights

    def finish_batch(self):
        with torch.no_grad():
            if self.counts.sum():
                target = self.counts.to(self.bias.dtype).mean()
                self.bias.add_(self.step_size * torch.sign(target - self.counts))
                self.counts.zero_()

# %% [verifiers]
def verify_baseline_feedback_sign(BiasFeedback):
    router = BiasFeedback(3, .1)
    ids, weights = router(torch.tensor([[4., 0., -2.]] * 4), k=1)
    assert torch.equal(ids, torch.zeros((4, 1), dtype=torch.long))
    assert torch.equal(weights, torch.ones((4, 1)))
    router.finish_batch()
    assert torch.allclose(router.bias, torch.tensor([-.1, .1, .1]))


def verify_followup_once_after_microbatch_accumulation(BiasFeedback):
    router = BiasFeedback(3, .2)
    router(torch.tensor([[5., 0., -5.]] * 2), k=1)
    assert torch.equal(router.bias, torch.zeros(3))
    assert torch.equal(router.counts, torch.tensor([2, 0, 0]))
    router(torch.tensor([[0., 5., -5.]]), k=1)
    assert torch.equal(router.bias, torch.zeros(3))
    assert torch.equal(router.counts, torch.tensor([2, 1, 0]))
    router.finish_batch()
    assert torch.allclose(router.bias, torch.tensor([-.2, 0., .2]))
    assert torch.equal(router.counts, torch.zeros(3, dtype=torch.long))
    router.finish_batch()
    assert torch.allclose(router.bias, torch.tensor([-.2, 0., .2]))


def verify_production_bias_selects_but_not_weights(BiasFeedback):
    router = BiasFeedback(3, .1).double()
    with torch.no_grad():
        router.bias.copy_(torch.tensor([0., 0., 2.]))
    logits = torch.tensor([[0., 1., -1.]], dtype=torch.float64, requires_grad=True)
    ids, weights = router(logits, k=2)
    assert torch.equal(ids, torch.tensor([[2, 1]]))
    raw = logits.detach().sigmoid()[0, [2, 1]]
    assert torch.allclose(weights[0], raw / raw.sum())
    assert not torch.allclose(weights[0], (raw + torch.tensor([2., 0.])) /
                              (raw.sum() + 2.))
    weights[0, 0].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2].abs() > 0 and router.bias.grad is None


def verify_production_stable_ties_and_empty_batch(BiasFeedback):
    router = BiasFeedback(3, .1)
    ids, weights = router(torch.zeros((2, 3)), k=2)
    assert torch.equal(ids, torch.tensor([[0, 1], [0, 1]]))
    assert torch.allclose(weights, torch.full((2, 2), .5))
    router.finish_batch()
    assert torch.allclose(router.bias, torch.tensor([-.1, -.1, .1]))
    before = router.bias.clone()
    ids, weights = router(torch.empty((0, 3)), k=2)
    assert ids.shape == weights.shape == (0, 2)
    router.finish_batch()
    assert torch.equal(router.bias, before)


def verify_production_equal_counts_and_invalid(BiasFeedback):
    router = BiasFeedback(2, .5)
    router(torch.tensor([[10., -10.], [-10., 10.]]))
    router.finish_batch()
    assert torch.equal(router.bias, torch.zeros(2))
    for args in ((0, .1), (2, -.1), (2, math.inf)):
        try:
            BiasFeedback(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('bad controller configuration accepted')
    for logits, k in ((torch.ones(2, 4), 1), (torch.ones(2, 2), 0),
                      (torch.ones(2, 2), 3), (torch.ones(2, 2).long(), 1),
                      (torch.tensor([[float('nan'), 0.]]), 1)):
        try:
            router(logits, k)
        except ValueError:
            pass
        else:
            raise AssertionError('bad router input accepted')


def verify_production_extreme_negative_logits(BiasFeedback):
    router = BiasFeedback(2, .1)
    logits = torch.tensor([[-1000., -1001.]], requires_grad=True)
    ids, weights = router(logits, k=2)
    assert torch.equal(ids, torch.tensor([[0, 1]]))
    expected = torch.tensor([-1000., -1001.]).softmax(0)
    assert torch.isfinite(weights).all() and torch.allclose(weights[0], expected, atol=1e-6)
    weights[0, 0].backward()
    assert torch.isfinite(logits.grad).all()
    # Both sigmoid values underflow in float32; the higher logit still wins.
    reversed_logits = torch.tensor([[-1001., -1000.]])
    chosen, _ = BiasFeedback(2, .1)(reversed_logits, k=1)
    assert torch.equal(chosen, torch.tensor([[1]]))

# %%
from torch_judge import evaluate
if not evaluate('0099_bias_feedback_eval.py', BiasFeedback):
    raise SystemExit(1)
