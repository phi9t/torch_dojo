"""Reference and checks for scalar pairwise reward-model training."""

# %%
import math
from numbers import Real
import torch
import torch.nn.functional as F


# %%
def pairwise_reward_loss(chosen_scores, rejected_scores, margin=0.0):
    if (not isinstance(chosen_scores, torch.Tensor)
            or not isinstance(rejected_scores, torch.Tensor)
            or chosen_scores.ndim != 1 or chosen_scores.numel() == 0
            or rejected_scores.shape != chosen_scores.shape
            or not chosen_scores.is_floating_point()
            or rejected_scores.dtype != chosen_scores.dtype
            or rejected_scores.device != chosen_scores.device
            or not torch.isfinite(chosen_scores).all()
            or not torch.isfinite(rejected_scores).all()):
        raise ValueError('need finite, matching, nonempty floating score vectors')
    if (not isinstance(margin, Real) or isinstance(margin, bool)
            or not math.isfinite(margin) or margin < 0):
        raise ValueError('margin must be finite and nonnegative')
    difference = chosen_scores - rejected_scores
    loss = F.softplus(margin - difference).mean()
    accuracy = (chosen_scores.detach() > rejected_scores.detach()).to(chosen_scores.dtype).mean()
    return loss, accuracy


# %% [verifiers]
def verify_baseline_mean_loss_and_strict_accuracy(pairwise_reward_loss):
    chosen = torch.tensor([2., 0.], requires_grad=True)
    rejected = torch.zeros(2, requires_grad=True)
    loss, accuracy = pairwise_reward_loss(chosen, rejected)
    expected = (math.log1p(math.exp(-2)) + math.log(2)) / 2
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)
    assert torch.allclose(accuracy, torch.tensor(.5))
    assert not accuracy.requires_grad
    loss.backward()
    assert torch.allclose(chosen.grad, torch.tensor([-.05960146, -.25]), atol=1e-6)
    assert torch.allclose(rejected.grad, -chosen.grad, atol=1e-6)


def verify_baseline_swapped_scores_change_loss(pairwise_reward_loss):
    preferred, other = torch.tensor([3., 2.]), torch.tensor([0., -1.])
    good_loss, good_accuracy = pairwise_reward_loss(preferred, other)
    bad_loss, bad_accuracy = pairwise_reward_loss(other, preferred)
    assert good_loss < .05 and good_accuracy == 1
    assert bad_loss > 3 and bad_accuracy == 0


def verify_followup_margin_and_ties(pairwise_reward_loss):
    chosen = torch.tensor([2., 0.], requires_grad=True)
    rejected = torch.zeros(2)
    loss, accuracy = pairwise_reward_loss(chosen, rejected, margin=1.)
    expected = (math.log1p(math.exp(-1)) + math.log1p(math.exp(1))) / 2
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)
    assert accuracy == .5  # The second pair is tied, even with a margin.
    loss.backward()
    assert torch.allclose(chosen.grad, torch.tensor([-.1344707, -.3655293]), atol=1e-6)


def verify_production_extreme_wrong_way_gap(pairwise_reward_loss):
    chosen = torch.tensor([-1000.], requires_grad=True)
    rejected = torch.tensor([1000.], requires_grad=True)
    loss, accuracy = pairwise_reward_loss(chosen, rejected, margin=2.)
    assert torch.isfinite(loss) and torch.allclose(loss, torch.tensor(2002.))
    assert accuracy == 0
    loss.backward()
    assert torch.equal(chosen.grad, torch.tensor([-1.]))
    assert torch.equal(rejected.grad, torch.tensor([1.]))


def verify_production_invalid_inputs_and_no_mutation(pairwise_reward_loss):
    chosen = torch.tensor([1., 2.], requires_grad=True)
    rejected = torch.tensor([0., 3.], requires_grad=True)
    chosen_before, rejected_before = chosen.detach().clone(), rejected.detach().clone()
    pairwise_reward_loss(chosen, rejected, margin=.3)
    assert torch.equal(chosen, chosen_before) and torch.equal(rejected, rejected_before)
    invalid = [
        (torch.empty(0), torch.empty(0), 0.),
        (torch.zeros(2, 1), torch.zeros(2, 1), 0.),
        (torch.zeros(2), torch.zeros(1), 0.),
        (torch.tensor([float('nan')]), torch.zeros(1), 0.),
        (torch.tensor([float('inf')]), torch.zeros(1), 0.),
        (torch.zeros(1), torch.tensor([float('-inf')]), 0.),
        (torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), 0.),
        (torch.zeros(1, dtype=torch.float64), torch.zeros(1), 0.),
        (torch.zeros(1), torch.zeros(1), -.1),
        (torch.zeros(1), torch.zeros(1), float('nan')),
        (torch.zeros(1), torch.zeros(1), True),
    ]
    for c, r, margin in invalid:
        try:
            pairwise_reward_loss(c, r, margin)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid reward-model input accepted')


# %%
from torch_judge import evaluate
if not evaluate('0089_pairwise_reward_eval.py', pairwise_reward_loss):
    raise SystemExit(1)
