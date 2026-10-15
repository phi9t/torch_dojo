"""Reference and staged checks for soft-score MoE load balancing."""
# %%
import torch

# %%
def route_balance(score_batches, route_batches, mode='global'):
    if (mode not in ('global', 'local', 'prefix')
            or not isinstance(score_batches, (tuple, list)) or not score_batches
            or not isinstance(route_batches, (tuple, list))
            or len(route_batches) != len(score_batches)):
        raise ValueError('need equally many score and route batches and a known mode')
    first = score_batches[0]
    if (not isinstance(first, torch.Tensor) or first.ndim != 2
            or not first.is_floating_point() or first.shape[1] < 2):
        raise ValueError('scores must be floating [T,E] with E >= 2')
    experts, dtype, device = first.shape[1], first.dtype, first.device
    counts = []
    for scores, ids in zip(score_batches, route_batches):
        if (not isinstance(scores, torch.Tensor) or scores.ndim != 2
                or scores.shape[1] != experts or scores.dtype != dtype or scores.device != device
                or not torch.isfinite(scores).all() or (scores < 0).any()
                or not torch.allclose(scores.sum(1), torch.ones(scores.shape[0], dtype=dtype, device=device),
                                      atol=1e-6, rtol=1e-6)
                or not isinstance(ids, torch.Tensor) or ids.dtype != torch.long
                or ids.device != device or ids.ndim != 2 or ids.shape[0] != scores.shape[0]
                or ids.shape[1] < 1 or ids.shape[1] > experts
                or (ids < 0).any() or (ids >= experts).any()):
            raise ValueError('invalid probability or selected-expert batch')
        if ids.shape[1] > 1 and (ids.sort(1).values[:, 1:] == ids.sort(1).values[:, :-1]).any():
            raise ValueError('duplicate expert in one token')
        counts.append(torch.bincount(ids.reshape(-1), minlength=experts))

    total_tokens = sum(scores.shape[0] for scores in score_batches)
    if total_tokens == 0:
        return sum(scores.sum() for scores in score_batches) * 0., 0.

    def hard_frequency(count):
        return count.to(dtype) / count.sum()

    def imbalance(count):
        return float(count.max()) * experts / int(count.sum())

    if mode == 'global':
        count = sum(counts)
        mean_scores = sum(scores.sum(0) for scores in score_batches) / total_tokens
        return experts * torch.dot(hard_frequency(count), mean_scores), imbalance(count)

    values, loads = [], []
    prefix_count = torch.zeros(experts, dtype=torch.long, device=device)
    for scores, count in zip(score_batches, counts):
        if scores.shape[0] == 0:
            continue
        prefix_count = prefix_count + count
        active_count = count if mode == 'local' else prefix_count
        values.append(experts * torch.dot(hard_frequency(active_count), scores.mean(0)))
        loads.append(imbalance(count))
    if mode == 'local':
        return torch.stack(values).mean(), sum(loads) / len(loads)
    weights = [scores.shape[0] / total_tokens for scores in score_batches if scores.shape[0]]
    return sum(value * weight for value, weight in zip(values, weights)), imbalance(prefix_count)

# %% [verifiers]
def verify_baseline_detached_counts_and_soft_gradient(route_balance):
    logits = torch.tensor([[1., -1.], [1., -1.], [-1., 1.]],
                          dtype=torch.float64, requires_grad=True)
    scores = logits.softmax(-1)
    ids = torch.tensor([[0], [0], [1]])
    loss, imbalance = route_balance([scores], [ids])
    hard = torch.tensor([2/3, 1/3], dtype=torch.float64)
    expected = 2 * (hard * scores.mean(0)).sum()
    assert torch.allclose(loss, expected) and abs(imbalance - 4/3) < 1e-12
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def verify_followup_gerrymandering_is_not_balance(route_balance):
    scores = torch.tensor([[1., 0.], [1., 0.]] + [[.4, .6]] * 4,
                          dtype=torch.float64)
    ids = torch.tensor([[0], [0], [1], [1], [1], [1]])
    loss, imbalance = route_balance([scores], [ids])
    assert abs(float(loss) - 14/15) < 1e-12
    assert abs(imbalance - 4/3) < 1e-12
    balanced = torch.tensor([[.8, .2]] * 3 + [[.4, .6]] * 3,
                            dtype=torch.float64)
    balanced_ids = torch.tensor([[0], [0], [0], [1], [1], [1]])
    balanced_loss, balanced_imbalance = route_balance([balanced], [balanced_ids])
    assert abs(float(balanced_loss) - 1.) < 1e-12
    assert abs(balanced_imbalance - 1.) < 1e-12
    assert float(loss) < float(balanced_loss) and imbalance > balanced_imbalance


def verify_production_global_local_prefix_scope(route_balance):
    a = torch.tensor([[.8, .2]] * 2, dtype=torch.float64)
    b = torch.tensor([[.4, .6]] * 4, dtype=torch.float64)
    ids_a, ids_b = torch.zeros((2, 1), dtype=torch.long), torch.ones((4, 1), dtype=torch.long)
    global_loss, global_imbalance = route_balance([a, b], [ids_a, ids_b], 'global')
    local_loss, local_imbalance = route_balance([a, b], [ids_a, ids_b], 'local')
    prefix_loss, prefix_imbalance = route_balance([a, b], [ids_a, ids_b], 'prefix')
    assert abs(float(global_loss) - 44/45) < 1e-12
    assert abs(float(local_loss) - 1.4) < 1e-12
    assert abs(float(prefix_loss) - 56/45) < 1e-12
    assert abs(global_imbalance - 4/3) < 1e-12
    assert abs(prefix_imbalance - 4/3) < 1e-12
    assert abs(local_imbalance - 2.) < 1e-12  # mean of per-batch imbalances


def verify_production_gradient_and_empty_policy(route_balance):
    logits = torch.tensor([[.7, -.4], [-.5, .8], [-.4, .9]],
                          dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0], [1], [1]])
    objective = lambda x: route_balance([x.softmax(-1)], [ids])[0]
    value = objective(logits)
    (gradient,) = torch.autograd.grad(value, logits)
    epsilon = 1e-6
    plus, minus = logits.detach().clone(), logits.detach().clone()
    plus[0, 0] += epsilon
    minus[0, 0] -= epsilon
    numeric = (objective(plus) - objective(minus)) / (2 * epsilon)
    assert torch.allclose(gradient[0, 0], numeric, rtol=1e-6, atol=1e-8)
    empty = torch.empty((0, 2), dtype=torch.float64, requires_grad=True)
    zero, load = route_balance([empty], [torch.empty((0, 1), dtype=torch.long)])
    assert zero.shape == () and zero.detach().item() == 0. and load == 0.
    zero.backward()
    assert empty.grad is not None and empty.grad.numel() == 0


def verify_production_rejects_bad_probabilities_and_routes(route_balance):
    good = torch.tensor([[.6, .4]])
    ids = torch.tensor([[0]])
    bad = [
        ([torch.tensor([[.6, .5]])], [ids], 'global'),
        ([good], [torch.tensor([[2]])], 'global'),
        ([good], [torch.tensor([[0, 0]])], 'global'),
        ([good], [ids], 'mystery'),
        ([torch.tensor([[float('nan'), .4]])], [ids], 'global'),
    ]
    for args in bad:
        try:
            route_balance(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid balance input accepted')

# %%
from torch_judge import evaluate
if not evaluate('0098_load_balance_eval.py', route_balance):
    raise SystemExit(1)
