"""Gradient accumulation reference and cumulative checks."""
# %%
import copy
import torch
import torch.nn as nn

# %%
def accumulated_step(model, optimizer, loss_fn, micro_batches):
    optimizer.zero_grad()
    total_examples = 0
    total_loss = 0.0
    for x, y in micro_batches:
        if x.ndim == 0 or y.ndim == 0 or x.shape[0] == 0 or x.shape[0] != y.shape[0]:
            raise ValueError("each microbatch must have matching positive size")
        count = x.shape[0]
        loss = loss_fn(model(x), y)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar per-example mean")
        (loss * count).backward()
        total_loss += loss.item() * count
        total_examples += count
    if total_examples == 0:
        raise ValueError("micro_batches must contain at least one batch")
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.div_(total_examples)
    optimizer.step()
    return float(total_loss / total_examples)

# %% [verifiers]
def _compare_to_full_batch(accumulated_step, sizes, *, seed=7):
    torch.manual_seed(seed)
    model = nn.Linear(3, 2, dtype=torch.float64)
    full = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.4)
    ref_opt = torch.optim.SGD(full.parameters(), lr=0.1, momentum=0.4)
    batches = [(torch.randn(n, 3, dtype=torch.float64), torch.randn(n, 2, dtype=torch.float64)) for n in sizes]
    loss_fn = nn.MSELoss()
    actual = accumulated_step(model, opt, loss_fn, iter(batches))
    x = torch.cat([batch[0] for batch in batches])
    y = torch.cat([batch[1] for batch in batches])
    ref_opt.zero_grad()
    expected = loss_fn(full(x), y)
    expected.backward()
    ref_opt.step()
    assert isinstance(actual, float) and abs(actual - expected.item()) < 1e-12
    for p, q in zip(model.parameters(), full.parameters()):
        assert torch.allclose(p, q, atol=1e-12)
    return model, full, opt, ref_opt


def verify_baseline_equal_batches(accumulated_step):
    _compare_to_full_batch(accumulated_step, [2, 2])


def verify_baseline_single_batch(accumulated_step):
    _compare_to_full_batch(accumulated_step, [4])


def verify_followup_unequal_microbatches(accumulated_step):
    _compare_to_full_batch(accumulated_step, [1, 3, 2])


def verify_followup_mean_not_batch_mean(accumulated_step):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [(torch.zeros(1, 1), torch.ones(1, 1)),
               (torch.zeros(3, 1), torch.ones(3, 1) * 3)]
    value = accumulated_step(model, opt, nn.MSELoss(), batches)
    assert abs(value - 7.0) < 1e-6  # (1 + 3*9)/4


def verify_production_one_pass_and_empty(accumulated_step):
    model = nn.Linear(3, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]
    try:
        accumulated_step(model, opt, nn.MSELoss(), iter(()))
    except ValueError:
        pass
    else:
        raise AssertionError("empty microbatch stream should fail")
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))
    _compare_to_full_batch(accumulated_step, [3, 1, 4])


def verify_production_repeated_steps(accumulated_step):
    torch.manual_seed(23)
    model = nn.Linear(3, 2, dtype=torch.float64)
    full = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.6)
    ref_opt = torch.optim.SGD(full.parameters(), lr=0.05, momentum=0.6)
    loss_fn = nn.MSELoss()
    for sizes in ([1, 4], [3, 2, 1]):
        batches = [(torch.randn(n, 3, dtype=torch.float64), torch.randn(n, 2, dtype=torch.float64)) for n in sizes]
        actual = accumulated_step(model, opt, loss_fn, iter(batches))
        ref_opt.zero_grad()
        expected = loss_fn(full(torch.cat([x for x, _ in batches])), torch.cat([y for _, y in batches]))
        expected.backward()
        ref_opt.step()
        assert abs(actual - expected.item()) < 1e-12
        for p, q in zip(model.parameters(), full.parameters()):
            assert torch.allclose(p, q, atol=1e-12)

# %%
from torch_judge import evaluate
if not evaluate("0031_gradient_accumulation_eval.py", accumulated_step):
    raise SystemExit(1)
