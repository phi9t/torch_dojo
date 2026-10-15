"""Three-stage reference for deterministic filtering and stochastic sampling."""
# %%
import math
import torch

# %%
def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("logits must be a nonempty floating vector")
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or torch.isneginf(logits).all():
        raise ValueError("logits need at least one finite value and no NaN/+inf")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if not isinstance(top_p, (float, int)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0,1]")
    if not isinstance(temperature, (float, int)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # Ordering on the original logits is identical to ordering after positive scaling.
    order = sorted(range(logits.numel()), key=lambda i: (-float(logits[i]), i))
    order = [i for i in order if math.isfinite(float(logits[i]))]
    if top_k:
        order = order[:top_k]
    values = torch.stack([logits[i] for i in order]).to(torch.float64) / temperature
    probs = torch.softmax(values, dim=0)
    if top_p < 1:
        count = int((torch.cumsum(probs, 0) < top_p).sum().item()) + 1
        order, probs = order[:count], probs[:count]
    chosen = torch.multinomial(probs, 1).item()
    return order[chosen]

# %% [verifiers]
def verify_baseline_top_k_and_tie(sample_top_k_top_p):
    x = torch.tensor([0., 4., 4.])
    for _ in range(5):
        assert sample_top_k_top_p(x, top_k=1) == 1
    for seed in range(20):
        torch.manual_seed(seed)
        assert sample_top_k_top_p(x, top_k=2) in (1, 2)


def verify_baseline_invalid_parameters(sample_top_k_top_p):
    for kwargs in ({"top_k": -1}, {"top_k": 1.2}, {"top_p": 0}, {"temperature": 0}):
        try:
            sample_top_k_top_p(torch.zeros(2), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {kwargs}")


def verify_followup_top_p_keeps_boundary_token(sample_top_k_top_p):
    # Softmax([log(3),log(2),log(1)]) = [1/2,1/3,1/6].
    x = torch.tensor([math.log(3), math.log(2), 0.])
    for seed in range(40):
        torch.manual_seed(seed)
        assert sample_top_k_top_p(x, top_p=.5) == 0
        torch.manual_seed(seed)
        assert sample_top_k_top_p(x, top_p=.8) in (0, 1)


def verify_followup_top_k_precedes_top_p(sample_top_k_top_p):
    x = torch.tensor([0., 1., 2., 3.])
    for seed in range(30):
        torch.manual_seed(seed)
        assert sample_top_k_top_p(x, top_k=2, top_p=.99) in (2, 3)


def verify_production_stability_and_immutability(sample_top_k_top_p):
    x = torch.tensor([10000., 10001., float('-inf')])
    before = x.clone()
    for seed in range(20):
        torch.manual_seed(seed)
        assert sample_top_k_top_p(x) in (0, 1)
    assert torch.equal(x, before)


def verify_production_replay_and_distribution(sample_top_k_top_p):
    x = torch.tensor([math.log(3), math.log(1)])
    torch.manual_seed(57)
    first = [sample_top_k_top_p(x) for _ in range(300)]
    torch.manual_seed(57)
    assert first == [sample_top_k_top_p(x) for _ in range(300)]
    assert 185 < first.count(0) < 265, first.count(0)


def verify_production_rejects_malformed_logits(sample_top_k_top_p):
    for x in (torch.empty(0), torch.zeros(2, 2), torch.tensor([float('nan'), 1.]),
              torch.tensor([float('inf'), 0.]), torch.tensor([float('-inf')])):
        try:
            sample_top_k_top_p(x)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted malformed {x}")

# %%
from torch_judge import evaluate
if not evaluate("0032_topk_sampling_eval.py", sample_top_k_top_p):
    raise SystemExit(1)
