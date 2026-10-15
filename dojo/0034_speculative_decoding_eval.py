"""Three-stage reference for speculative acceptance and residual correction."""
# %%
import torch

# %%
def speculative_decode(target_probs, draft_probs, draft_tokens):
    p, q, tokens = target_probs, draft_probs, draft_tokens
    if (not isinstance(p, torch.Tensor) or not isinstance(q, torch.Tensor)
            or not isinstance(tokens, torch.Tensor) or p.ndim != 2 or q.shape != p.shape
            or p.shape[1] < 1 or tokens.shape != (p.shape[0],)
            or not p.is_floating_point() or not q.is_floating_point()
            or tokens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or p.device != q.device or tokens.device != p.device):
        raise ValueError("need probability matrices [K,V] and integer draft tokens [K]")
    if (not torch.isfinite(p).all() or not torch.isfinite(q).all()
            or (p < 0).any() or (q < 0).any()
            or not torch.allclose(p.sum(1), torch.ones(p.shape[0], device=p.device, dtype=p.dtype), atol=1e-6, rtol=0)
            or not torch.allclose(q.sum(1), torch.ones(q.shape[0], device=q.device, dtype=q.dtype), atol=1e-6, rtol=0)):
        raise ValueError("rows must be finite probability distributions")
    if ((tokens < 0) | (tokens >= p.shape[1])).any():
        raise ValueError("draft token outside vocabulary")
    if tokens.numel() and (q[torch.arange(len(tokens), device=q.device), tokens.long()] <= 0).any():
        raise ValueError("draft token must have positive proposal probability")
    accepted = []
    for i, t in enumerate(tokens.tolist()):
        chance = min(1., float(p[i, t]) / float(q[i, t]))
        if chance == 1 or (chance > 0 and torch.rand(()).item() < chance):
            accepted.append(t)
            continue
        residual = torch.clamp(p[i] - q[i], min=0)
        mass = residual.sum()
        if mass <= 0:
            raise ValueError("rejection requires positive residual mass")
        accepted.append(int(torch.multinomial(residual / mass, 1).item()))
        return accepted
    return accepted

# %% [verifiers]
def verify_baseline_perfect_proposal(speculative_decode):
    p = torch.tensor([[.2, .8], [.7, .3]])
    assert speculative_decode(p, p.clone(), torch.tensor([1, 0])) == [1, 0]


def verify_baseline_reject_and_stop(speculative_decode):
    p = torch.tensor([[0., 1.], [1., 0.]])
    q = torch.tensor([[1., 0.], [1., 0.]])
    assert speculative_decode(p, q, torch.tensor([0, 0])) == [1]


def verify_followup_zero_block_and_positive_residual(speculative_decode):
    assert speculative_decode(torch.empty((0, 2)), torch.empty((0, 2)), torch.empty(0, dtype=torch.long)) == []
    p = torch.tensor([[0., .25, .75]])
    q = torch.tensor([[1., 0., 0.]])
    for seed in range(40):
        torch.manual_seed(seed)
        assert speculative_decode(p, q, torch.tensor([0]))[0] in (1, 2)


def verify_followup_residual_distribution(speculative_decode):
    p = torch.tensor([[0., .25, .75]])
    q = torch.tensor([[1., 0., 0.]])
    torch.manual_seed(3)
    results = [speculative_decode(p, q, torch.tensor([0]))[0] for _ in range(300)]
    assert 195 < results.count(2) < 255, results.count(2)


def verify_production_validation(speculative_decode):
    valid = torch.tensor([[.5, .5]])
    cases = ((valid, valid, torch.tensor([2])),
             (torch.tensor([[-.1, 1.1]]), valid, torch.tensor([0])),
             (torch.tensor([[float('nan'), .5]]), valid, torch.tensor([0])),
             (torch.tensor([[.2, .2]]), valid, torch.tensor([0])),
             (valid, torch.tensor([[0., 1.]]), torch.tensor([0])),
             (valid, valid, torch.tensor([0, 1])))
    for args in cases:
        try:
            speculative_decode(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted malformed inputs {args}")


def verify_production_replay_and_immutability(speculative_decode):
    p = torch.tensor([[.2, .8], [.3, .7]])
    q = torch.tensor([[.8, .2], [.7, .3]])
    before = p.clone(), q.clone()
    tokens = torch.tensor([0, 0])
    torch.manual_seed(47)
    first = [speculative_decode(p, q, tokens) for _ in range(40)]
    torch.manual_seed(47)
    assert first == [speculative_decode(p, q, tokens) for _ in range(40)]
    assert torch.equal(p, before[0]) and torch.equal(q, before[1])

# %%
from torch_judge import evaluate
if not evaluate("0034_speculative_decoding_eval.py", speculative_decode):
    raise SystemExit(1)
