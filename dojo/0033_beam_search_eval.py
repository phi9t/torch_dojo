"""Three-stage reference for bounded beam search with completed paths."""
# %%
import math
import torch

# %%
def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    if (isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 1
            or isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1
            or isinstance(start_token, bool) or not isinstance(start_token, int) or start_token < 0
            or isinstance(eos_token, bool) or not isinstance(eos_token, int) or eos_token < 0):
        raise ValueError("limits and token IDs must be valid integers")
    beams = [(0.0, [start_token])]
    for _ in range(max_len - 1):
        if all(seq[-1] == eos_token for _, seq in beams):
            break
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_token:
                candidates.append((score, seq))
                continue
            values = log_prob_fn(torch.tensor(seq, dtype=torch.long))
            if (not isinstance(values, torch.Tensor) or values.ndim != 1 or values.numel() == 0
                    or not values.is_floating_point() or torch.isnan(values).any() or torch.isposinf(values).any()):
                raise ValueError("log_prob_fn must return a log-probability vector")
            if start_token >= values.numel() or eos_token >= values.numel():
                raise ValueError("token ID outside vocabulary")
            viable = [j for j in range(values.numel()) if math.isfinite(float(values[j]))]
            viable.sort(key=lambda j: (-float(values[j]), j))
            for j in viable[:beam_width]:
                candidates.append((score + float(values[j]), seq + [j]))
        if not candidates:
            break
        candidates.sort(key=lambda item: (-item[0], item[1]))
        beams = candidates[:beam_width]
    return list(min(beams, key=lambda item: (-item[0], item[1]))[1])

# %% [verifiers]
def verify_baseline_greedy_and_length(beam_search):
    def next_token(seq):
        values = torch.full((5,), float('-inf'))
        values[min(len(seq), 4)] = 0.
        return values
    assert beam_search(next_token, 0, 5, 1, 4) == [0, 1, 2, 3, 4]
    assert beam_search(next_token, 0, 1, 2, 4) == [0]


def verify_baseline_wider_beam_finds_path(beam_search):
    def path(seq):
        values = torch.full((4,), float('-inf'))
        if len(seq) == 1:
            values[1], values[2] = -.6, -.4
        else:
            values[3] = -.1 if seq[-1] == 1 else -2.
        return values
    assert beam_search(path, 0, 3, 2, 3) == [0, 1, 3]


def verify_followup_completed_beam_survives(beam_search):
    calls = []
    def path(seq):
        calls.append(seq.tolist())
        values = torch.full((3,), float('-inf'))
        if len(seq) == 1:
            values[2], values[1] = -.2, -.1
        else:
            values[1] = -1.
        return values
    assert beam_search(path, 0, 5, 2, 2) == [0, 2]
    assert [0, 2] not in calls


def verify_followup_ties_prefer_lexicographic_path(beam_search):
    def equal(seq):
        return torch.tensor([float('-inf'), 0., 0.])
    assert beam_search(equal, 0, 3, 2, 2) == [0, 1, 1]


def verify_production_invalid_input(beam_search):
    def valid(seq):
        return torch.zeros(3)
    for args in ((0, 0, 1, 2), (0, 2, 0, 2), (-1, 2, 1, 2), (0, 2, 1, 3)):
        try:
            beam_search(valid, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {args}")
    for values in (torch.tensor([float('nan'), 0.]), torch.tensor([float('inf'), 0.]), torch.zeros(2, 2)):
        try:
            beam_search(lambda _: values, 0, 2, 1, 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted scores {values}")


def verify_production_all_impossible_and_termination(beam_search):
    calls = []
    def impossible(seq):
        calls.append(seq.tolist())
        return torch.full((2,), float('-inf'))
    assert beam_search(impossible, 0, 100, 4, 1) == [0]
    assert len(calls) == 1
    def immediate(seq):
        calls.append(seq.tolist())
        return torch.tensor([float('-inf'), 0.])
    assert beam_search(immediate, 0, 100, 1, 1) == [0, 1]
    assert calls[-1] == [0]

# %%
from torch_judge import evaluate
if not evaluate("0033_beam_search_eval.py", beam_search):
    raise SystemExit(1)
