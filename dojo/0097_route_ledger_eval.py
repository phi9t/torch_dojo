"""Reference and staged checks for route demand versus admitted work."""
# %%
import torch

# %%
def route_ledger(route_ids, expert_to_rank, num_ranks, keep_mask=None):
    if (not isinstance(route_ids, torch.Tensor) or route_ids.ndim != 2
            or route_ids.dtype != torch.long or not isinstance(expert_to_rank, torch.Tensor)
            or expert_to_rank.ndim != 1 or expert_to_rank.dtype != torch.long
            or expert_to_rank.numel() == 0 or expert_to_rank.device != route_ids.device
            or type(num_ranks) is not int or num_ranks < 1):
        raise ValueError('need long route IDs [T,K], long expert owners [E], and R >= 1')
    experts = expert_to_rank.numel()
    if ((route_ids < 0).any() or (route_ids >= experts).any()
            or (expert_to_rank < 0).any() or (expert_to_rank >= num_ranks).any()):
        raise ValueError('expert and rank IDs must be in range')
    if route_ids.shape[1] > 1 and (route_ids.sort(dim=1).values[:, 1:] ==
                                    route_ids.sort(dim=1).values[:, :-1]).any():
        raise ValueError('a token cannot request one expert twice')
    if keep_mask is None:
        keep_mask = torch.ones_like(route_ids, dtype=torch.bool)
    if (not isinstance(keep_mask, torch.Tensor) or keep_mask.shape != route_ids.shape
            or keep_mask.dtype != torch.bool or keep_mask.device != route_ids.device):
        raise ValueError('keep mask must be boolean [T,K] on the same device')
    requested_expert = torch.bincount(route_ids.reshape(-1), minlength=experts)
    kept_expert = torch.bincount(route_ids[keep_mask], minlength=experts)
    requested_rank = torch.bincount(expert_to_rank[route_ids].reshape(-1), minlength=num_ranks)
    kept_rank = torch.bincount(expert_to_rank[route_ids[keep_mask]], minlength=num_ranks)

    def imbalance(counts):
        total = int(counts.sum())
        return 0. if total == 0 else float(counts.max()) * counts.numel() / total

    views = {'requested_expert': requested_expert, 'kept_expert': kept_expert,
             'requested_rank': requested_rank, 'kept_rank': kept_rank}
    return {**views,
            'dropped_expert': requested_expert - kept_expert,
            'dropped_rank': requested_rank - kept_rank,
            'imbalance': {key: imbalance(counts) for key, counts in views.items()}}

# %% [verifiers]
def verify_baseline_requested_counts(route_ledger):
    ids = torch.tensor([[0, 2], [2, 1], [0, 2]])
    owner = torch.tensor([0, 1, 0])
    got = route_ledger(ids, owner, 2)
    assert torch.equal(got['requested_expert'], torch.tensor([2, 1, 3]))
    assert torch.equal(got['requested_rank'], torch.tensor([5, 1]))


def verify_followup_kept_and_dropped(route_ledger):
    ids = torch.tensor([[0, 2], [2, 1], [0, 2]])
    keep = torch.tensor([[True, False], [False, True], [True, True]])
    got = route_ledger(ids, torch.tensor([0, 1, 0]), 2, keep)
    assert torch.equal(got['kept_expert'], torch.tensor([2, 1, 1]))
    assert torch.equal(got['dropped_expert'], torch.tensor([0, 0, 2]))
    assert torch.equal(got['kept_rank'], torch.tensor([3, 1]))
    assert torch.equal(got['dropped_rank'], torch.tensor([2, 0]))
    assert int(got['kept_expert'].sum() + got['dropped_expert'].sum()) == ids.numel()


def verify_production_four_imbalance_views(route_ledger):
    ids = torch.tensor([[0, 2], [0, 1], [0, 2], [0, 1]])
    keep = torch.tensor([[True, True], [False, True], [False, True], [False, True]])
    got = route_ledger(ids, torch.tensor([0, 1, 0]), 2, keep)
    # Expert demand [4,2,2], expert kept [1,2,2]. Rank demand [6,2], rank kept [3,2].
    expected = {'requested_expert': 1.5, 'kept_expert': 1.2,
                'requested_rank': 1.5, 'kept_rank': 1.2}
    for key, value in expected.items():
        assert abs(got['imbalance'][key] - value) < 1e-12


def verify_production_empty_and_validation(route_ledger):
    got = route_ledger(torch.empty((0, 2), dtype=torch.long), torch.tensor([0, 1]), 3)
    assert torch.equal(got['requested_rank'], torch.zeros(3, dtype=torch.long))
    assert all(value == 0. for value in got['imbalance'].values())
    bad = [
        (torch.tensor([[0, 0]]), torch.tensor([0, 1]), 2, None),
        (torch.tensor([[2]]), torch.tensor([0, 1]), 2, None),
        (torch.tensor([[0]]), torch.tensor([0, 2]), 2, None),
        (torch.tensor([[0]]), torch.tensor([0, 1]), 2, torch.tensor([[1]])),
    ]
    for args in bad:
        try:
            route_ledger(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid route ledger input accepted')

# %%
from torch_judge import evaluate
if not evaluate('0097_route_ledger_eval.py', route_ledger):
    raise SystemExit(1)
