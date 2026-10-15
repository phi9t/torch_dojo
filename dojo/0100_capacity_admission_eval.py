"""Reference and staged checks for local MoE route admission."""
# %%
import numpy as np

# Drill 1: It prints `[1 0 2] [1 2 0]`. A stable sort keeps tied items in
# their original order, so sorting `-scores` puts route 0 before route 2, as the
# tie rule asks. Reversing an ascending stable sort also reverses the ties, so
# the later route 2 would take the last slot. Negate the scores rather than
# reversing the order.

# Drill 2: It prints `{2: 2, 7: 4} {2: 2, 7: 2}`. A rank's pool is the sum of
# its experts' private slots, `count * capacity`: rank 7 owns two experts, so
# 4 slots. Giving each rank only `capacity` would shrink the pool and drop
# routes that the private policy keeps.

# %%
def capacity_admission(expert_ids, scores, owner, capacity, policy="expert"):
    ids = np.asarray(expert_ids)
    values = np.asarray(scores)
    ranks = np.asarray(owner)
    if (ids.ndim != 2 or ids.shape[1] == 0 or ids.dtype.kind not in "iu"
            or values.shape != ids.shape or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
            or ranks.ndim != 1 or len(ranks) == 0 or ranks.dtype.kind not in "iu"
            or np.any(ranks < 0) or np.any(ids < 0) or np.any(ids >= len(ranks))
            or isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, (int, np.integer))
            or capacity < 0 or policy not in ("expert", "rank")):
        raise ValueError("need finite scores, valid expert ownership, and nonnegative capacity")
    keep = np.zeros(ids.size, dtype=bool)
    if policy == "expert":
        pools = ids.ravel()
        budgets = {e: capacity for e in range(len(ranks))}
    else:
        pools = ranks[ids.ravel()]
        unique, counts = np.unique(ranks, return_counts=True)
        budgets = dict(zip(unique, counts * capacity))
    # Stable sorting leaves lower flattened route positions first at ties.
    for pos in np.argsort(-values.ravel(), kind="stable"):
        pool = pools[pos]
        if budgets[pool] > 0:
            keep[pos] = True
            budgets[pool] -= 1
    return keep.reshape(ids.shape)

# %% [verifiers]
def _route_tally(ids, owner, keep):
    """Count requested, kept, dropped routes independently of admission."""
    expert_count = len(owner)
    requested_expert = np.bincount(ids.ravel(), minlength=expert_count)
    kept_expert = np.bincount(ids[keep], minlength=expert_count)
    _, dense_ranks = np.unique(owner, return_inverse=True)
    requested_rank = np.bincount(dense_ranks, weights=requested_expert,
                                 minlength=dense_ranks.max() + 1).astype(int)
    kept_rank = np.bincount(dense_ranks, weights=kept_expert,
                            minlength=dense_ranks.max() + 1).astype(int)
    return ((requested_expert, kept_expert, requested_expert - kept_expert),
            (requested_rank, kept_rank, requested_rank - kept_rank))


def verify_baseline_private_budget_and_scores(capacity_admission):
    ids = np.array([[0, 1], [0, 1], [0, 1]])
    scores = np.array([[.1, .3], [.9, .8], [.7, .4]])
    got = capacity_admission(ids, scores, np.array([0, 0]), 2)
    assert got.dtype == np.bool_ and np.array_equal(got, [[False, False], [True, True], [True, True]])


def verify_baseline_tie_favors_first_route(capacity_admission):
    got = capacity_admission(np.zeros((3, 1), dtype=int), np.ones((3, 1)),
                             np.array([0]), 1)
    assert np.array_equal(got.ravel(), [True, False, False])


def verify_followup_pool_idle_peer_capacity(capacity_admission):
    ids = np.array([[0], [0], [1]])
    scores = np.array([[.9], [.8], [.1]])
    owner = np.array([0, 0])
    assert np.array_equal(capacity_admission(ids, scores, owner, 1, "expert").ravel(),
                          [True, False, True])
    assert np.array_equal(capacity_admission(ids, scores, owner, 1, "rank").ravel(),
                          [True, True, False])


def verify_followup_requested_kept_dropped_ledger_and_inequality(capacity_admission):
    ids = np.array([[0], [0], [0]])
    scores = np.array([[.9], [.8], [.7]])
    owner = np.array([0, 0])
    private = capacity_admission(ids, scores, owner, 1, "expert")
    pooled = capacity_admission(ids, scores, owner, 1, "rank")
    assert np.array_equal(private.ravel(), [True, False, False])
    assert np.array_equal(pooled.ravel(), [True, True, False])
    private_expert, private_rank = _route_tally(ids, owner, private)
    pooled_expert, pooled_rank = _route_tally(ids, owner, pooled)
    assert all(np.array_equal(actual, expected) for actual, expected in zip(
        private_expert, ([3, 0], [1, 0], [2, 0])))
    assert all(np.array_equal(actual, expected) for actual, expected in zip(
        pooled_expert, ([3, 0], [2, 0], [1, 0])))
    assert all(np.array_equal(actual, expected) for actual, expected in zip(
        private_rank, ([3], [1], [2])))
    assert all(np.array_equal(actual, expected) for actual, expected in zip(
        pooled_rank, ([3], [2], [1])))
    assert pooled_rank[2].sum() < private_rank[2].sum()


def verify_followup_independent_rank_pools(capacity_admission):
    ids = np.array([[0, 2], [1, 2], [0, 3]])
    scores = np.array([[.2, .5], [.9, .4], [.8, .3]])
    # Rank IDs need not be dense or in sorted expert order.
    got = capacity_admission(ids, scores, np.array([7, 7, 2, 2]), 1, "rank")
    assert np.array_equal(got, [[False, True], [True, True], [True, False]])


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid route input accepted")


def verify_production_empty_zero_budget_and_no_mutation(capacity_admission):
    ids = np.array([[0, 1], [1, 0]], dtype=int)
    scores = np.array([[.3, .2], [.4, .1]])
    owner = np.array([4, 4], dtype=int)
    before = [a.copy() for a in (ids, scores, owner)]
    assert not capacity_admission(ids, scores, owner, 0).any()
    for policy in ("expert", "rank"):
        assert capacity_admission(np.empty((0, 2), dtype=int),
                                  np.empty((0, 2)), owner, 1, policy).shape == (0, 2)
    assert all(np.array_equal(a, b) for a, b in zip((ids, scores, owner), before))


def verify_production_validation(capacity_admission):
    ids = np.array([[0, 1]])
    scores = np.array([[.4, .8]])
    owner = np.array([0, 0])
    invalid = [
        (ids[:, 0], scores, owner, 1, "expert"),
        (ids, scores[:, :1], owner, 1, "expert"),
        (ids.astype(float), scores, owner, 1, "expert"),
        (np.array([[0, 2]]), scores, owner, 1, "expert"),
        (ids, np.array([[np.nan, .8]]), owner, 1, "expert"),
        (ids, scores, np.array([0., 1.]), 1, "expert"),
        (ids, scores, np.array([0, -1]), 1, "expert"),
        (ids, scores, owner, -1, "expert"),
        (ids, scores, owner, True, "expert"),
        (ids, scores, owner, 1, "unknown"),
    ]
    for args in invalid:
        _reject(lambda args=args: capacity_admission(*args))


# %%
from torch_judge import evaluate
if not evaluate("0100_capacity_admission_eval.py", capacity_admission):
    raise SystemExit(1)
