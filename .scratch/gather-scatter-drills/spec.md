# Gather/scatter fundamentals drills

Status: needs-triage

## Objective

Seven small, original question/eval pairs, `0114`–`0120`, that isolate the indexing patterns behind voxel pooling (`0107`), BEV scatter (`0108`), sparse dispatch (`0101`) and the planned submanifold conv (`0121`). Each takes 10–20 minutes, has one function (or one `autograd.Function`), and keeps the repo format: top-level `# %%` cells, `# %% [drill]` predict cells, one `# %% [answer]`, staged `check(...)` calls, and a neighbouring `_eval.py` with a standalone reference and `# %% [verifiers]`. Drill answers live in the eval header comments, as in `0077`.

## Drills

| # | Name | Function | Pattern |
|---|---|---|---|
| 0114 | Collisions are contributions | `scatter_sum(index, src, size)` | overwrite vs accumulate; `index_add` |
| 0115 | Shares of the group | `group_shares(group_ids, values, num_groups)` | scatter totals, gather them back, normalize before multiply |
| 0116 | Flat addresses | `ravel_index`, `unravel_index` | row-major flatten; bounds before keys (aliasing); int64 overflow |
| 0117 | Compact and put back | `compact_with_inverse(x, mask)` | mask, `nonzero`, aligned boolean indexing, inverse with -1 |
| 0118 | Group by key | `group_rows(keys)` | `unique(dim=0)`, lexicographic order, no packed key |
| 0119 | Pillars to image | `pillars_to_bev`, `bev_to_pillars` | view vs copy, permute, strides |
| 0120 | The backward of a gather | `GatherRows` (`autograd.Function`) | gather forward, scatter-add backward |

## Stages

Each pair has `baseline`, `followup` and `production` verifiers, kept small. Baseline is a hand-sized example. Followup adds generated cases against a dense or autograd oracle, empties, and order or permutation properties. Production adds `ValueError` validation, overflow and aliasing traps, gradients where relevant, and no input mutation. No performance gates.

## Out of scope

Sparse conv (see `submanifold-conv-dojo`), GPU atomics and determinism, distributed collectives, benchmarks.

## Acceptance

All seven references pass every stage. Each candidate stub loads and fails for an answer-related reason. `make eval` passes. README links the practice guide `docs/practice/gather-scatter-drills.md`; the problem table and the "102 exercises" count were already stale and are left alone. Do not edit unrelated untracked files. No `/tmp`.
