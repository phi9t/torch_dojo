# Submanifold sparse convolution dojo workout

Status: resolved

Blocked by: gather-scatter-drills (drills 0114–0120 come first; this workout was renumbered from 0114 to 0121)

## Objective

Add one original dojo question/eval pair, `0121_submanifold_conv`, teaching submanifold sparse convolution as a rulebook lookup plus per-offset gather, matrix multiply and add. Keep the existing script workflow: top-level `# %%` cells, one `# %% [answer]`, staged `check(...)` calls, and a neighboring `_eval.py` with a standalone reference plus `# %% [verifiers]`. Judge machinery stays outside `dojo/`. Follows `0022_conv2d`, `0107_voxel_pillar_pool` and `0108_bev_scatter`; `0101_sparse_dispatch` is the nearest relative.

## Contract

Two functions in one answer cell. The answer may not call `F.conv*`; dense conv is the eval oracle only.

```python
def build_rulebook(coords, spatial_shape, batch_size, kernel_size=3) -> Tensor  # int64 [O, K]
def apply_rulebook(features, weight, rulebook) -> Tensor                        # [K, Cout]
```

- `coords` int64 `[K, 1+D]` = `[batch, *spatial]`, any order, unique. `D` inferred from `coords` (2 in baseline, 3 from followup). `features` float32/64 `[K, Cin]`. `weight` `[O, Cin, Cout]` with `O = kernel_size**D`, offsets row-major over `(-1..1)^D`, cross-correlation (no flip).
- `rulebook[o, k]` is the input row at offset `o` from site `k`, or `-1` if that site is not active. Offset 0 maps every site to itself.
- Output row `k` corresponds to input row `k`: same order, same coordinates, none added.
- Submanifold: no dilation. Inactive neighbours contribute zero; an active site with an all-zero feature is still a row.
- Per offset the map is injective, so no collisions and no atomics (contrast `0108`).

## Stages

| Stage | Scope | Checks |
|---|---|---|
| baseline | 2D, 3x3, loop over offsets | output equals dense conv sampled at active sites; rulebook correct on a hand-sized fixture |
| followup | 3D, batches, empty input, gradients | batch separation; input row order preserved; asymmetric weights; `d/dfeatures` and `d/dweight` match the dense oracle |
| production | validation and limits | key aliasing; int64 key overflow; duplicates; out-of-range coords; strided features; dtypes; sizing |

Each stage opens with a predict-first cell: (1) a hole in a row of active sites, does submanifold conv output anything there; (2) a zero-feature site is still an active row; (3) a site at `x = X-1` beside one at `(y+1, x = 0)` must not be neighbours under a flat key.

## Traps the verifiers must catch

- Flipped kernel or transposed offset order (asymmetric weights).
- Key aliasing: a neighbour at `x = X`, `x = -1`, or outside `y`, `z`, batch range must be rejected before a key is formed.
- Batch bleed: same spatial coordinate in two batches.
- Duplicate coordinates: `ValueError`.
- Out-of-range coordinates, and `B * prod(spatial_shape)` overflowing int64: `ValueError` before any lookup, with no dense allocation.
- Unsorted input: output order follows input order.
- Zero feature vector treated as empty.

## Production framing (explore, not gated)

- Sizing cell: rulebook `O * K * 4` bytes int32 (about 43 MB at K = 400k for 3x3x3, double with a reverse map) against the feature payload.
- Optional: custom `autograd.Function` that recomputes gathers in backward instead of saving per-offset slices.
- Speed claims are specific to a measured workload, not a correctness gate.

## Out of scope

Regular strided sparse conv (output active set `o = (i - d) / s` where divisible; candidate follow-on `0122`), Triton/CUDA kernels, hash tables (sorted-key `searchsorted` is enough), bias, distributed execution, performance gates.

## Open decisions

1. 2D then 3D with the same code (recommended) versus separate workouts.
2. Include the custom-backward challenge, or leave as a note.

## Acceptance

Reference passes baseline/followup/production; candidate stub loads and fails for an answer-related reason; `make eval` and `make test` pass; README links the pair and its count is updated. Do not edit unrelated untracked files. No `/tmp`.

## Comments

- 2026-10-11: All four submanifold convolution tickets are resolved. The `0121_submanifold_conv` question/eval pair is published in the README catalog, its staged reference passes baseline/followup/production, the saved starter candidate loads and fails for answer-stub reasons, and repository `eval` and `test` gates pass from the ticket worktree.
