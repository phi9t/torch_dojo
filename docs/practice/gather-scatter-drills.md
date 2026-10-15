# Gather/scatter fundamentals

Seven short drills, 10–20 minutes each, for the indexing patterns that voxel
pooling ([0107](../../dojo/0107_voxel_pillar_pool.py)), BEV scatter
([0108](../../dojo/0108_bev_scatter.py)) and sparse dispatch
([0101](../../dojo/0101_sparse_dispatch.py)) are built from. Do them first, in
order. For each: run the `[drill]` cells and predict before you look, fill in
the one answer, then climb `baseline`, `followup`, `production`. Drill answers
sit in the header comments of each `_eval.py`.

| # | Drill | Function | What it isolates |
|---|---|---|---|
| 0114 | [Collisions are contributions](../../dojo/0114_scatter_sum.py) | `scatter_sum` | overwrite and buffered `+=` lose duplicates; `index_add` does not |
| 0115 | [Shares of the group](../../dojo/0115_group_shares.py) | `group_shares` | scatter totals, gather them back, normalize before multiplying |
| 0116 | [Flat addresses](../../dojo/0116_flat_addresses.py) | `ravel_index` | row-major addresses; bounds per axis so a coordinate cannot alias another cell |
| 0117 | [Compact and put back](../../dojo/0117_compact_with_inverse.py) | `compact_with_inverse` | `nonzero` and boolean indexing share an order; inverse with -1 |
| 0118 | [Group by key](../../dojo/0118_group_rows.py) | `group_rows` | `unique(dim=0)`, lexicographic order, no packed key |
| 0119 | [Pillars to image](../../dojo/0119_pillars_to_bev.py) | `pillars_to_bev` | view vs copy, permute, strides |
| 0120 | [The backward of a gather](../../dojo/0120_gather_rows.py) | `GatherRows` | gather forward, scatter-add backward |

## How they connect

- **0107** is 0117 → 0116 → 0118 → 0115 → 0114: mask the points, address the
  cells, group the addresses, weight each point, and sum into the groups.
- **0108** is 0114 + 0115 + 0119: scatter counts, gather the totals back, scatter
  the weighted features, then lay the pillars out as an image.
- **Backward** (0120) explains why the gradient of every gather above is a
  scatter-add, and the gradient of a scatter-add is a gather.

Go on to [0107](../../dojo/0107_voxel_pillar_pool.py) and
[0108](../../dojo/0108_bev_scatter.py) when all seven pass `production`.
