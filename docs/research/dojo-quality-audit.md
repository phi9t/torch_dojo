# Dojo quality migration audit

Checked October 6, 2026. The dojo has 81 question/eval pairs. Before this migration, every pair had one answer cell; the median eval script had four verifiers, and 28 had three or fewer. All 81 pairs now have exactly one answer cell, an interactive check for each stage, and at least one eval verifier in each of `baseline`, `followup`, and `production`. The final audit counted 128 baseline, 146 follow-up, and 171 production verifiers; no stage exceptions were needed.

The groups below were migration batches, not claims that every item needs the same constraint. Each pair has a baseline that is independently useful, at least one meaningful changed constraint, and an explicit production contract tied to its algorithm. A production verifier checks observable behavior; memory or performance claims that cannot be safely enforced by a small in-process test received targeted source review.

| Batch | Exercises | Main contract gaps to inspect |
|---|---|---|
| A: scalar and tensor primitives | 0001–0004, 0007–0008, 0015–0022, 0024, 0037–0041, 0048–0050, 0059–0060 | Feature axes, reductions, finite gradients, dtype/device, parameter state, numerical extremes, noncontiguous input, and mutation policy. |
| B: attention and sequence models | 0005, 0009–0014, 0023, 0025–0028 | Mask meaning, all-masked rows, batch isolation, cache invariants, dropout state, output/gradient parity, and peak score memory. |
| C: training, sampling, and text | 0029–0036, 0061, 0067–0068, 0071, 0078–0080 | Seeded replay, edge probabilities, malformed input, large counts, streaming, optimizer state, and tie or endpoint semantics. |
| D: geometry and perception | 0042–0047, 0051–0054, 0062–0065, 0069–0070, 0075–0077 | Coordinate frames, boundary inclusion, degenerate geometry, finite values, inverse/round-trip invariants, sparse memory, and duplicate-index accumulation. |
| E: motion and evaluation | 0055–0058, 0073–0074 | Masks, unequal horizons, invalid trajectories, deterministic ties, paired resampling, matching policy, and evaluation unit. |
| F: authored module | 0081 | Parameter count, state registration, noncontiguous input, dtype/device, gradients, and a controlled comparison with its ungated residual baseline. |

The pilot pairs are [attention](../../dojo/0006_multihead_attention.py), [voxel pooling](../../dojo/0066_voxel_pool.py), and [CSV parsing](../../dojo/0072_csv_rows.py). The final review sampled memory-sensitive kernels including blocked pairwise IoU and K-means assignment, sliding-window and tiled attention, sparse voxel pooling, and chunked CSV parsing. Their source uses bounded pair or sequence blocks, occupied-voxel storage, or incremental chunk consumption as claimed. The large-input verifiers are smoke checks, not formal complexity proofs. Keep the existing [interactive-script ADR](../adr/0001-keep-dojo-exercises-as-interactive-scripts.md) in force.
