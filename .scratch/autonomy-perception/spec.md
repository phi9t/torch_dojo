# Autonomy perception workouts

Status: resolved

## Approved scope

Implement the approved design and thirteen associated tickets. The design is
in design.md and tickets are in issues/. Those documents supply stage contracts,
independent-oracle requirements, and the task graph. User-approved public test
seams are exercise callables, get_eval_data/compare/replay/check/evaluate, saved
candidate execution, and Emacs/Jupyter cells. No new internal judge seam is needed.

## Exercise allocation and interfaces

Reserve these IDs for this integration. Metadata origin/voxel_size/grid_shape
are tensors [3], xyz order; grid_shape is int64. Core floating inputs share
float32/float64 dtype and device. Functions reject invalid metadata with ValueError.
Optional chunk_size is a positive integer. Defaults are full-size chunks.

- 0103 multi_camera_project: project_points(points, K, T_camera_from_ego,
  image_hw, point_valid, camera_valid, min_depth=0.001, chunk_size=None)
  returns (uv, depth, valid).
- 0104 bilinear_features: sample_features(features, uv, valid,
  feature_from_image, chunk_size=None) returns (samples, supported).
- 0105 masked_view_fusion: fuse_views(samples, valid, confidence,
  chunk_size=None) returns (fused, supported).
- 0106 point_decoration: decorate_points(points, attributes, point_valid,
  origin, voxel_size, grid_shape) returns (descriptors, inside).
- 0107 voxel_pillar_pool: pool_points(points, features, point_valid,
  origin, voxel_size, grid_shape, pillar=False) returns
  (indices, counts, means, inverse). Inverse IDs are global occupied-row IDs.
- 0108 bev_scatter: scatter_bev(indices, features, counts, batch_size,
  grid_shape) returns (bev, point_counts, occupied).
- 0109 perception_bev: perception_bev(scene) returns a dictionary of named
  intermediate outputs and final BEV. scene is a dictionary of named tensors
  matching the above interfaces plus confidence and LiDAR features. No imports
  of another exercise reference; local complete helpers or injected callables.
- Upgrade 0027 patch embedding; 0110 dense_feature_layout:
  dense_features(tokens, grid_hw, prefix_tokens=0) returns [B,C,Hf,Wf].
- 0111 position_grid: resize_positions(positions, old_hw, new_hw,
  prefix_tokens=0) returns positions with preserved prefixes. positions [B,L,C],
  bilinear interpolation with align_corners=False, no antialias; float32/float64
  core. Same-size identity is exact.
- 0112 frozen_feature_adapter: FrozenFeatureAdapter(backbone, grid_hw,
  prefix_tokens, in_channels, out_channels, feature_from_image). Backbone returns
  [B,L,C]; forward(images) returns (maps, feature_from_image expanded to [B,3,3]).
  Freeze/eval backbone, preserve requested input gradients, trainable 1x1 adapter;
  train(mode) preserves backbone eval while adapting adapter state.
- 0113 bev_cross_attention: bev_attention(queries, image_features,
  reference_points, K, T_camera_from_ego, image_hw, camera_valid,
  feature_from_image, min_depth=0.001, chunk_size=None) returns
  (updates, supported, intermediates). queries [B,Q,C], reference_points [B,Q,A,3].
  Project and sample anchors, attend over M*A keys using scaled dot product
  without learned projections; all-masked update is zero. Masks/weights/samples
  remain inspectable. A=1 baseline; explicit multiple heights follow-up.

B,M,C,Hf,Wf are positive in the camera core; N may be zero. Support uses integer
pixel-center inclusive bounds. BEV physical bounds are half-open. Invalid outputs
are zero with masks. Projection means projectable, never verified unoccluded.
Backbone and position/layout exercises may support empty batch as ticketed.

## Implementation and qualification

Every ticket produces a complete practice slice and records red/green evidence
through the approved seams. Use top-level scripts and reference-free verifier
contexts. Add complete reference versions and named EvalCase data. Property
trials and backward checks supplement analytic/scalar independent oracles.

Integration branch: codex/autonomy-perception-integration. Each implementer owns
one worktree/branch, starts from integration, and merges its latest tip before
reporting. Merge completed work with a merger agent. Qualification tickets check
actual Emacs/Jupyter and saved-file isolation; final two-axis code review covers
all changes. Resolve every ticket with evidence and remove owned worktrees.
No remote push. Never use /tmp; scratch/runtime/TMPDIR live under project target.
Preserve unrelated untracked research notes.

## Completion

All thirteen tickets are resolved. The implementation adds eleven autonomy
question/eval pairs and upgrades patch embedding with progressive references.
Core, vision, and review-fix qualification ran actual Emacs/Jupyter practice
loops (29, 23, and 14 cells respectively), saved-candidate isolation, independent
oracles, and backward checks. Both independent review axes have zero remaining
findings after c5d9309.

Final required gates on merged code: all 113 reference scripts passed (exit 0);
49 repository tests passed with seven optional skips (exit 0). Full logs remain
under target/autonomy-perception/final-eval.log and final-test.log. All fifteen
owned worktrees were removed and the owned kernel exited successfully. Local
main receives the integration by fast-forward; no remote push is performed.
