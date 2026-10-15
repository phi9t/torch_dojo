# Autonomy perception track

Status: implemented and qualified

## Objective

Prioritize vision foundation models, LiDAR/voxel/BEV processing, and
multi-view projection/fusion in the dojo.

## Settled decisions

The user accepted these recommendations in the first design discussion:

- Use shared camera/LiDAR scenes and coordinate conventions, while keeping
  each question/eval pair independently runnable with its own fixtures.
- Begin vision foundation model preparation with dense features and
  adaptation for geometry and fusion; add pretraining objectives afterward.
- Keep core workouts runnable on CPU with small deterministic PyTorch
  fixtures; offer optional CUDA scale and mixed-precision experiments.

Existing practice contracts remain: top-level Emacs/Jupyter cells, one editable
answer section, cumulative baseline/follow-up/production stages, complete
runnable reference progressions, and inspectable eval data with fresh replay.
Infrastructure lives outside dojo.

## Existing foundations

Patch embedding (0027), pixel-center mapping (0042), camera projection (0044),
BEV indexing (0046), camera unprojection (0051), pose inversion (0053), LiDAR
polar coordinates (0054), temporal transforms (0065), and voxel pooling (0066).
Extend or deepen existing exercises where their contracts fit; avoid duplicate
introductions.

The user confirmed the second round's recommendations:

- Start with LiDAR-guided image-feature fusion into BEV. Camera depth lifting
  becomes a later branch.
- Use deterministic synthetic scenes for mandatory evals. Real data and
  pretrained checkpoints are optional exploration.
- Target a 30–45 minute baseline per workout. Deeper stages run separately.

## Learning outcome and scope

The learner can construct, inspect, differentiate, and debug a camera/LiDAR
feature pipeline through a BEV representation. They can explain how geometry,
missing observations, collisions, numerical precision, and memory affect it.

This is a multi-workout curriculum extension. It reuses the judge and eval-data
interfaces, and deepens relevant existing prerequisites. The first batch has
six new question/eval pairs and a small integration exercise. The next batch
connects vision foundation model features to the same geometry contracts.

The core uses supplied feature tensors, not a required pretrained model. This
keeps geometry failures inspectable. Optional checkpoint exploration later
uses an explicit feature adapter; it must document spatial layout and
preprocessing rather than assume all backbones expose the same feature grid.

## Alternatives and chosen progression

| Approach | Benefit | Cost | Decision |
|---|---|---|---|
| Independent topic bank | Easy to add individual exercises | Frame and layout mismatches remain hidden | Keep independent execution, share scene conventions |
| Camera depth lifting first | Introduces camera-only BEV reasoning | Adds depth-distribution ambiguity before projection is secure | Later branch |
| LiDAR-guided camera features into BEV | Known 3D support makes projection, sampling, and pooling debuggable | Initially depends on LiDAR coverage | First batch |

## Shared scene contract

Each pair owns its fixtures and can run from dojo without executing another
question or importing another pair's reference. Scene generation and reusable
fixture support, if needed, live outside dojo. Each eval retains its own
independent oracle and problem-specific verifiers.

Use one right-handed ego frame: x forward, y left, z up, in meters. Camera
coordinates use x right, y down, z forward. Name transforms by direction:
`T_camera_from_ego`. Store points as rows; transform them as `points @ R.T + t`.
Do not silently reuse a world transform as an ego transform. The rotation
between these frames must be explicit in fixtures.

The common batched shapes are:

| Value | Shape / meaning |
|---|---|
| points | `[B,N,3]`, ego coordinates |
| point_valid | `[B,N]`, excludes padding |
| point_features | `[B,N,F]`, supplied LiDAR attributes |
| intrinsics | `[B,M,3,3]`, pinhole matrices |
| camera_from_ego | `[B,M,4,4]`, rigid transforms |
| camera_valid | `[B,M]`, available cameras |
| image_hw | `[B,M,2]`, positive image height and width |
| image_features | `[B,M,C,Hf,Wf]`, common feature-map size within a batch |
| feature_from_image | `[B,M,3,3]`, explicit affine pixel-coordinate mapping |
| projected_uv / depth / valid | `[B,M,N,2]` / `[B,M,N]` / `[B,M,N]` |
| sampled_features | `[B,M,N,C]` |
| fused_features / supported | `[B,N,C]` / `[B,N]` |

Baseline fixtures set B=M=1 without changing these axes. Follow-ups introduce
multiple batches, cameras, unequal valid point counts, and unavailable views.
N=0 is valid; B, M, C, Hf, and Wf are positive. Common feature-map sizes are an
intentional first-batch limit; ragged camera feature maps are a later extension.

Pixel coordinates identify integer pixel centers. Projection support uses
`0 <= u <= W-1`, `0 <= v <= H-1`, and `depth > min_depth`, where min_depth is
finite and positive. This is a center-coordinate sampling contract. It must
not be confused with BEV's half-open physical bounds.

Image-to-feature coordinates use the supplied affine transform, which can
encode crop, resize, padding, or an explicitly chosen feature-grid convention.
Do not infer patch-center or receptive-field alignment from tensor sizes.
The sampler also enforces support within the feature grid.

Floating core inputs use a common device and float32 or float64. Metadata,
matrix structure, shapes, and finite valid values are validated explicitly.
Masked padding can contain NaNs and must be sanitized before arithmetic.
Missing cameras have finite calibration placeholders; their features may be
nonfinite and must not contribute. Invalid outputs use zeros plus a boolean
mask. A genuine zero feature remains distinguishable from missing support.

Projection validity means geometric sampling support. No occlusion claim is
made without depth evidence. An optional depth-map follow-up states its
comparison tolerance and which observations it can reject.

## First batch: six workouts

Each pair keeps one editable answer section, named inspectable cases, three
cumulative stages, and complete reference versions in separate eval cells.
Number new pairs using the next available four-digit IDs at implementation
time; do not reserve IDs that concurrent work may consume.

### 1. Multi-camera projection with validity

Return uv, camera-z depth, and projectable masks. Mask padded points, unavailable
cameras, depth at/below min_depth, and out-of-image projections. Wrong shapes
and invalid calibration raise ValueError; behind-camera points are ordinary
masked observations, unlike the stricter existing 0044 API.

- Baseline: one camera, hand-computed axes, translations, and image edges.
- Follow-up: multiple rotated cameras and batches; all-invalid and empty points.
- Production: noncontiguous inputs, masked nonfinite padding, no mutation,
  chunked point processing, and valid-point/calibration gradient checks.
- References: explicit per-camera transform; broadcasted transform; point chunks.
  Chunking bounds temporary storage but cannot remove the required M-by-N output.

### 2. Bilinear sampling of camera features

Map projected image coordinates through feature_from_image and return sampled
features plus refined support. Bilinear interpolation handles integer centers,
fractional centers, and the last row/column without duplicate-weight errors.
Outside coordinates return zero and false; interpolation never blends padding
from outside the image into a purported valid sample.

- Baseline: a numbered grid and analytic affine-valued features.
- Follow-up: non-square grids, multiple channels/views, width or height one.
- Production: gradients through features and interior coordinates; masked NaNs;
  noncontiguous inputs; bounded temporary work for point chunks.
- References: explicit four-corner interpolation; vectorized gathers; optional
  grid_sample implementation with its normalized-coordinate convention derived.
  Reference parity is checked only under matching boundary semantics.

### 3. Masked multi-view feature fusion

Return a weighted mean across valid sampled views plus supported. A supplied
nonnegative confidence tensor has shape `[B,M,N]`. Zero total eligible weight
returns zero features and false, even if some views are geometrically valid.
Weights need not already sum to one. Missing entries contribute neither values
nor normalization mass.

- Baseline: equal-weight mean and one available camera.
- Follow-up: confidence weights, camera permutations, zero weights, all missing.
- Production: widened accumulation, rescaling weights to avoid sum overflow,
  nonfinite masked values, and gradients through eligible features/weights.
- References: transparent masked reduction; stable normalized reduction;
  chunked points. Learned confidence and attention belong to a later workout.

### 4. LiDAR point-feature decoration

Given fixed voxel origin, size, and integer grid shape, return inside-grid
validity and point descriptors in one documented channel order: supplied
attributes, ego xyz, offset from the occupied voxel's point centroid, offset
from the voxel center, and Euclidean range. Invalid rows are zero. Centroids
include exactly the valid inside-grid points; repeated points contribute.

- Baseline: work one voxel's centroid and offsets by hand.
- Follow-up: multiple voxels/batches, anisotropic size, duplicates, permutations.
- Production: stable accumulation, sparse occupied-cell grouping, empty input,
  gradient checks through coordinates away from cell boundaries.
- References: per-voxel teaching loop; grouped scatter reductions; reusable
  grouping within a call. Outputs retain original point order.

### 5. Differentiable voxel/pillar feature pooling

Accept coordinates, point features, point_valid, and fixed grid metadata. Return
sorted occupied `[batch,z,y,x]` indices, counts, mean features, and inverse IDs
with -1 for excluded points. Pillar mode collapses z grouping explicitly.
Do not mix batches or overwrite collisions. Preserve zero-valued occupied cells.

- Baseline: mean/count pooling and a hand-computed inverse map.
- Follow-up: empty inputs, batch separation, voxel versus pillar grouping,
  permutation invariance, and half-open maximum boundaries.
- Production: differentiable features, widened mean accumulation, sparse storage
  proportional to points/occupied cells, integer-key overflow checks if packing.
- References: clear grouping loop; unique plus scatter; sorted segment reduction.
  Torch extends existing NumPy 0066; its earlier API remains unchanged.

### 6. Collision-safe BEV scatter

Consume occupied indices and pooled features/counts. Return dense `[B,C,H,W]`,
point-count `[B,1,H,W]`, and occupied `[B,1,H,W]` maps. Map y to H and x to W;
collapse z by count-weighted mean so results equal direct point-to-pillar means.
Validate indices and positive counts. Empty input gives zero/false maps.

- Baseline: one cell and a numbered asymmetric grid.
- Follow-up: collisions across z, unequal counts, multiple batches, empty input.
- Production: accumulate repeated indices; widened numerator/denominator;
  feature gradients; validate dimensions before allocating the dense output.
- References: explicit accumulator; flattened index_add; grouped reduction.
  Dense output inherently costs O(B*C*H*W); it must not be labeled sparse.

## Integration exercise

After the six pairs, add a seventh, small integration pair. Its answer composes
helpers in the same answer/setup sections or consumes supplied stage outputs;
it never imports learner answers from numbered question scripts. Keep fixtures
and scalar oracle expectations local to its eval context.

Pipeline: projection -> feature sampling -> view fusion -> concatenate supplied
LiDAR and image features -> voxel pooling -> count-weighted BEV scatter.

Retain every intermediate output in a named result so the learner can locate a
bad transform, unsupported point, lost collision, or wrong axis. Points without
image support keep their LiDAR features and a separate image-support indicator.
Point decoration is an optional richer input branch, not a hidden requirement
of the basic composition.

Acceptance examples:

- An asymmetric hand-built scene lands known point/image features in known cells.
- Reordering points or cameras leaves the fused BEV unchanged within tolerance.
- Duplicates and unequal z-cell counts preserve direct point-mean equivalence.
- Missing image support preserves LiDAR occupancy; unavailable cameras cannot
  contaminate features with NaNs.
- Feature gradients reach both eligible camera maps and LiDAR features.
- Fresh replay after replacing a faulty learner helper passes while the old
  EvalResult remains inspectable.

## Vision foundation model batch

Prioritize this immediately after the first fusion batch. Upgrade 0027 patch
embedding with complete reference versions and inspectable cases as its entry.

1. **Dense visual feature layout:** reconstruct `[B,C,Hf,Wf]` from spatial tokens;
   remove an explicitly specified prefix-token count; preserve rectangular
   ordering and gradients. Do not infer spatial positions from sqrt(token_count).
2. **Positional grid interpolation:** separate prefix and spatial positions;
   interpolate rectangular grids under a declared convention; test identity,
   gradients, dtype, and channel layout.
3. **Frozen-backbone feature adaptation:** use a tiny supplied backbone and
   trainable adapter; verify train/eval state, frozen parameter behavior, and
   gradients to the adapter. Distinguish freezing weights from disabling all
   input gradients. Optional real checkpoints use the same explicit adapter seam.
4. **Geometry-guided BEV cross-attention:** project supplied 3D BEV reference
   points, sample camera features, then attend over eligible views. Specify all-
   masked behavior and test geometry plus backward together. Ground-plane
   queries and multiple height anchors are separate stated stages.

These extend the core's feature maps without assuming a universal vision
backbone. Select checkpoint-specific exercises only after checking primary
implementation sources and documenting the exact feature contract.

## Later branches

- Camera depth-distribution lifting and depth-weighted BEV splatting.
- Ego-motion BEV warping and temporal fusion with stale/missing observations.
- Explicit depth-based occlusion tests and calibration perturbation diagnostics.
- Masked-image or self-distillation pretraining objectives, grounded in selected
  primary sources rather than an invented common model contract.
- Sparse convolution, distributed processing, and specialized kernels after
  portable correctness and workload measurements are established.

## Gradients, precision, and resource contracts

Check float64 central differences or gradcheck on smooth interior examples,
away from depth cutoffs, pixel knots, masks, and voxel boundaries. Hard floor,
membership, sorting, and indices are discrete; no coordinate derivative is
promised through an assignment change. With fixed membership, centroid/offset
and feature operations retain their ordinary derivatives.

Core correctness runs use float32/float64 on CPU. Low-precision CUDA variants
are optional and declare accumulator dtype, output dtype, error tolerances,
and reduction determinism. GPU scatter may change rounding with execution order.
Numerical parity uses justified tolerances, not bit equality for reductions.

Each reference states its highest qualified stage and shared comparison domain.
Later versions can change working storage or stability without changing public
outputs. Any changed semantics become a new explicit follow-up contract.
The production stage measures only the stated reusable implementation scope.

Experiments report workload dimensions, device/dtype, warmup, synchronization,
repeats, variability, and numerical error. Separate analytical storage estimates,
measured allocator peaks, and autograd-retained state. No speed threshold is a
correctness gate. Use structural resource checks only for declared memory limits.

## Eval experience and decisive cases

Every pair exposes named EvalCase fixtures through make_eval_data. Question
cells bind eval_data and demonstrate direct calls, compare, retained failures,
selective fresh replay, and cumulative check. Reference code never enters the
learner verifier namespace. Imports remain at script top level.

Include analytic constants/ramps, 90-degree camera poses, asymmetric grids,
projection and cell boundaries, missing cameras, all-invalid inputs, duplicate
indices, noncontiguous tensors, and masked NaN padding. Independent hand/scalar
oracles establish expected outputs; optimized references do not serve as their
own sole oracle. Property trials exercise permutations, round trips where
defined, conservation of counts, and separation between batches.

Hypothesis belongs in eval verifiers for bounded generated cases. Concrete
fixtures remain retrievable without executing property trials. Make randomized
failures reproducible and reduce them to an inspectable example.

Give learners prediction probes and deliberate broken-example challenges,
such as swapped xy axes, reversed transforms, and duplicate-index overwrite.
Optional plots inspect points, projected samples, support, and BEV maps from
the kernel. Plots add no mandatory dependency to saved-file evaluation.

## Delivery and exit criteria

1. Write the buildable spec from this design, with exact function signatures
   and per-stage acceptance criteria; reserve exercise IDs against current main.
2. Split tickets into prerequisite upgrades, each core pair, integration,
   qualification/docs, and the immediate vision-feature follow-up batch.
3. Implement dependency-ready tickets, preserving independent pair execution.
4. Qualify every reference against its advertised stage, saved answer loading,
   isolation from teaching/reference cells, and an actual Emacs/Jupyter
   wrong-answer -> inspect -> revise -> replay session.
5. Review Standards and Spec, then land locally on main when implementation is
   authorized. Preserve concurrent work and use project-local scratch paths.

Design review: the six approved choices are captured; existing exercise APIs
remain compatible; core fixtures require no dataset/checkpoint downloads;
projection support and occlusion are distinct; feature-grid alignment is
explicit; count-weighted pooling and gradients are defined; later branches are
separated from first-batch acceptance. No implementation has started.
