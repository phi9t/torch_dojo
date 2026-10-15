# 07: Inspect camera and LiDAR fusion through BEV

**What to build:** Compose projection, sampling, fusion, feature concatenation, pooling, and scatter with inspectable intermediate outputs.

**Blocked by:** 01 (Multi-camera projection with validity), 02 (Bilinear camera feature sampling), 03 (Stable masked multi-view fusion), 05 (Differentiable voxel and pillar pooling), 06 (Collision-safe count-weighted BEV scatter)

**Status:** resolved

- [x] Create an independent integration pair with local setup/answer helpers or supplied stage outputs, never imports of learner scripts/references.
- [x] Retain named projected, sampled, fused, concatenated, pooled, and BEV outputs. Preserve LiDAR features/occupancy without image support and carry a separate support indicator. Decoration is optional and does not gate this ticket.
- [x] Use asymmetric hand-built expected cells/features/counts. Check permutations, unavailable-camera NaNs, duplicate/unequal z counts, empty points, batches, and gradients to eligible image/LiDAR features.
- [x] Provide transparent and improved complete compositions. Demonstrate a faulty helper, retained mismatch, corrected helper, and fresh selective replay.

## Practice contract

Use top-level Emacs/Jupyter cells, one editable answer section, and cumulative
baseline/follow-up/production checks. Target a 30–45 minute baseline. Keep
imports at top level, ordinary shared verifier helpers in the verifier section,
and reference execution outside learner checks. Each pair owns deterministic
CPU fixtures and runs without importing another question or reference.
Infrastructure stays outside dojo. Allocate noncolliding four-digit exercise IDs.

Provide complete runnable teaching and progressively improved reference versions
in separate cells. State each version's qualified stage, comparison domain,
precision, and resource scope. Bind eval_data in the question and demonstrate
direct calls, retained failures, answer revision, compare, fresh replay, and check.
Independent analytic/scalar oracles supplement property and gradient checks.
Use float32/float64, common dtype/device, explicit validation, no input mutation,
and safe masked-NaN handling. No required dataset/checkpoint downloads. Optional
experiments report workload, device, dtype, timing variability, numerical error,
and measurement method; correctness has no speed threshold. Preserve existing APIs.

## Implementation evidence

Delivered exercise0109 with all thirteen documented scene keys and optional
chunk_size/min_depth controls. The output dictionary retains projected uv,
depth/support, sampled features/support, fused features/image support,
concatenated LiDAR/image/support channels, canonical voxel indices/counts/
means/inverse, and BEV/count/occupancy maps. Missing images never gate LiDAR
occupancy. The final support channel is the fraction of contributing points
with image support. Decoration remains optional.

Question setup supplies one local stable helper per prerequisite stage; answer
composes them. No numbered question/reference is imported. Helpers can be
modified in the kernel or saved candidate. Reference progressions are complete
transparent, stable-vectorized and chunked compositions, with explicit stage
limits and forward/output/backward storage distinctions. Local copied helper
snapshots include the integrated0105 and0107 maximum-finite fixes; only
algorithms actually used by those compositions remain.

Evidence at approved public seams:

- Baseline red0/1 (NotImplementedError), green1/1 after composing kernels.
  Literal hand BEV cells are [5,5.5,0.5] and [10,5.5,1]; unavailable image
  support does not remove either LiDAR point from the first pillar.
- Transparent reference passes3/3 followup checks and6/6 stage replay cases;
  both production references pass7/7 checks and13/13 replay cases.
- Independent scalar geometry/interpolation and direct original-point pillar
  means verify all output fields; optimized kernels are not the oracle.
- Sixteen bounded deterministic Hypothesis scenes verify point/camera
  permutations, counts and final BEV. Concrete scenes include duplicate
  unequal-z counts, feature affine mapping, half-open grid boundaries,
  batches, unavailable cameras, masked/all-masked NaNs, empty points,
  float32 strides/chunks, extreme confidence and live gradients.
- Float64 gradcheck and literal image-map/LiDAR derivatives pass. Empty,
  all-masked and no-camera backward paths remain finite with zero inactive
  gradients; input immutability and documented ValueError cases pass.
- Public saved-candidate production evaluation passes7/7 with raising poison
  in excluded teaching/reference cells. Modifiable setup helpers are loaded.
- Kernel-namespace helper revision through public replay: ignoring camera
  confidence retains a mismatch; restoring the helper gives a fresh passing
  result while the old diagnostics remain inspectable. Question cells expose
  this same edit/inspect/fix loop for actual Emacs/Jupyter qualification.
- Opt-in CPU experiment repeats the hand scene four times (N12), chunk4,
  float32: scalar BEV error zero for all versions; approximate medians1.37ms
  transparent,1.38ms stable,3.10ms chunked. Warmup3/repeats10 and variability
  are returned with workload/method; no speed gate or universal speed claim.

Ordinary smooth feature derivatives are covered; discrete masks/cell changes
and numerical clamp/scale rounding boundaries use the documented algorithm
contract. Extreme confidence derivatives are not an arbitrary-precision
guarantee.

## Integration review correction

The capstone copied pooling helper and editable question setup both received the included-cell upper-index correction. Named float32/64 literal cases retain LiDAR without images, producing a single BEV cell[3,0,0] with count2 and excluded outside inverse IDs. Initial checks gave6/8 with out-of-range scatter ValueError; all production/default checks now pass8/8. Saved-candidate qualification fills only the composition answer and exercises the actual question setup helpers.

Affected public-seam qualification passed all12 advertised reference scopes
and12 filled-question saved-candidate/stage combinations across0106,0107,
0109 and0112. Actual Emacs/Jupyter ran14 stateful cells retaining old failures
and passing corrected fresh named-case replay; artifacts live under
target/autonomy-perception/verification/review-fix. Formal Standards/Spec
recheck is pending and the qualification ticket's final checkbox stays open.
