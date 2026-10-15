# 04: LiDAR point-feature decoration

**What to build:** Construct inspectable point descriptors using occupied voxel centroids, centers, and range.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Accept batched points/attributes, point mask, origin, positive voxel sizes, and positive integer xyz grid shape. Use half-open bounds and preserve point order.
- [x] Define channels as supplied attributes, ego xyz, point-minus-voxel-centroid, point-minus-voxel-center, and range. Invalid rows are zero/masked. Batch-separated centroids include all valid duplicates.
- [x] Check one-voxel hand examples, multiple voxels/batches, anisotropic sizes, permutations, empty points, masked padding, metadata rejection, strides, and immutability.
- [x] Provide per-voxel, grouped-reduction, and stable/reused-grouping complete references. Check coordinate gradients with fixed membership away from boundaries/range zero; allocate no dense grid.

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

## Resolution and evidence

Delivered independently runnable 0106 question/eval scripts with one editable
answer cell, explicit channel order, direct inspection, retained result, fresh
replay, and a centroid-offset backward playground. Complete loop and grouped
references qualify followup; stable reused-grouping reference qualifies
production. Sparse grouping uses coordinate tuples instead of packed int64 keys
or dense voxel storage. Stable centroid contributions divide before reduction;
float32 geometry widens to float64 and range rescales even float64 coordinates.

Ten named fixtures use an independent Python grouping/math.fsum/math.hypot
oracle. Cases cover translated anisotropic cells, half-open boundaries,
duplicates/batches, zero-length point axes, masked padding/outside NaN attributes,
large float32/float64 ranges, enormous sparse grids, and expected metadata errors.
Bounded deterministic Hypothesis trials check independent parity and permutation
behavior. Public verifiers include float64 gradcheck, analytic [.5,-.5] centroid
coordinate derivatives, attribute gradients, all-excluded zero gradients,
noncontiguous tensors, immutability, and metadata validation.

TDD baseline: initial callable stub failed baseline (0/1), then complete loop
passed (1/1). Final reference gates: loop and grouped followup 3/3 each; their
production gates intentionally fail overflow (6/7), stable production 7/7.
Fresh replay all ten stable cases passes while the old range failure remains
inspectable. Both scripts compile and whitespace checks pass. Actual interactive
kernel and saved-candidate isolation qualification belongs to ticket08.

Valid points must be finite, included attributes must be finite, excluded
attributes may be nonfinite; geometry metadata and derived float64 upper bounds
are finite. Discrete voxel membership has no derivative across its boundaries.

## Integration review correction

Point decoration now groups already-included cells after guarding finite/int64-representable coordinates and correcting rounded upper indices. All three complete versions share this rule. Float32/64 literal cases establish a single voxel with centroid x=-0.25 and offsets +0.25/-0.25; exact zero and positive tiny points stay outside. Initial new verifier failed (7/8), then stable default passed8/8; loop/grouped followup scopes also pass.

Affected public-seam qualification passed all12 advertised reference scopes
and12 filled-question saved-candidate/stage combinations across0106,0107,
0109 and0112. Actual Emacs/Jupyter ran14 stateful cells retaining old failures
and passing corrected fresh named-case replay; artifacts live under
target/autonomy-perception/verification/review-fix. Formal Standards/Spec
recheck is pending and the qualification ticket's final checkbox stays open.
