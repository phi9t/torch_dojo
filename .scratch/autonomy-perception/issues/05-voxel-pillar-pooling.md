# 05: Differentiable voxel and pillar pooling

**What to build:** Pool features into sparse occupied cells and inspect counts, inverse IDs, excluded points, and gradients.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Accept points [B,N,3], features [B,N,C], point mask, and fixed grid metadata. Return lexicographically sorted [V,4] batch/z/y/x indices, int64 counts, means, and global inverse IDs [B,N] with -1 outside.
- [x] Use half-open bounds, preserve batches and occupied zero features, and count duplicates. Pillar mode collapses grouping z to zero while retaining declared 3D inclusion bounds.
- [x] Check hand means/counts/inverse IDs, permutations, duplicates, voxel/pillar modes, boundaries, empty points, masked NaNs, strides, feature gradients, widened accumulation, and no mutation.
- [x] Provide grouping-loop, unique/scatter, and sorted-segment references with independent scalar groups/count conservation. Storage scales with points/occupied cells; guard packed-key overflow if used and preserve the NumPy API.

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

Added exercise 0107 with a top-level editable answer, stepwise grouping probes,
inspectable eval_data/direct compare/retained mismatch/fresh replay cells, and
live feature-backward exploration. Complete grouping-loop, unique/scatter, and
stable lexicographic sorted-segment references use sparse occupied rows and
avoid packed keys. Existing NumPy 0066 remains unchanged.

At the approved callable/evaluate seam, the independent hand group/count/mean/
inverse verifier failed with NotImplementedError before implementations existed;
it passed after runnable references were added. Six cumulative production
verifiers now pass for EACH reference; eleven fresh replay cases also pass for
each. Evidence covers hand means/inverse IDs, voxel/pillar modes, batch
separation, occupied zero features, shifted anisotropic lexicographic cells,
all excluded and empty inputs, masked NaNs, noncontiguous feature views, wide
float32 values and finite float64 means with overflowing raw sums. Analytic
1/count feature gradients and float64 gradcheck agree, inputs are preserved,
and twelve deterministic Hypothesis permutations preserve pooled values and
conserve included counts. Invalid dtype/shape/mask/grid metadata is rejected.

Float32 reductions widen to float64 and restore output dtype. All versions
divide contributions before summing to avoid overflowing unnormalized float64
sums. Sparse storage O(points*channels + occupied*channels), Python loop cost,
GPU ordering differences and extra retained backward state are stated.

Opt-in benchmark executed CPU float32 N=32,C=2,seed107,25 occupied cells, three
warmups and ten synchronized wall-clock repetitions; all references matched
with zero max mean error. No performance threshold gates correctness.
Qualification used project eval-venv Python, evaluate/replay, and project-local
TMPDIR target/t. Final diff whitespace check passed.


### Production numerical correction

A qualification follow-up exposed a decisive counterexample: seventeen
identical float64 finfo.max features overflowed after division-before-sum,
because rounding accumulated slightly past the convex mean bound. Added a
retrievable finite_max_convex_mean case: the old production verifier failed
5/6 before the fix. All three references now normalize by per-group/channel
max magnitude, reduce bounded values, clamp the normalized mean to [-1,1],
and restore a safe nonzero scale. The zero-scale path preserves ordinary
zero-feature derivatives. The independent scalar oracle uses math.fsum.

All three versions again pass all six production verifiers and twelve replay
cases, including finite maximum output and finite analytic 1/17 gradients.
Existing zero-feature gradcheck also passes. At a floating rounding-bound
correction, the derivative follows clamp/max; this is documented explicitly
rather than claiming a real-arithmetic gradient through the correction.
This supersedes the earlier division-before-sum numerical strategy.

## Integration review correction

Voxel/pillar pooling now corrects a rounded upper index only after physical half-open membership and finite/int64 conversion guards. Float32/64 named literal cases establish indices[[0,0,0,0]], count2, mean3 and inverse[[0,0,-1,-1]]. The exact zero/positive edge rows remain excluded. Initial regression gave5/7, then all three production versions passed7/7. Scalar cell assignment uses Decimal represented-input arithmetic so its oracle does not repeat floating cancellation.

Affected public-seam qualification passed all12 advertised reference scopes
and12 filled-question saved-candidate/stage combinations across0106,0107,
0109 and0112. Actual Emacs/Jupyter ran14 stateful cells retaining old failures
and passing corrected fresh named-case replay; artifacts live under
target/autonomy-perception/verification/review-fix. Formal Standards/Spec
recheck is pending and the qualification ticket's final checkbox stays open.
