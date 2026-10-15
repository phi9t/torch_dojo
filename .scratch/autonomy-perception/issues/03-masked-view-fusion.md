# 03: Stable masked multi-view fusion

**What to build:** Combine supported camera views into point features and diagnose wrong denominators or unstable confidence weights.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Accept samples [B,M,N,C], support [B,M,N], and nonnegative eligible confidence [B,M,N]; return weighted means [B,N,C] and support. Zero eligible mass returns zero/false.
- [x] Masked values/weights cannot contaminate arithmetic. Eligible weights need not sum to one; reject invalid eligible weights and define accumulation/output dtype.
- [x] Check equal weights, one view, confidence weighting, camera permutations, all missing, zero mass, extreme weights, empty points, strides, smooth gradients, and immutability across cumulative stages.
- [x] Provide transparent reduction, rescaled stable normalization, and point-chunked references with independent scalar expectations and an inspectable overflow counterexample. Learned attention is deferred.

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

Delivered 0105 masked-view fusion as top-level question/eval scripts. Three
complete reference cells expose raw masked reduction (followup), scaled/widened
normalization (production), and independently implemented point chunks
(production). The public output preserves float32/float64 dtype/device and
separates missing support from a genuine zero feature. Positive-weight valid
samples must be finite; zero-weight/excluded samples may contain NaNs.

Nine named EvalCase fixtures use an independent Python scalar/math.fsum oracle,
including an expected ValueError fixture. Bounded deterministic Hypothesis
trials check scalar parity and camera permutations. Verifiers cover finite
zero gradients on all-missing/zero-mass NaN padding, analytic gradients and
float64 gradcheck, invalid contracts, noncontiguous batches, and immutability.
Question cells retain a failed result across fresh replay and offer live
backward inspection. Overflow and workload measurements are opt-in functions.

Public-seam red/green evidence: transparent production evaluation fails the
weight-overflow case (5/6); transparent followup passes 3/3; stable and chunked
production each pass 6/6. Fresh replay retains the old overflow mismatch while
all nine new chunked cases pass. CPU backward passes; CUDA was unavailable.
Both scripts compile and git diff --check passes. Integration qualification
owns saved-candidate and actual Emacs/Jupyter session checks.

Storage scope is explicit: point chunks bound reduction temporaries, while
validation uses input-sized selection and backward can retain whole-call state.
No universal peak-memory or speed claim is made.


### Finite maximum feature boundary correction

A decisive production regression with twenty-two equal unit-weight cameras and
positive/negative float64 finfo.max channels failed before the change (5/6
checks): rounding normalized confidence past unit mass produced infinity.
Added a named finite_max_convex_mean fixture with literal finite expectations.
Stable and point-chunked references now detach per-point/channel feature
conditioning, reduce normalized values, and enforce the convex magnitude bound
with a straight-through floating-rounding correction. This preserves ordinary
weighted-mean gradients; extreme confidence derivatives can still overflow
and that boundary scope is stated in question/reference comments.

Transparent reference remains follow-up only (3/3). Stable and chunked each
pass all seven cumulative production verifiers, including ordinary feature and
confidence gradcheck and finite-max feature derivative 1/22, and all ten fresh
replay cases including expected exceptions. Project eval-venv Python and local
TMPDIR target/t were used; no external services or remote push.
