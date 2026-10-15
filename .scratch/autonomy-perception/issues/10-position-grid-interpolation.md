# 10: Rectangular positional grid interpolation

**What to build:** Resize spatial positional embeddings while preserving prefix positions and explicit coordinate conventions.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Accept position tokens, explicit old/new rectangular grids and prefix count; validate token count and preserve prefixes exactly.
- [x] State interpolation mode, align_corners meaning, antialias policy, accumulation/output dtype, and parity domain. Unchanged grids have an exact identity path.
- [x] Check rectangular up/down sampling, constants/analytic cases, singleton axes, prefixes, batches, malformed metadata, strides, gradients, and no mutation.
- [x] Provide transparent and optimized complete versions, adding stability versions only for demonstrated improvements. Independent analytic/scalar expectations complement framework parity; expose inspectable eval data without checkpoints.

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

Added exercise 0111 with explicit rectangular/prefix/half-pixel contracts and
three complete reference versions: four-neighbor teaching loops, separable
vectorized gathers, and layout-aware framework interpolation. All qualify
production; no extra numerical-stability version is invented without evidence.
Question cells expose eval_data, direct implementation calls, retained compare
results, fresh selective replay, cumulative check and live gradient inspection.

At the approved callable/evaluate seam the hand ramp [0,2] -> [0,.5,1.5,2]
failed with NotImplementedError before references existed, then passed. Each
reference now passes six cumulative production verifiers and seven fresh named
case replays. Checks include rectangular up/down sampling, exact prefixes and
same-grid identity, singleton axes, strided tensors, empty batch, invalid
metadata/count/dtype/nonfinite data, no mutation, float64 gradcheck and weighted
backward parity against matching framework settings. Fifteen deterministic
Hypothesis rectangular/singleton/prefix constant trials agree with independent
scalar half-pixel expectations.

Opt-in benchmark ran CPU float32 B=2,old=(2,3),new=(3,5),C=2,prefix=1,seed111,
three warmups and ten synchronized wall-clock repeats. Maximum error against
teaching reference was 0 for teaching, 5.96e-8 separable, 2.38e-7 framework.
Output/intermediate layout and backend workspace scope are documented; timing
is not a correctness gate. Qualification used project eval-venv Python,
evaluate/replay, project-local TMPDIR target/t, and diff whitespace checks.
