# 02: Bilinear camera feature sampling

**What to build:** Sample dense camera features and inspect coordinate mappings, corner values, interpolation weights, and refined support.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Accept features [B,M,C,Hf,Wf], uv [B,M,N,2], support, and explicit affine feature_from_image; return samples [B,M,N,C] and support. Never infer spatial alignment from tensor sizes.
- [x] Use integer feature centers and inclusive feature-grid bounds; outside outputs are zero/false. Handle the final row/column and singleton dimensions without duplicate-weight errors.
- [x] Baseline numbered/analytic ramps, follow-up rectangular multichannel/multiview grids, and production empty points, masked NaNs, strides, gradients through features/interior coordinates, and validation pass.
- [x] Provide explicit four-corner and vectorized references plus a chunked or grid_sample version with derived matching boundary semantics. Independent scalar expectations expose swapped axes and crop/resize errors.

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

Delivered exercise 0104 as top-level question/eval scripts with one editable
answer, eleven fresh named cases, independent scalar expectations, and three
complete reference versions: four-corner teaching loop, vectorized gathers,
and bounded-point chunked gathers. All qualify production float32/float64;
resource scopes are distinguished from retained backward state.

TDD at the approved exercise/evaluate seam:

- Red: numbered-grid baseline failed with NotImplementedError (0/1).
- Green: explicit interpolation returned literal 5.25, 12, and masked zero (1/1).
- Red: deeper public checks exposed empty-stack, missing zero-output autograd,
  and dtype error failures (3/7).
- Green: validation/sanitization and graph-connected empty/unsupported outputs
  pass all eight production verifiers, including bounded deterministic
  Hypothesis affine-ramp trials and float64 central-difference gradcheck.
- Every complete version independently passed 8/8 verifiers and 11/11 replay
  cases. Float32 CPU parity and common-device/dtype checks pass.
- Opt-in CPU experiment at B=1, M=2, N=32, C=4, H=8, W=12, float32, seed104:
  max scalar-oracle error 1.2e-7; synchronized wall-clock medians approximately
  5.77 ms teaching, 0.292 ms vectorized, 0.830 ms chunks of eight. Warmup3,
  repeats10 and variability are returned; no speed gate.

Unavailable camera maps/uv may be NaN; finite affine metadata remains required.
Any camera with incoming support requires its entire feature map finite.
This explicit policy is stated in both learner and implementation contracts.
