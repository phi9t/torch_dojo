# 01: Multi-camera projection with validity

**What to build:** Project batched ego points into cameras and inspect coordinates, camera-z depth, and geometric support.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Document points [B,N,3], point/camera masks, intrinsics [B,M,3,3], camera_from_ego [B,M,4,4], image sizes, and uv/depth/support outputs [B,M,N,...]. Use ego x-forward/y-left/z-up and camera x-right/y-down/z-forward, meters, row points, and explicit transform direction.
- [x] Support requires valid point/camera, depth greater than finite positive min_depth, and inclusive integer pixel-center bounds [0,W-1] and [0,H-1]. Invalid outputs are zero/masked; behind-camera points do not reject the whole call. Distinguish projectable from unoccluded.
- [x] Baseline hand examples, follow-up rotations/batches/missing cameras, and production empty points, strides, masked NaNs, calibration validation, smooth gradients, and no mutation pass.
- [x] Complete per-camera, broadcasted, and point-chunked references explain bounded temporaries versus required M-by-N output; preserve the existing strict projection API.

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

Implemented exercise 0103 with one editable top-level answer section, explicit
frame/support/validation contracts, inspectable cases and direct compare/fresh
replay cells. The original 0044 strict projection API is untouched.

TDD evidence at the approved exercise callable/evaluate seam: the independent
hand-projection verifier failed with NotImplementedError before any reference
implementation existed. It then passed after the first runnable reference;
subsequent independent scalar, shape/calibration, generated translation, and
smooth-gradient checks qualified the complete progression. A float32 threshold
oracle mismatch was discovered and corrected by representing min_depth in the
input dtype, explicitly documented in the question.

Qualification: per-camera, broadcast, and point-chunked references each passed
all six cumulative production verifiers and seven fresh EvalCase replays.
Checks include 90-degree rotation, translated multi-batch/missing cameras,
all-cameras-missing, empty and strided points, masked NaN padding, malformed
calibration/metadata, float64 central differences through points/K/T, no input
mutation, and 15 deterministic Hypothesis translation trials. Optional timing
experiment was exercised on CPU float32 B=1,M=2,N=16,chunk=3 with three warmups
and ten synchronized wall-clock repeats; all versions had zero uv error.
Chunked forward temporaries and required outputs/retained autograd state are
explicitly distinguished. No timing threshold gates correctness.

Command: project eval-venv Python ran the eval script and qualified all three
versions through evaluate plus replay. Runtime TMPDIR was project-local target/t.
