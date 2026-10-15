# 12: Geometry-guided BEV cross-attention

**What to build:** Connect adapted visual features to projected BEV queries and inspect geometry, support, attention weights, and updates.

**Blocked by:** 01 (Multi-camera projection with validity), 02 (Bilinear camera feature sampling), 11 (Frozen-backbone dense feature adaptation)

**Status:** resolved

- [x] Compose local/supplied adaptation, projection, and sampling with explicit queries [B,Q,D] and ego reference points. Baseline has one height; follow-up introduces a declared camera/height key axis.
- [x] State mask semantics and output shape. All-masked queries return zero update/false support with no softmax NaNs or projection-bias leakage; residual composition is separate.
- [x] Check analytic one-key attention, batches, permutations, missing-camera NaNs, unsupported points, geometry order, heights, shapes/strides, and eligible smooth gradients.
- [x] Provide transparent and vectorized/chunked complete references with precision/storage scope and independent analytic expectations. Expose coordinates, samples, masks, weights, and updates; infer neither depth nor occlusion.

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

Delivered independently runnable 0113 with exact bev_attention signature and
local production-safe projection/sampling helpers. No numbered imports. Keys
are camera-major/anchor-major; update is scaled-dot attention over supported
features without projections, bias, dropout, or residual. All-masked queries
return zero/false. Inspectable named diagnostics expose image uv/depth/projected
mask, samples/refined sample mask, and float64 scores/weights. Updates preserve
input dtype. Eligible dot scores outside finite float64 range raise ValueError.

Three complete attention reference cells cover per-query loop, broadcast, and
query chunks. All qualify production. Value accumulation scales convex sums
and keeps ordinary feature derivatives at exactly zero valid features. Query
chunks bound transient corner/score work; required diagnostic output, spatial
map sanitization/validation and retained backward state are separate costs.
Opt-in shared-workload measurements report error, dtype/device, timing variability
and analytical score/output bytes, without peak-memory or universal-speed claims.

Ten named EvalCases use independent Python scalar geometry, four-corner sampling
and attention arithmetic. Fixtures include single key, multiple cameras/heights,
missing NaN cameras, all-masked/outside support, zero queries, widened float32
scores, invalid queries and expected float64-score overflow. Adapted-map fixture
performs one deterministic supervised update on a locally frozen token backbone
and exposes learned adapter weights/loss/step lineage; no checkpoint download.

Public-seam TDD: initial stub failed baseline 0/1, then completed implementation
passed baseline. Final each loop/vector/chunk version passes production 7/7 and
all ten fresh replay cases. A deliberate wrong query-return implementation fails
baseline 0/2; its retained mismatch survives a successful fresh replay. Independent
differentiable scalar geometry/attention oracle verifies gradients through query,
maps, reference points, intrinsics, extrinsics and feature-coordinate mapping.
Float64 gradcheck, all-masked finite zero gradients, valid-zero feature gradient,
Hypothesis camera/anchor permutations and batch separation, strides, chunks,
immutability, and metadata rejection pass. Syntax/whitespace checks pass.
Saved-candidate and actual Emacs/Jupyter qualification belongs to ticket13.
