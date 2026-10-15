# 11: Frozen-backbone dense feature adaptation

**What to build:** Adapt dense features from a tiny frozen backbone while diagnosing detached gradients and unintended state changes.

**Blocked by:** 09 (Patch embedding and dense visual feature layout)

**Status:** resolved

- [x] Use a supplied tiny backbone with explicit token layout and dense reconstruction; return adapted maps with declared channels and feature-coordinate metadata.
- [x] Freeze backbone weights, preserve requested input and adapter gradients, and state backbone eval policy independently from adapter train/eval. Verify a controlled stochastic/stateful fixture.
- [x] Check frozen parameters, finite requested gradients, train/eval transitions, fresh module/leaf fixtures, layout rejection, dtype/device, and buffer-mutation policy.
- [x] Provide transparent and improved complete references. Any no_grad/cache variant is explicitly inference-only when it cannot preserve input gradients. Include a detached-feature failure and optional real-checkpoint adapter boundary without downloads.

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

Delivered independently runnable 0112 class exercise with constructor/forward
contract, explicit grid/prefix/channel layout, trainable .adapter Conv2d and
copied fixed feature-coordinate buffer. Constructor freezes/evaluates the
supplied backbone; train(mode) keeps backbone eval and lets adapter follow the
wrapper. Forward preserves requested image/adapter gradients without backbone
parameter gradients. A supplied backbone must honor eval-state buffer purity;
arbitrary custom side effects are not silently purified by this wrapper.

Complete references cover dense reconstruction + 1x1 convolution, equivalent
token-side projection without an explicit contiguous input map, and a declared
inference-only no_grad version that rejects requires_grad images. The latter
qualifies forward-only baseline, not followup/production gradient behavior.
Six named fixtures contain fresh frozen/eval backbones and leaf image tensors,
explicit projection weights/bias, and independent literal image-pixel oracles.
Question cells construct the learner class, inspect retained outputs, backward,
revise, and replay with fresh modules; no numbered reference imports/downloads.

Public-seam TDD: initial class stub failed baseline 0/1; complete dense/tokens
versions passed. Final dense and token versions each pass production 6/6 and
all six fresh replay cases. Inference baseline passes 1/1; requested-gradient
followup intentionally fails (3/4). A deliberate no_grad feature experiment
reproduces equal forward maps with missing input gradient; correct gradient is
7.5 for every toy pixel. Independent analytic adapter weight/bias gradients and
float64 input gradcheck pass. Deterministic Hypothesis values, train/eval
transitions with dropout/training-counter fixture, frozen parameter/buffer state,
strides, empty batch, dtype transitions, immutability, metadata/layout errors pass.

Opt-in measurements compare shared weights under identical inference no_grad
scope, report workload/device/dtype/error/timing variability, and distinguish
analytical explicit input-map bytes from allocator/backend/backward storage.
Question documents optional real-checkpoint wrapping boundary without downloads.
Saved-candidate and Emacs/Jupyter qualification belongs to tickets08/13.

## Integration review correction

Every adapter constructor now rejects a singular2x2 affine linear block before building/freeze/eval state changes, matching sampler metadata requirements. The named singular_feature_affine case has a valid bottom row and dependent spatial rows; its constructor verifier checks float32/64. Initial default6/7 accepted the singular transform, then production versions passed7/7. The inference-only reference still qualifies baseline forward only, while its constructor also rejects the named metadata fixture.

Affected public-seam qualification passed all12 advertised reference scopes
and12 filled-question saved-candidate/stage combinations across0106,0107,
0109 and0112. Actual Emacs/Jupyter ran14 stateful cells retaining old failures
and passing corrected fresh named-case replay; artifacts live under
target/autonomy-perception/verification/review-fix. Formal Standards/Spec
recheck is pending and the qualification ticket's final checkbox stays open.
