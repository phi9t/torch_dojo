# 09: Patch embedding and dense visual feature layout

**What to build:** Trace images into patch tokens and reconstruct rectangular dense maps while diagnosing prefix and channel-order mistakes.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Upgrade existing patch embedding with inspectable fixtures and progressive references, preserving its API/stages. Add an independent dense-layout pair.
- [x] Layout accepts tokens [B,P+Hf*Wf,C], explicit positive Hf/Wf and nonnegative prefix count P; return row-major [B,C,Hf,Wf]. Never infer sqrt(token_count).
- [x] Check numbered multichannel patches, rectangular grids, prefixes, batches/empty batch, invalid metadata, strides, gradients, dtype/device, and immutability.
- [x] Provide transparent/vectorized layout versions and meaningful patch extraction/projection alternatives with shared-weight parity. Explain copies/storage and explicit feature-coordinate metadata; require no checkpoint.

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

Upgraded 0027 while preserving PatchEmbedding constructor, .proj nn.Linear,
.num_patches nominal count, rectangular runtime behavior, state_dict shape, and
original staged verifiers. Complete loop, unfold, and convolution-forward
versions share the same projection weights. Convolution avoids an explicit
Python patch matrix without claiming zero backend workspace. Five named fixtures
carry fixed constructor/weight/bias metadata and independent scalar projection
expectations; the question shows weight copying, direct compare, and fresh replay
with a newly instantiated learner class after revision.

Added independent 0110 dense-layout pair with complete loop, view, and contiguous
versions. Explicit prefix count and rectangular height/width control token
removal and ordering. Five named fixtures use scalar-index expectations,
including expected invalid-layout errors. The question supports retained results,
fresh replay, and a prefix-gradient playground; output view aliasing is explicit.
Coordinate metadata cannot be inferred from token counts; unpadded patch-center
mapping is explained for the specific 0027 extractor.

Public seam red/green: initial dense callable stub failed baseline 0/1, then
complete loop passed 1/1. Deliberately wrong direct BC-HW reshape fails the
numbered baseline 0/1. Final three dense references each pass 5/5, three patch
references each pass 6/6, including all original patch gates. Shared-weight
forward/backward matches independent scalar examples and analytic gradients.
Bounded deterministic Hypothesis trials exercise rectangular grids and prefixes;
gradcheck, strides, float32/float64, emptiness, immutability and validation pass.
Fresh replay passes all five cases per pair/reference. Optional measurements
report workload, error and timing variability; dense storage aliases/strides and
analytical patch matrix bytes are distinct from allocator peaks/backward state.

Saved candidates and actual Emacs/Jupyter qualification belong to tickets08/13.
