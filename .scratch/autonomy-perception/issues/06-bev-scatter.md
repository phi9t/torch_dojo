# 06: Collision-safe count-weighted BEV scatter

**What to build:** Accumulate sparse cells into BEV without losing duplicate contributions or confusing zero features with emptiness.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Accept batch/z/y/x indices, features, positive int64 counts, and positive batch/xyz grid dimensions. Return features [B,C,H,W], counts [B,1,H,W], and boolean occupancy with y mapped to H and x to W.
- [x] Collapse z using count-weighted means, including duplicate indices; match direct point-to-pillar means. Reject invalid indices/dimensions and count overflow before unsafe allocation/accumulation.
- [x] Check asymmetric grids, unequal-count z collisions, batches, empty inputs, strides, widened arithmetic, validation, gradients, and immutability.
- [x] Provide explicit, flattened index_add, and grouped-reduction references plus an overwrite counterexample. Independent point means validate results; declare unavoidable dense output storage separately from temporaries.

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

Delivered exercise0108 as independently runnable question/eval scripts with
thirteen inspectable cases, including expected count-overflow rejection. The
teaching numerator loop qualifies followup; normalized, per-cell-scaled
index_add and grouped references qualify production. Exact count preflight
happens before dense allocation; only per-cell totals are limited to int64.

TDD at approved callable/evaluate seams:

- Baseline red0/1 (NotImplementedError), then green1/1 with literal weighted
  mean3.5 and explicit occupied-zero cell.
- Numerical red2/3: float32 weighted products overflowed; normalized float64
  work restored green3/3 including huge float64 products.
- Validation/empty backward red3/5: invalid batch raised IndexError and empty
  output had no graph; exact preflight and differentiable sparse sources
  restored green5/5.
- A further decisive17-row maximum-finite fixture exposed normalized-sum
  rounding overflow (6/7). Per-cell magnitude conditioning and detached
  convex-range rounding correction restored7/7. Ordinary float64 gradcheck
  and analytic count/total gradients at the maximum-finite case pass.
- Teaching version passes2/2 followup verifiers and5/5 fresh stage cases; both
  production versions pass7/7 verifiers and13/13 replay cases. Checks include
  twenty deterministic bounded Hypothesis direct-point-mean/permutation
  trials, strides, float32/64, immutability, duplicate z/xyz collisions, batch
  separation, empty backward, exact huge counts, and metadata overflow.
- Opt-in CPU timing K24,C4,grid12x8x3,float32,seed108: scalar-oracle max error
  zero; medians about0.615ms explicit,0.125ms index_add,0.402ms grouped.
  Warmup3/repeats10 and variability are returned; no speed gate. Grouped
  avoids a dense float64 accumulator, while dense output remains mandatory.

Core stores only finite occupied features; upstream point masks are already
resolved by pooling. Numerical conditioning is detached to preserve the
linear weighted derivative, and finite differences beyond float64 maximum
are explicitly excluded as an oracle.
