# 08: Deepen MLP, convolution, RoPE, LoRA, patch, and MoE lessons

**What to build:** Upgrade dojo exercises 0015,0022,0024,0026,0027,0028 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Module state, parameter counts, tensor layout, odd shapes, gradient flow, routing ties, and reference parity.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Comments

Resolved with six staged question/eval pairs. The contracts exercise module registration, parameter counts, exact arithmetic, gradient flow, non-contiguous layouts, validation, stable RoPE angles, and deterministic MoE ties. The sparse MoE reference iterates over experts and accumulates selected token outputs without constructing every expert-token output. All six reference evals passed, as did 34 judge tests (6 optional skips), syntax checks, and `git diff --check`.
