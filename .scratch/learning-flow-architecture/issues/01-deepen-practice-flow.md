# Deepen the three practice-flow modules

Status: resolved

Implement the rep lifecycle, Emacs-owned scenario state, and shared script interpretation described in ../spec.md. Preserve stored rep archives and current notebook/bridge contracts. Record validation and remaining environment limits here.

## Resolution

Implemented all three modules and learner feedback improvements. Preserved rep
archive compatibility, selected-answer checks, cumulative stages and the Emacs
execution path. Added behavior coverage for switching and recovery races,
timer pauses, shared script rules, source-order helpers, stale results and
navigation. No ADR changes were needed.

## Verification

- 124 native reference eval scripts passed.
- Shared Python gate: 94 tests, passed with 3 environment skips.
- Final scenario Python gate: 48 passed; Emacs ERT: 30 passed.
- TypeScript typecheck passed; 86 tests passed.
- Browser Python gate including offline build: 97 passed; the additional
  empty-stage regression also passed in the final 11-test contract run.
- 39 browser end-to-end rep/Bench/case/timing tests passed, followed by 4
  feedback/offline/recovery tests after the feedback changes.
- Live GPU rootfs scenario passed on isolated port 8898: CUDA output, shared
  workspace evidence and confirmed owned-kernel removal. The existing server
  on 8899 was left running.
- Browser feedback was rendered and captured for inspection.

Logs and the screenshot are in ignored `target/architecture-validation/`.
