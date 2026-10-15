# 13: Deepen unprojection, triangulation, pose, polar, and temporal geometry

**What to build:** Upgrade dojo exercises 0051,0052,0053,0054,0065 into one editable answer with cumulative baseline, follow-up, and production checks. Focus on Consistent world/camera/ego transforms, invalid depth, near-degeneracy, inverse checks, and dtype preservation.

**Blocked by:** 12: vision-foundations

**Status:** resolved

- [x] Each question script gives a worked probe, explicit contracts for all three stages, and a live `check(..., stage=...)` cell for each stage.
- [x] Each eval script has a runnable reference and independent `verify_baseline_*`, `verify_followup_*`, and `verify_production_*` functions; later stages keep earlier behavior.
- [x] Reference evals for this ticket, judge tests, syntax checks, and `git diff --check` pass; any new production claim has an observable test or a reviewable complexity argument.

## Answer

Exercises 0051–0054 and 0065 now teach camera unprojection, triangulation,
rigid-pose inversion, polar conversion, and temporal ego frame conversion in
cumulative stages. Their reference checks cover frame order, front-of-camera
and near-parallel geometry, invalid inputs, empty and strided data, gradients,
dtype preservation, and masked invalid trajectories. All five reference evals
pass (23 checks); 34 repository tests pass with 6 optional skips. Syntax
checks and `git diff --check` pass.
