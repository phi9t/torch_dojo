# 04: Qualify and document the complete pilot practice loop

Status: resolved
Blocked by: 02 — Explore paged attention from dense to streaming; 03 — Explore MoE routes, expert batching, and gradients
Parent: Progressive dojo references

## What to build

Demonstrate the complete three-pilot workflow, document how to use and author it, review the implementation against the spec and repo standards, and land the qualified result locally on main. Establish evidence that the new experience works in the live-kernel and saved-file workflows before expanding it to more exercises.

## Acceptance criteria

- [x] Document retrieving eval data, inspecting a case, executing the current answer directly, retaining comparisons, selecting failures, and replaying fresh inputs. Show how to explore reference versions and request measurements explicitly.
- [x] Document the distinction between reference versions and cumulative learner stages, including supported-contract qualification and the limits of a vanilla implementation.
- [x] Document the author contract for cases, independent expectations, structured results, expected exceptions, shared fixtures, changed inputs, and stateful or property verifiers. Avoid claiming that a finite data collection represents all generated property trials.
- [x] Demonstrate the edit/inspect/run/fix/replay loop in a real Jupyter kernel usable from Emacs. Verify that top-level data/results remain available between cells and that fixing the answer uses the current kernel definition.
- [x] Exercise saved-file candidate evaluation for each pilot through every stage it advertises. Prove that teaching probes, exploration cells, reference versions, and optional measurements stay excluded from saved-answer loading.
- [x] Check every reference version against its declared supported stage and independent common-domain parity requirements. Verify an intentionally wrong implementation is rejected with reproducible diagnostic data for each pilot.
- [x] Run the full reference eval gate, the repository test gate, pilot syntax checks, and diff checks. Add or repeat checks only to address a concrete remaining risk or required gate.
- [x] Complete separate Standards and Spec reviews, fix the findings, and recheck affected behavior. Confirm the interactive-script ADRs and the small judge interface remain intact.
- [x] Land verified changes on local main without a remote push. Use project-local artifact locations rather than /tmp, preserve unrelated work, and clean up worktrees created for this feature.
- [x] Record verification evidence and remaining limitations in ticket comments. Resolve implementation tickets only when their required work is complete; leave parent-spec status management to the implementation workflow.

## Comments

Ticket 01 is a transitive dependency through tickets 02 and 03. This slice qualifies the combined learner experience and documents the reusable pattern; migrating the rest of the exercise bank remains outside this pilot.

### Implementer qualification evidence

- Added the learner/author guide `docs/practice/interactive-eval.md` and README
  entry. It documents ordinary top-level kernel variables, inspection, fresh
  replay, exception cases, stale expectations, detached result snapshots,
  supported reference stages, and opt-in measurements.
- Qualified seven complete versions through the public `check` interface:
  shifted softmax and dense paged attention at followup; wide softmax, streaming
  paged attention, and traceable/batched/grouped dispatch at production. These
  checks retain independent forward/backward expectations.
- Exercised nine saved-candidate combinations (three stages for each pilot)
  through the public `evaluate` interface. Project-local answer copies contain
  complete algorithms; excluded cells contain raising poison. The copied eval
  files likewise poison reference and exploration cells while retaining imports
  and the verifier section. All nine pass without executing the poison.
- An owned Jupyter kernel, reached through Emacs 29.3 and emacs-jupyter, passed
  12 sequential cells. Each pilot exposes data, retains a concrete failure,
  redefines the answer in a later cell, and passes fresh replay. The MoE final
  cell also performs backward on a fresh live output and inspects leaf gradients.
  The owned kernel was shut down after the check.
- Repository tests: 48 ran successfully with seven optional environment skips, using
  `TMPDIR=$PWD/target/t`. A first run
  with the longer worktree-local temporary path hit existing Unix socket path
  limits; the shorter project-local directory resolved those environment errors.
- Root integration full reference gate: 102 scripts passed (exit 0), recorded
  in `target/progressive-references/verification/final-eval.log`. Six pilot
  scripts parsed successfully and `git diff --check` passed.

Artifacts (ignored, local): `qualification/run.py`, `qualification/report.json`,
`qualification/steps.json`, `qualification/connect.el`, and
`qualification/test.log` beneath `target/progressive-references/verification/`.
All scratch state stayed under the project. No remote operations were performed.

Remaining workflow gates: separate Standards and Spec reviews, any resulting
fixes, local main landing, and worktree cleanup. Ticket 04 remains pending those
gates. This qualification covers CPU portable algorithms and Emacs/Jupyter
request execution; it does not claim manual GUI usability testing or GPU
throughput. Kernel shutdown logged a jupyter-client destructor warning during
interpreter finalization after successful shutdown; it did not affect results.

### Resolution

- Separate Standards and Spec reviews completed against baseline `cdbaec7`.
  The duplicated MoE verification oracle and large-integral stale-comparison
  issue were fixed in `70204f9`. Affected checks passed; both reviewers
  confirmed zero remaining findings. See `../review.md`.
- Final affected repository gate: 49 tests successful, with seven optional
  environment skips. All three MoE reference versions still pass 12 production
  verifiers and ten replay cases each. The earlier complete reference gate
  remains 102/102, and saved-candidate plus Emacs/Jupyter qualification remains
  as recorded above; the fixes preserve those interfaces.
- Fast-forwarded local main to the verified implementation `70204f9`. No
  remote push. Removed all five feature implementer worktrees, preserving
  unrelated worktrees, concurrent tracked Emacs/insula edits, and untracked
  session research/experiments. Verification artifacts remain under target.
- The three-pilot scope is complete. Broader bank migration, arbitrary module
  side-effect tracking, fused GPU kernels, and formal verification remain
  outside this spec.
