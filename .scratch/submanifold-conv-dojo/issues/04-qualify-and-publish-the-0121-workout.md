# 04: Qualify and publish the 0121 workout

**What to build:** Qualify the completed `0121_submanifold_conv` workout as part of the dojo catalog. The reference should pass every advertised stage, the learner starter should fail only because the answer is incomplete, and the public index should list the new workout with the correct total count.

Blocked by: 03: Production validation and sparse limits

Status: resolved

- [x] The full reference evaluation passes baseline, follow-up, and production verifiers for the new workout.
- [x] Saved candidate evaluation loads the question script and fails for an answer-related reason, with teaching probes and reference code excluded from learner checks.
- [x] The repository eval and test gates pass after adding the workout.
- [x] The README links the new question and eval pair and updates the workout count.
- [x] Final verification records that unrelated untracked files were not edited and project-local scratch or target locations were used instead of `/tmp`.

## Comments

- 2026-10-11: Published `0121_submanifold_conv` in the README catalog and updated the visible staged-workout count from 102 to 103.
- 2026-10-11: Reference verification passed with `PYTHONPATH=. uv run python dojo/0121_submanifold_conv_eval.py --stage baseline` (2/2), `PYTHONPATH=. uv run python dojo/0121_submanifold_conv_eval.py --stage followup` (5/5), and `PYTHONPATH=. uv run python dojo/0121_submanifold_conv_eval.py --stage production` (8/8). Saved starter candidate loaded with `PYTHONPATH=. uv run python dojo/0121_submanifold_conv_eval.py --candidate --stage production` and failed for answer-stub reasons: `NoneType` outputs and missing `ValueError` validation from `pass` bodies.
- 2026-10-11: Repository gates passed with `PYTHONPATH=. uv run make eval` and `PYTHONPATH=. uv run make test`. Final status check showed only this ticket's intended files changed; verification used the worktree-local `.venv` and project paths, not `/tmp`.
