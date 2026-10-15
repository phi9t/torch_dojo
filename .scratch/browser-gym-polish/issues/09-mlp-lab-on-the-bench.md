# 09: MLP lab on the Bench layout

**What to build:** The MLP lab (see the **Workout** entry: a longer workout checked in independent checkpoints) uses the Bench layout. The rail lists its checkpoints instead of stages, the verifier grid groups squares by checkpoint, and the detail panel shows checkpoint evidence and any reference substitutions. Splits are recorded per checkpoint.

**Blocked by:** 04, 05

**Status:** done

- [x] Checkpoints can be checked in any order. A checkpoint's seal stamps on its first pass, independent of the others.
- [x] The detail panel names any reference substitutions used for a checkpoint, so a pass that relied on them is never mistaken for a full pass.
- [x] Lab semantics are unchanged: same checkpoints, verifiers and evidence as today.
- [x] The lab's brief is typed like 0059's: a Goal, one section per checkpoint, and a Watch out on reference overrides, with `$…$` math. Its source lives in the lab's directory, not as markdown strings in the build.
- [x] The lab has two or three drills with reasoning, kept outside the learner's starter, so the 13 browser workouts are all migrated (spec Out of Scope note).
- [x] The native MLP checks and the existing lab e2e behaviours pass in the new layout. e2e adds checkpoint splits and per-checkpoint stamps.

## Comments

- Most of the Bench layout already applied to the lab through tickets 03–05 (its seals, independent-order splits and stamps). This ticket filled the gaps: the rail's list is named "Checkpoints", the brief doesn't fade "later" checkpoints, and the grid shows every checkpoint group at once.
- Grid: for independent checkpoints each group shows the latest result of that checkpoint in the rep ("from an earlier check" in the detail panel), and a checkpoint never checked shows one not-run `checkpoint_<name>` square. Only the check just run picks the first failing square.
- The detail panel names each reference substitution and says the pass is not independent credit. It shows a checkpoint's latest check only; the full receipt history stays in the Check log.
- Brief and drills: `labs/mlp/brief.py` uses the question-script markers with a heading per checkpoint (`# Forward — …`); `brief_sections` and `render_brief` now take the workout's stage labels. Drill reasoning lives in `labs/mlp/reasoning.py`, outside the starter and outside the checker bundle, so the lab's `checker_version` is unchanged.
- Native: `python -m browser_gym.labs.mlp.checks --candidate reference --checkpoint all` passes; checker sources are byte-identical to before.
