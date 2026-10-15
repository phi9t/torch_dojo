# 13: Qualify the connected vision foundation model practice loop

**What to build:** Qualify the path from patch features and adaptation into BEV attention, with reproducible interactive debugging.

**Blocked by:** 08 (Qualify and document the core autonomy practice loop), 09 (Patch embedding and dense visual feature layout), 10 (Rectangular positional grid interpolation), 11 (Frozen-backbone dense feature adaptation), 12 (Geometry-guided BEV cross-attention)

**Status:** resolved

- [x] Qualify vision reference scopes/stages/default evals, saved candidates, verifier isolation, and required repository gates without regressing the core batch.
- [x] Run actual Emacs/Jupyter layout/interpolation/adaptation/attention inspection and replay; retain a layout failure and a gradient/state failure before correction.
- [x] Document composition of rectangular layouts, prefix tokens, feature coordinates, masks, and attention. Keep real-checkpoint exploration optional; model-specific extensions require primary implementation evidence.
- [x] Record commands/results, limits, and Standards/Spec review evidence; preserve unrelated work and use project-local runtime/scratch paths.

## Qualification evidence

Qualification implementation is complete. The final acceptance item remains
unchecked until the integration-wide Standards/Spec review evidence is added.
That review is the next skill phase, not a claim made by this ticket.

On integration86bbec3, the public check/evaluate seams qualified15 advertised
reference/stage scopes across0027,0110,0111,0112 and0113. Every patch/layout/
position/attention version passed production; adapter Simple and Tokens passed
production, and the deliberately inference-only version passed baseline only.
All15 filled actual-question saved-candidate/stage combinations passed, with
raising poison in excluded teaching/reference cells and an executed tagged
setup marker. The class answer runner builds new instances; adapter case replay
uses fresh supplied backbones and explicit fixture weights.

Actual Emacs29.3/emacs-jupyter/ipykernel ran23 sequential stateful cells on
the owned core verification kernel. The retained artifacts include:

- Scrambled rectangular/prefix layout failure: output max absolute error6.
  Editing the function and replaying fresh fixtures passed; the old result
  stayed inspectable.
- Adapter forward matched literal expected maps, but detached images had
  grad=None despite finite adapter gradients. That observation was retained
  before replacing the class and constructing fresh instances.
- Training the buggy wrapper enabled backbone dropout and incremented its
  training_calls buffer. The retained result reports state/input changes and
  output max error111. Corrected train(True) kept backbone eval and counter0.
- Corrected image gradients were7.5 per pixel; frozen backbone gradients
  stayed absent while adapter gradients were present.
- Rectangular positional interpolation preserved prefix entries, and layout
  reconstruction produced [2,3,3,5].
- A live frozen token backbone plus adapter produced camera maps for a
  geometry-guided BEV attention query using explicit image-to-feature affine.
  Attention backward reached image inputs and adapter weights; backbone
  parameters stayed frozen. One additional cell captured retained failures
  and corrected gradient/state observations as a JSON artifact.

## Commands and artifacts

Run from the repository root with project-local temporary state:

- TMPDIR=$PWD/target/t target/eval-venv/bin/python target/autonomy-perception/verification/vision/run.py
  ->15 reference scopes and15 saved-candidate stages pass; report.json and
  steps.json under the same vision artifact directory.
- python target/autonomy-perception/verification/enter.py vision emacs --batch -Q -l target/autonomy-perception/verification/vision/connect.el
  ->exit0; emacs.log confirms stateful vision loops passed.
- The corresponding capture.el invocation ->exit0; kernel-report.json records
  retained layout/gradient/state failures and corrected live bridge gradients.
- TMPDIR=$PWD/target/t make eval PYTHON=target/eval-venv/bin/python
  ->exit0,113 default reference scripts, zero failed checks; make-eval.log.
- TMPDIR=$PWD/target/t make test PYTHON=target/eval-venv/bin/python
  ->exit0,49 tests run, seven optional Textual dependency skips; make-test.log.
- git diff --check ->clean. No remote push, no /tmp, no dataset/checkpoint
  downloads. Unrelated untracked research work was preserved.

The practice guide now documents the connected vision sequence, rectangular
layout/prefix conventions, interpolation and explicit feature coordinates,
freezing versus detaching, attention support/key order and optional experiments.
The owned kernel remains ready for the final review-fix agent; root will stop
it during final cleanup.

## Final review and limits

Independent Standards review found zero issues. Spec review found two P2
issues: rounded outer-boundary cell IDs and singular adapter affine metadata.
Both were fixed in c5d9309, with literal regression fixtures, affected native
candidate checks, and fourteen actual Emacs/Jupyter cells. Both reviewers
independently rechecked the fixes: zero remaining findings on either axis.
The integration review record contains the reports and evidence.

Root's final merged-code gates passed: 113 reference scripts (exit 0), and
49 repository tests with seven optional skips (exit 0). Logs remain under
project-local target/autonomy-perception. All owned worktrees were removed;
the owned qualification kernel shut down successfully. Unrelated work was
preserved. CUDA throughput and real-checkpoint equivalence remain outside the
small deterministic CPU qualification scope.
