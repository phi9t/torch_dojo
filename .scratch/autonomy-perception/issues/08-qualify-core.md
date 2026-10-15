# 08: Qualify and document the core autonomy practice loop

**What to build:** Qualify a usable core learning path in Emacs/Jupyter with trustworthy reference stages and candidate isolation.

**Blocked by:** 01 (Multi-camera projection with validity), 02 (Bilinear camera feature sampling), 03 (Stable masked multi-view fusion), 04 (LiDAR point-feature decoration), 05 (Differentiable voxel and pillar pooling), 06 (Collision-safe count-weighted BEV scatter), 07 (Inspect camera and LiDAR fusion through BEV)

**Status:** resolved

- [x] Qualify every core reference at its advertised cumulative stage and strongest default target; run required judge/repository gates and preserve prior exercise behavior.
- [x] Verify saved setup/answer selection and reference-free verifier loading using raising poison in excluded teaching/reference/experiment cells.
- [x] Run actual Emacs/Jupyter eval_data inspection, deliberately wrong answer, retained diagnostics, answer correction, fresh replay, and live backward inspection.
- [x] Document sequence, geometry/support/discrete-gradient contracts, reference scope, experiments, and optional decoration. Record commands/results and Standards/Spec review evidence; keep scratch/runtime project-local and preserve unrelated work.


## Resolution and qualification evidence

Added the autonomy perception practice guide and README entry. It documents
learning order, optional point decoration, geometry/support and discrete
assignment contracts, live gradient inspection, each reference's qualified
stage and numerical/resource scope, named diagnostics and opt-in experiments.

Qualification covered all 21 advertised core reference/stage combinations and
21 saved-candidate/stage combinations (three cumulative stages per pair).
Raising poison in excluded question teaching/experiment and eval reference/
experiment cells never executed. Setup execution was required by an explicit
answer assertion. The real capstone question setup was additionally preserved
with a filled alias-based answer and poisoned teaching cells; all three stages
passed. The seven core factories expose 76 inspectable named cases, including
expected exceptions. Each deliberately wrong answer retained diagnostics,
then fresh replay with the corrected callable passed.

Actual Emacs29.3/emacs-jupyter/ipykernel ran 28 separate edit/inspect/correct/
fresh-replay cells plus one live camera-map/LiDAR backward cell. Globals and
older failed results remained available across cells; both live feature gradient
families were finite. This is a small CPU workflow qualification, not CUDA
throughput or checkpoint/dataset qualification. The owned verification kernel
remains available for the following vision qualification; it must be stopped
only after that work, without touching unrelated sessions.

Commands completed with exit status 0:

- Project eval-venv Python with PYTHONPATH=. and project-local TMPDIR ran the
  prepared core qualification and native-capstone candidate qualification.
- Task-local rootfs entry invoked Emacs --batch -Q with the core connection
  script against the owned kernel.
- make eval PYTHON=target/eval-venv/bin/python: all 112 reference eval scripts
  present at this qualification completed successfully.
- make test PYTHON=target/eval-venv/bin/python: 49 tests run, seven optional
  skips, all remaining tests passed. Expected negative-example verifier output
  in the test log is part of passing regression tests.
- git diff --check: clean.

Reports, source selections, poisoned candidates, kernel steps and logs remain
in ignored target/autonomy-perception/verification/core. No /tmp paths, remote
push, required downloads, or changes to unrelated research work were used.

## Review handoff

Local contract review checked the guide against source interfaces/stages and
actual qualification reports. The final independent Standards and Spec reviews
are the integration-wide gate after the vision batch, as specified by
implement-spec; their findings and fixes belong in the feature review record.
This ticket does not claim those future reviews have already passed. The final
vision qualification repeats the complete repository gate after its additions.
