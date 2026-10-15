# Staged Dojo Quality Implementation Plan

> **For agentic workers:** Use the specification and keep each migration batch independently verifiable. Steps use checkboxes for tracking.

**Goal:** Turn each dojo pair into a baseline, follow-up, and production learning path while preserving interactive scripts.

**Architecture:** Extend the existing judge only to select cumulative named verifier stages. Each question script retains one answer section; each eval script owns its reference and native verifiers. Roll out by complete exercise pairs.

**Tech Stack:** Python, NumPy, PyTorch, Hypothesis, unittest, Emacs/Jupyter `# %%` scripts.

**Spec:** `docs/specs/staged-dojo-quality.md`

## Global constraints

- Keep question/eval pairs under `dojo/` and judge infrastructure under `torch_judge/`.
- Preserve direct script execution, interactive top-level cells, and the single `# %% [answer]` cell.
- Keep old exercise pairs working during the migration.
- Never represent authored follow-ups as source requirements.

## Review focus

- A production stage must include baseline and follow-up checks, not select only its own verifiers.
- Saved candidate evaluation must skip teaching probes and use the stage requested by the learner.
- All-masked attention rows need finite forward and backward values.
- Sparse voxel grids must not allocate memory proportional to the dense grid volume.
- CSV records split across chunk boundaries must parse identically to one-piece input.

### Task 1: Stage-aware judge

- [x] Add failing public-interface tests for cumulative stage selection, legacy behavior, candidate default, and invalid stage.
- [x] Implement selection in `torch_judge/engine.py` without changing answer/verifier cell loading.
- [x] Run judge tests and all current reference scripts.

### Task 2: Attention pilot

- [x] Rewrite question as a baseline/follow-up/production script with explicit mask and model-state rules.
- [x] Upgrade reference to an `nn.Module` with safe masking, validation, and finite gradients.
- [x] Add independent stage verifiers and run all stages against the reference.

### Task 3: Voxel pilot

- [x] Rewrite question with boundary, invalid-data, reproducibility, and sparse-memory stages.
- [x] Tighten reference validation and preserve sorted occupied coordinates and repeated-index accumulation.
- [x] Add generated and large-sparse-grid verifiers.

### Task 4: CSV pilot

- [x] Rewrite question with a clear grammar, malformed-input policy, and chunked-input follow-up.
- [x] Upgrade reference parser for iterable chunks without concatenating the input.
- [x] Add split-at-every-character and malformed-input verifiers.

### Task 5: Migration batches

- [x] Audit all remaining pairs by family and list the contract gaps before editing them.
- [x] Migrate foundation/autograd, geometry/vision, algorithms/data, and evaluation exercises in independently passing batches.
- [x] Keep special-purpose design/CUDA exercises scoped to meaningful production constraints.
- [x] Run `make eval`, judge tests, syntax checks, and diff checks after each batch.

### Task 6: Documentation and completion

- [x] Document the stage workflow, `--stage` usage, and per-exercise quality bar in the README.
- [x] Confirm all 81 pairs have baseline, follow-up, and production checks or a documented reason a stage does not apply.
- [x] Verify the entire repository and land locally.
