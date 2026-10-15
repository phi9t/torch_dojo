# Browser ML Gym Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a standalone JupyterLite browser gym for 12 existing NumPy exercises and one full NumPy MLP lab, with interactive notebooks, isolated checks, durable attempts, and portable exports.

**Architecture:** `browser_gym/` builds notebooks and verifier bundles from canonical `dojo/` files, then serves a pinned JupyterLite/Pyodide site. A frontend extension owns the exercise panel, solution-cell capture, receipts, and attempts; a disposable Pyodide worker runs each authoritative check. The exploration notebook and checker never share an interpreter.

**Tech Stack:** Python 3, NumPy, Hypothesis, Matplotlib, JupyterLite core 0.8.5, JupyterLite Pyodide kernel 0.8.6, Pyodide 314.0.6, JupyterLab 4.6-compatible TypeScript extension, IndexedDB, Playwright.

**Spec:** `docs/superpowers/specs/2026-10-05-browser-ml-gym-design.md`. Decision research: `docs/research/marimo-vs-jupyterlite-browser-gym.md`.

## Global Constraints

- Keep the browser feature under `browser_gym/`; read canonical contracts and verifiers from `dojo/` without changing the native runner or existing exercise files.
- Qualify exactly `0059`, `0060`, `0061`, `0062`, `0063`, `0064`, `0065`, `0066`, `0069`, `0070`, `0073`, and `0074` for the release; keep PyTorch exercises native-only.
- Treat notebook solution cells as the only submitted code. Snapshot and hash their concatenated source before every check; scratch cells and exploration state are excluded.
- Use a fresh worker for every check, with a controller-owned deadline and hard termination. Keep notebook exploration persistent until Stop/restart.
- Record exact source, versions, fixture identity, result details, and assistance in append-only receipts. A reveal stays recorded after its UI closes.
- Keep the MLP lab's canonical Python, reference, fixtures, and native check inside `browser_gym/labs/mlp/`.
- Use local browser storage for convenience and an explicit archive for backup. Do not claim offline support until a network-disabled reload passes.
- Pin and qualify the installed browser packages against the actual verifiers; package availability does not grant permission to call a solution helper.

## File map

- `browser_gym/requirements-build.txt`, `jupyter-lite.json`, `Makefile`, `extension/package.json`: pinned build, site configuration, and commands.
- `browser_gym/catalog.json`, `build_catalog.py`, `tests/test_build_catalog.py`: ID mapping, derived notebooks/verifier bundles, build validation.
- `browser_gym/extension/src/source.ts`, `source.spec.ts`: selected-cell snapshot and SHA-256.
- `browser_gym/extension/src/checker.ts`, `checker.worker.ts`, `checker.spec.ts`: worker protocol, timeout, and termination.
- `browser_gym/check_runtime.py`, `tests/test_check_runtime.py`: execute submitted source and canonical verifier functions in a fresh Python interpreter; return structured evidence.
- `browser_gym/extension/src/attempts.ts`, `attempts.spec.ts`: attempt metadata, receipts, assistance, and archive import/export; JupyterLite Contents owns the editable notebook files.
- `browser_gym/extension/src/panel.tsx`, `index.ts`, `panel.spec.tsx`: notebook-facing gym commands and results.
- `browser_gym/lesson_ui.py`, `lessons/`, `tests/test_lesson_ui.py`: tensor cards, hints, probes, and exercise-specific visual templates.
- `browser_gym/labs/mlp/`: canonical capstone source, deterministic data, reference/checks, mutant, notebook template, and native command.
- `browser_gym/tests/e2e/`, `README.md`: browser acceptance and learner/build documentation.

## Review Focus

- A notebook with no marked solution cell, or a solution cell outside the active exercise, must refuse Check without producing a misleading receipt (Task 2).
- A source edit during an outstanding Check must not change that Check's receipt; the UI must label the result stale (Task 4).
- A package/import failure in the worker must become a visible failed receipt, with the captured source retained (Task 3).
- A malformed import archive or a receipt whose source hash differs from its snapshot must be rejected without overwriting stored attempts (Task 5).
- Stopping a runaway check or exploration cell must leave the interface usable for a new execution (Task 3 and Task 12).

---

### Task 1: Reproducible site and derived `0059` lesson

**Files:** Create `browser_gym/requirements-build.txt`, `browser_gym/extension/package.json`, `browser_gym/extension/playwright.config.ts`, `browser_gym/jupyter-lite.json`, `browser_gym/Makefile`, `browser_gym/catalog.json`, `browser_gym/build_catalog.py`, `browser_gym/tests/test_build_catalog.py`, `browser_gym/tests/e2e/site_smoke.spec.ts`, and `browser_gym/README.md`.

**Interfaces:** `build_catalog.build(root: Path, out: Path, ids: list[str]) -> list[dict]` reads question/eval pairs with `# %% [answer]` and `# %% [verifiers]`, writes a versioned browser bundle and `.ipynb` with metadata `dojo.exercise_id` and `dojo.role="solution"`, and returns catalog entries. Initial enabled ID is `0059`; the full 12-ID catalog is explicit but disabled until qualified.

- [ ] **Step 1: Write failing tests** named `test_build_0059_from_canonical_pair`, `test_reject_missing_or_duplicate_markers`, and `test_build_does_not_execute_reference`. Assert the output answer text equals the tagged source, exactly one cell has `dojo.role="solution"`, scratch cells have no solution role, malformed pairs raise `ValueError`, and a top-level reference side effect never runs. Write `site_smoke.spec.ts` to assert the `0059` notebook opens and a browser cell returns `numpy.__version__`.
- [ ] **Step 2: Verify red:** `python -m pytest browser_gym/tests/test_build_catalog.py -q` fails on the absent builder.
- [ ] **Step 3: Implement** the catalog builder, pinned site configuration, `make build`, and `make serve`; build output goes to ignored `browser_gym/dist/`. Use the canonical `dojo/0059_*` pair, not a copied hand-edited exercise.
- [ ] **Step 4: Verify green:** `python -m pytest browser_gym/tests/test_build_catalog.py -q`, `make -C browser_gym build`, and `npm --prefix browser_gym/extension run test:e2e -- site_smoke.spec.ts` pass.
- [ ] **Step 5: Commit** only Task 1 files as `feat: build first browser dojo lesson`.

### Task 2: Exact solution-cell capture

**Files:** Create `browser_gym/extension/src/source.ts` and `source.spec.ts`; modify `browser_gym/extension/package.json`.

**Interfaces:** `captureSolution(model: INotebookModel, exerciseId: string): Promise<{source: string; sha256: string}>` reads code cells in document order with `dojo.role="solution"`, joins them with one blank line, and hashes UTF-8 bytes. It rejects missing cells, mismatched `dojo.exercise_id`, or non-code solution cells.

- [ ] **Step 1: Write failing tests** named `captures_answer_in_document_order` and `rejects_missing_or_foreign_answer`. Assert exact joined source and SHA-256, scratch/plot exclusion, and rejection of a cell with the wrong exercise ID.
- [ ] **Step 2: Verify red:** `npm --prefix browser_gym/extension test -- source.spec.ts` fails on the absent capture function.
- [ ] **Step 3: Implement** `captureSolution` against public JupyterLab notebook model APIs; do not read rendered DOM or kernel globals.
- [ ] **Step 4: Verify green:** `npm --prefix browser_gym/extension test -- source.spec.ts` and `npm --prefix browser_gym/extension run test:e2e -- site_smoke.spec.ts` pass; the browser test compares captured source to visible answer text.
- [ ] **Step 5: Commit** Task 2 files as `feat: capture browser solution snapshots`.

### Task 3: Isolated browser checker and Stop

**Files:** Create `browser_gym/check_runtime.py`, `browser_gym/tests/test_check_runtime.py`, `browser_gym/extension/src/checker.ts`, `checker.worker.ts`, `checker.spec.ts`, and `browser_gym/tests/e2e/checker.spec.ts`.

**Interfaces:** `run_check(source: str, exercise_id: str, bundle_dir: Path) -> dict` executes source in a new namespace, loads only the generated verifier imports/functions, and returns `status`, `checks[]`, `error`, `runtime`, and `fixture_id`. `runCheck({source, sha256, exerciseId, checkerVersion, timeoutMs}): Promise<CheckReceipt>` creates one worker, validates its response against the request ID, and always terminates it after completion/Stop/deadline.

- [ ] **Step 1: Write failing Python tests** named `test_reference_starter_and_batch_gradient_mutant` and `test_isolated_source_and_import_error`: assert `0059` reference passes, starter fails, the batch-gradient mutant fails its intended verifier, scratch-only names are unavailable, and bad imports produce structured errors. Write `checker.spec.ts` cases asserting success, timeout, Stop, stale response ID rejection, and a successful second check after termination.
- [ ] **Step 2: Verify red:** `python -m pytest browser_gym/tests/test_check_runtime.py -q` and `npm --prefix browser_gym/extension test -- checker.spec.ts` fail for missing implementations.
- [ ] **Step 3: Implement** the minimal Python runner and disposable Pyodide worker protocol; package/import startup failures resolve to failed results with the source hash. The controller owns the deadline and calls `terminate()`; it never waits for Python cleanup.
- [ ] **Step 4: Verify green:** both Step 2 commands and `npm --prefix browser_gym/extension run test:e2e -- checker.spec.ts` pass; the browser test covers reference, starter, mutant, prior-kernel-state isolation, and stopping `while True: pass` followed by a successful new check.
- [ ] **Step 5: Commit** Task 3 files as `feat: check snapshots in fresh browser workers`.

### Task 4: Focused exercise and result panel

**Files:** Create `browser_gym/extension/src/index.ts`, `panel.tsx`, `panel.spec.tsx`, and `browser_gym/tests/e2e/panel.spec.ts`; modify `browser_gym/extension/src/source.ts` only for model-change signals.

**Interfaces:** `openExercise(id: string): Promise<void>` opens its generated notebook and binds the panel. Panel actions are Run cell, Check solution, Stop, Hint, New attempt, and Export; Hint, New attempt, and Export become active in Task 5. Check passes one captured immutable snapshot to `runCheck`. Results display named verifier outcomes, evidence, traceback location, and the checked hash, with a separate hint explanation.

- [ ] **Step 1: Write failing UI tests** named `shows_one_active_exercise` and `labels_inflight_result_stale_after_edit`. Assert Check keeps its original hash when source changes during execution, the resulting status says stale, a failure's evidence is shown separately from a hint, and Run cell never submits a check.
- [ ] **Step 2: Verify red:** `npm --prefix browser_gym/extension test -- panel.spec.tsx` fails.
- [ ] **Step 3: Implement** the extension plugin and panel using JupyterLab services; keep native notebook cell editing and output rendering.
- [ ] **Step 4: Verify green:** `npm --prefix browser_gym/extension test -- panel.spec.tsx` and `npm --prefix browser_gym/extension run test:e2e -- panel.spec.ts` pass; Playwright opens `0059`, edits a solution cell, checks it, and displays the exact checked hash.
- [ ] **Step 5: Commit** Task 4 files as `feat: add focused gym panel`.

### Task 5: Attempts, hints, receipts, and archive

**Files:** Create `browser_gym/extension/src/attempts.ts`, `attempts.spec.ts`, and `browser_gym/tests/e2e/attempts.spec.ts`; modify `browser_gym/extension/src/panel.tsx` and `browser_gym/catalog.json` for hints/solution text.

**Interfaces:** `AttemptStore` exposes `newAttempt(exerciseId)`, `appendReceipt(attemptId, receipt)`, `recordAssistance(attemptId, kind, level)`, `exportAttempt(attemptId): Promise<Blob>`, `exportAll(): Promise<Blob>`, and `importArchive(blob): Promise<string[]>`. The store keeps attempt ID, JupyterLite Contents notebook path, notes, append-only receipts, and assistance; JupyterLite Contents owns the only editable notebook copy. Export reads notebooks and includes `.ipynb`, derived `solution.py`, notes, and JSON receipts with SHA-256 validation.

- [ ] **Step 1: Write failing tests** named `creates_distinct_notebooks_for_attempts`, `keeps_reveal_history_after_collapse`, and `rejects_corrupt_archive_atomically`. Assert two attempt paths remain distinct across reload, later independent re-solve is displayed separately from an assisted pass, edited-since-check state follows the source hash, both single-attempt and whole-history archives round-trip, and malformed/hash-mismatched archives leave existing attempts intact.
- [ ] **Step 2: Verify red:** `npm --prefix browser_gym/extension test -- attempts.spec.ts` fails.
- [ ] **Step 3: Implement** the store, notes editor, and panel integration. A revealed reference stays out of answer cells; New attempt never rewrites the prior notebook/receipt.
- [ ] **Step 4: Verify green:** `npm --prefix browser_gym/extension test -- attempts.spec.ts` and `npm --prefix browser_gym/extension run test:e2e -- attempts.spec.ts` pass, including reload/export/import.
- [ ] **Step 5: Commit** Task 5 files as `feat: retain and export gym attempts`.

### Task 6: Notebook-native visual lessons for `0059` and `0060`

**Files:** Create `browser_gym/lesson_ui.py`, `browser_gym/tests/test_lesson_ui.py`, `browser_gym/lessons/0059.py`, `browser_gym/lessons/0060.py`, and `browser_gym/tests/e2e/visual_lessons.spec.ts`; modify `browser_gym/build_catalog.py` to insert generated probe/plot cells.

**Interfaces:** `tensor_card(name: str, value: np.ndarray, axes: tuple[str, ...] = ())` renders shape, dtype, axis labels, and a bounded preview. Lesson templates provide a parameter control, a named experiment question, a plot, and a gated Run action for training; all lesson cells are scratch/experiment roles and excluded by `captureSolution`.

- [ ] **Step 1: Write failing tests** for tensor shapes `(2, 3)` and dtype `float64`, bounded preview for a large tensor, escaped labels, and notebook generation preserving the `0059`/`0060` answer cells. Add browser assertions that changing logit scale updates the `0059` plot and `0060` training requires an explicit Run.
- [ ] **Step 2: Verify red:** `python -m pytest browser_gym/tests/test_lesson_ui.py -q` and the named Playwright visual lesson test fail.
- [ ] **Step 3: Implement** the small Python display helper and widget-backed lesson cells; keep numerical values/plots adjacent to their questions and provide text alternatives.
- [ ] **Step 4: Verify green:** `python -m pytest browser_gym/tests/test_lesson_ui.py -q` and `npm --prefix browser_gym/extension run test:e2e -- visual_lessons.spec.ts` pass; manually inspect both views at desktop and narrow widths.
- [ ] **Step 5: Commit** Task 6 files as `feat: add interactive numerical lessons`.

### Task 7: Canonical NumPy MLP checkpoints

**Files:** Create `browser_gym/labs/mlp/starter.py`, `reference.py`, `fixtures.py`, `checks.py`, `mutants.py`, `test_checks.py`, `README.md`, `browser_gym/lessons/mlp.py`, and `browser_gym/tests/e2e/mlp.spec.ts`; add MLP to `browser_gym/catalog.json`.

**Interfaces:** Learner functions are `init_params(rng, input_dim, hidden_dim, classes)`, `forward(X, params)`, `loss_and_grads(X, y, params)`, `train_epoch(X, y, params, state, rng, batch_size, learning_rate)`, `predict_proba(X, params, batch_size)`, `evaluate_model(X, y, params)`, `save_state(params, state)`, and `restore_state(snapshot)`. `checks.run_checkpoint(name: str, candidate: dict, reference_overrides: set[str]) -> dict` grades one checkpoint and identifies every substituted reference component in its result.

- [ ] **Step 1: Write failing native tests** named `test_mlp_reference_checkpoints` and `test_mlp_mutant_and_substitution`. Assert fixed nonlinear 2D splits, forward shapes, directional finite differences by parameter block, broad training improvement, batch-size-invariant inference, evaluation without mutation, state round trip, and that the starter fails each owned checkpoint. The omitted-batch-mean mutant must report a gradient magnitude discrepancy. A substituted backward pass must be named in the receipt and must not mark backward independently passed.
- [ ] **Step 2: Verify red:** `python -m pytest browser_gym/labs/mlp/test_checks.py -q` fails.
- [ ] **Step 3: Implement** the deterministic fixture, reference, starter, checkpoint harness, one SGD optimizer, mutant, and native check command. Allow explicit reference substitution for later checkpoints without marking the substituted checkpoint independently passed. The debugging variant requires a short diagnosis naming the failed invariant, supporting evidence, and a regression check.
- [ ] **Step 4: Verify green:** `python -m pytest browser_gym/labs/mlp/test_checks.py -q` and `npm --prefix browser_gym/extension run test:e2e -- mlp.spec.ts` pass; browser tests run loss, boundary, confidence, and gradient-norm experiments and check each checkpoint in a fresh worker.
- [ ] **Step 5: Commit** Task 7 files as `feat: add full NumPy MLP modeling lab`.

### Task 8: Calibration and connected-path acceptance

**Files:** Create `browser_gym/lessons/0069.py` and `tests/e2e/connected_path.spec.ts`; modify `catalog.json` and the panel's next-exercise link.

**Interfaces:** `0059 → 0060 → MLP → 0069` links are explicit catalog edges. `0069` controls bin count and displays a reliability diagram plus confidence distribution while preserving the canonical metric contract.

- [ ] **Step 1: Write failing browser acceptance** for broken then corrected `0059` gradient, `0060` training, MLP training/evaluation, `0069` plot, fresh-worker isolation, Stop, new attempt, and export/import with identical checked source.
- [ ] **Step 2: Verify red:** `npm --prefix browser_gym/extension run test:e2e -- connected_path.spec.ts` fails.
- [ ] **Step 3: Implement** the calibration lesson and next links; fix only defects surfaced in the existing core workflow.
- [ ] **Step 4: Verify green:** connected-path acceptance, `make test`, and `make eval` pass.
- [ ] **Step 5: Commit** Task 8 files as `feat: connect modeling and calibration path`.

### Task 9: Qualify K-means and box geometry

**Files:** Create `browser_gym/lessons/{0061,0062,0063,0064}.py`; modify `browser_gym/catalog.json`; create `browser_gym/tests/e2e/qualification.spec.ts`.

**Interfaces:** `qualify(id: string): Promise<QualificationReport>` runs the canonical reference, starter, and a selected mutant in fresh browser workers and checks one visual. Each enabled catalog entry names its canonical pair, allowed helpers, checker version, fixture ID, hints, and reference reveal.

- [ ] **Step 1: Write failing qualification cases** for `0061`–`0064`: assert reference pass, starter fail, intended mutant fail, and one meaningful visual each. Include a Pyodide import check for Hypothesis in `0062`.
- [ ] **Step 2: Verify red:** `npm --prefix browser_gym/extension run test:e2e -- qualification.spec.ts` fails for these four IDs.
- [ ] **Step 3: Implement** the four lessons and diagnostics; enable an ID only when its browser qualification passes.
- [ ] **Step 4: Verify green:** the four cases and `make test` pass.
- [ ] **Step 5: Commit** Task 9 files as `feat: qualify clustering and box labs`.

### Task 10: Qualify sensor, voxel, and anchor labs

**Files:** Create `browser_gym/lessons/{0065,0066,0070}.py`; modify `browser_gym/catalog.json` and `browser_gym/tests/e2e/qualification.spec.ts`.

**Interfaces:** Reuse `qualify(id)` from Task 9 and the same catalog fields. Visuals show transformed points, voxel contributions, and matching collisions with shape cards beside the relevant arrays.

- [ ] **Step 1: Write failing qualification cases** for `0065`, `0066`, and `0070`: assert reference pass, starter fail, intended mutant fail, and the named visual plus tensor shape output.
- [ ] **Step 2: Verify red:** `npm --prefix browser_gym/extension run test:e2e -- qualification.spec.ts` fails for these three IDs.
- [ ] **Step 3: Implement** the three lessons and diagnostics, reading canonical `dojo/` pairs.
- [ ] **Step 4: Verify green:** these cases and prior qualification cases pass.
- [ ] **Step 5: Commit** Task 10 files as `feat: qualify sensor and voxel labs`.

### Task 11: Qualify grouped evaluation and complete catalog

**Files:** Create `browser_gym/lessons/{0073,0074}.py`; modify `browser_gym/catalog.json` and `browser_gym/tests/e2e/qualification.spec.ts`.

**Interfaces:** Reuse `qualify(id)` from Task 9. `0073` plots grouped bootstrap results; `0074` plots the precision-recall curve under its canonical AP convention.

- [ ] **Step 1: Write failing qualification cases** for `0073` and `0074`, plus a catalog assertion of exactly 12 enabled existing IDs and MLP. Each ID must pass its reference, fail its starter and selected mutant for the intended reason, and render a meaningful visual.
- [ ] **Step 2: Verify red:** `npm --prefix browser_gym/extension run test:e2e -- qualification.spec.ts` fails for these two IDs and the catalog count.
- [ ] **Step 3: Implement** the two lessons and diagnostics, then enable the complete catalog.
- [ ] **Step 4: Verify green:** all 12 existing exercises qualify; `python -m pytest browser_gym/tests browser_gym/labs/mlp -q` passes.
- [ ] **Step 5: Commit** Task 11 files as `feat: qualify complete NumPy browser catalog`.

### Task 12: Runtime recovery, offline proof, and release documentation

**Files:** Modify `browser_gym/jupyter-lite.json`, `browser_gym/extension/src/panel.tsx`, `browser_gym/README.md`, and build files; create `browser_gym/tests/e2e/recovery_offline.spec.ts`.

**Interfaces:** Stop requests exploration-kernel interrupt where supported and offers hard restart; hard restart discards only that kernel's state. `make offline-build` bundles pinned runtime, packages, widgets, and exercise assets. The documentation states browser persistence/export limits and the native route.

- [ ] **Step 1: Write failing browser tests** for a runaway exploration cell followed by hard restart and a working new cell; a stopped runaway check followed by a working new check; network-disabled reload running a `0059` cell, tensor card, widget, and check; and no lost attempt after reload.
- [ ] **Step 2: Verify red:** `npm --prefix browser_gym/extension run test:e2e -- recovery_offline.spec.ts` fails.
- [ ] **Step 3: Implement** interrupt/restart controls, offline asset build, and setup/usage docs; pin the final tested runtime/package identities in receipts.
- [ ] **Step 4: Verify green:** recovery/offline tests, complete browser suite, `make test`, and `make eval` pass. Record actual browser/version results and package hashes in `browser_gym/README.md`.
- [ ] **Step 5: Commit** Task 12 files as `docs: qualify browser gym release`.
