# Browser ML Gym

## Purpose and release boundary

Add a Python-first browser practice environment to Torch Dojo without changing the existing editor and native evaluation workflow. A learner opens one exercise, edits and runs a notebook, submits only designated solution cells to an isolated check, inspects evidence, preserves the attempt, and can export ordinary Python source. The first release covers 12 existing NumPy exercises and one end-to-end NumPy MLP lab. PyTorch exercises remain native-only until separately qualified.

The browser feature lives in `browser_gym/`. Existing `dojo/` question and eval files remain the source of truth for their contracts and verifiers. Browser build outputs are derived artifacts, never a second hand-edited curriculum. Work already in progress elsewhere in the checkout is outside this project's scope.

## Exercise mapping and progression

The browser manifest maps the brief's proposed path to this repository's IDs:

| Stage | Existing exercise | Browser focus |
| --- | --- | --- |
| Numerical ML | `0059_stable_cross_entropy`, `0060_classifier_step`, `0061_kmeans` | Numerical stability, gradient evidence, training, convergence |
| Geometry and point clouds | `0062_pairwise_iou`, `0063_oriented_box_sampling`, `0064_point_in_rotated_box`, `0065_temporal_transforms`, `0066_voxel_pool`, `0070_anchor_match` | Boundary cases, overlays, assignment and pooling diagnostics |
| Evaluation | `0069_calibration_metrics`, `0073_paired_trip_bootstrap`, `0074_detection_ap` | Reliability and grouped-result plots, explicit metric conventions |

The path built first is `0059 → 0060 → MLP capstone → 0069`. The remaining nine existing exercises use the same workspace and checker. The capstone has a canonical Python source, fixtures, reference implementation, and checks under `browser_gym/labs/mlp/`, with a documented native check command. It is listed in the browser manifest alongside the existing exercises. Its browser notebook is generated from those canonical files.

## Workspace and source ownership

Use pinned JupyterLite and `jupyterlite-pyodide-kernel` releases for notebook editing, persistent exploration, IPython, inline Matplotlib, and supported `ipywidgets`. This keeps one notebook editor and execution model. Marimo's browser notebooks are a viable separate route for reactive lessons, but mixing their `.py` cell model with JupyterLite notebooks would give attempts two editable formats and two state models. The first release uses notebook widgets for interactivity and records the marimo option for a future evaluated migration, rather than embedding a second editor.

A small JupyterLab frontend extension owns the focused exercise panel, commands, status, attempt history, and results panel. It opens one exercise notebook at a time. Each notebook has solution cells marked by notebook metadata, plus scratch and experiment cells that are excluded from checking. The extension concatenates solution cells in document order into `solution.py`; imports and helpers needed by the answer must be in those cells. Notebook content is the editable browser document. The Python file is a derived snapshot and export.

The exercise panel shows the contract, examples, allowed helpers, and progressive hints. Hints expand one step at a time; a reference solution is behind an explicit Reveal solution control. Hints and solutions can be collapsed again, but the attempt keeps any reveal as an assistance event. A solution reveal never edits the learner's solution cells. Results distinguish measured failures from suggested explanations and show the check's exact source hash. The main commands are Run cell, Check solution, Stop, Hint, New attempt, and Export. A changed solution cell marks the previous pass as stale.

The notebook is the interactive teaching surface, not just a text editor next to a checker. Exercise templates include executable probes, compact tensor cards showing shape, dtype, named axes, and a bounded value preview, plus Matplotlib figures beside the question they answer. Widget controls let the learner vary a meaningful parameter such as logit scale, learning rate, batch size, IoU threshold, or calibration bins and rerun only the relevant probe. Training remains behind an explicit Run action so a slider cannot launch repeated long jobs. Tensor cards inspect only values explicitly passed by the learner or template; they do not read private checker state. Each exercise has at least one shape/behavior probe and one relevant plot or visual diagnostic. The MLP lab includes interactive loss curves, decision boundary, confidence, and gradient-norm views. Visual styling should make stages and results easy to recognize while keeping code, numeric evidence, and accessibility clear.

## Browser checker

The persistent exploration kernel is never the authoritative checker. Each Check launches a fresh Pyodide worker, loads a versioned exercise bundle and the captured source, executes the source, then runs the exercise's verifier functions. The worker returns structured results by verifier name, observed exception and relevant traceback location, elapsed time, and diagnostic measurements where the verifier provides them. A failed import, missing answer, wrong return value, or invalid source is reported as its own reproducible failure. Browser adapters may add diagnostics and controlled fixtures without changing the canonical exercise contract. The build rejects an unsupported verifier rather than silently omitting it.

The controller records source before dispatch. It enforces a per-check deadline and terminates the worker on timeout or Stop, returning a timeout/stopped receipt even if Python cannot clean up. A fresh worker is created for the next check. Exploration Stop first requests a kernel interrupt when supported; hard restart remains available and discards that kernel's state. The site must supply cross-origin isolation headers for cooperative interrupts using `SharedArrayBuffer`; the hard termination path works independently. This is a practice environment with inspectable tests, not an adversarial grading service.

Bundle and pin the tested Pyodide runtime and the packages actually used by the qualified exercises. Qualify NumPy, SciPy, Matplotlib, pytest, and Hypothesis separately where needed; an installed package is not automatically a permitted solution helper. Each manifest entry declares allowed helpers and its checker version. The browser build verifies that all included imports and checks run in the chosen Pyodide distribution.

## MLP lab

Provide a bundled two-dimensional nonlinear classification dataset with fixed train, validation, and test splits and deterministic seeds. Checkpoints cover initialization, forward pass, backward pass, mini-batch training, batched inference, evaluation, and saving/restoring parameters plus optimizer state. The initial model is a small MLP with one optimizer. Each checkpoint can be checked independently; a learner can explicitly substitute a supplied reference component to continue later checkpoints. The receipt records that substitution and never credits the substituted checkpoint as independently solved.

Checks use small deterministic fixtures: shapes and mutation, stable loss, directional finite differences by parameter block, training improvement with broad bounds, inference batch invariance, evaluation without updates, and state round trips. At least one broken-gradient variant must fail with a numerical discrepancy and a targeted hint. The lab includes an intentionally defective variant with a required short diagnosis and regression check. Plots must answer a named experiment question rather than merely decorate the notebook.

## Attempts, persistence, and export

An attempt owns a notebook copy, notes, assistance events, and an append-only list of check receipts. Every receipt stores the exact source snapshot and hash, exercise and checker versions, runtime/package identity, fixture/seed identity, check results, status, timestamp, and any reference substitution. New attempt creates a fresh editable notebook without overwriting old attempts. Browser storage is local convenience; the UI makes export visible.

Export can package one attempt or the entire local history. Each portable archive contains the relevant notebooks, derived `solution.py` files, notes, and machine-readable receipts. Reimport restores attempts and verifies source hashes. The UI separately represents current-source pass, edited-since-check, assisted pass, and later independent re-solve. It does not compute a mastery percentage. Export and reimport preserve the source that was checked, even after subsequent notebook edits.

## Delivery and verification

`browser_gym/` contains the JupyterLite site configuration, frontend extension, checker worker, build/qualification scripts, manifest, capstone, and browser tests. The existing native runner remains usable. The first integrated slice proves `0059 → 0060 → MLP → 0069` in a real browser before extending the manifest to all 12 exercises. Qualification is per exercise: reference passes, starter fails, and selected plausible wrong solutions fail for the intended verifier. No exercise is marked browser-ready solely because its imports load.

The release acceptance run covers a broken and corrected gradient, a bounded training run, an evaluation figure, a widget-driven probe update, a tensor card with correct shape and dtype, a hint and solution reveal recorded as assistance, an isolated check that ignores prior notebook state, a stopped runaway check, a fresh attempt, and an export/reimport whose receipt still matches the exact checked source. CI runs native regression tests, browser build checks, worker tests, and automated browser acceptance. Offline use is a separate qualification: only claim it after caching required assets and verifying a reload with network disabled. No account, sync, leaderboard, cloud compute, AI tutor, arbitrary package installer, or custom autodiff framework is required for this release.

## External references

- [JupyterLite frontend extensions](https://jupyterlite.readthedocs.io/en/stable/howto/extensions/frontend.html)
- [JupyterLite Pyodide kernel](https://jupyterlite.readthedocs.io/en/stable/howto/configure/kernels.html)
- [JupyterLite interactive visualization support](https://jupyterlite.readthedocs.io/en/stable/)
- [Marimo browser notebooks](https://docs.marimo.io/guides/wasm/)
- [Marimo to Jupyter conversion limits](https://docs.marimo.io/guides/coming_from/jupyter/)
- [Pyodide packages](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)
- [Pyodide interrupt buffer API](https://pyodide.org/en/stable/usage/api/js-api.html)
