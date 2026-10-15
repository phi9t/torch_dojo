# Browser ML Coding Gym

This standalone directory builds a JupyterLite site with 30 dojo workouts and
the NumPy MLP modeling lab, grouped on the home page by topic:

| Topic | Workouts |
| --- | --- |
| Training | `0059`, `0060`, `0061`, `0030`, `0050`, `0075`, `0100`, MLP lab |
| Evaluation | `0069`, `0073`, `0074`, `0094` |
| Geometry | `0062`, `0063`, `0064`, `0070` |
| Sensors | `0065`, `0066` |
| Arrays and indexing | `0067`, `0068`, `0071`, `0076`, `0077` |
| Parsing | `0035`, `0072`, `0078`, `0079` |
| Reinforcement learning | `0082`, `0083`, `0091` |
| Systems | `0080` |

These are the exercises that run on NumPy and the standard library; the
PyTorch exercises need the native workflow, since Pyodide has no PyTorch. The
dojo notebooks and checker bundles are generated from the current tagged
question and eval scripts in `dojo/`; the MLP lab is generated from canonical
Python sources under `browser_gym/labs/mlp/`. A workout without a lesson in
`browser_gym/lessons/` simply has no Explore panels. To add one, give it an
entry in `QUALIFIED_DOJO_EXERCISES` in `build_catalog.py` (topic first in
`concepts`), a typed brief and drills, and a row in
`extension/tests/e2e/numpy_workouts.spec.ts`.

```sh
make -C browser_gym setup
make -C browser_gym build
make -C browser_gym serve
```

Open `http://localhost:8000/lab/` for the home page. It lists every workout
grouped by topic with your best time,
rep count, highest stage passed and how long since your last rep. **Next up**
suggests the workout you have gone longest without: one you have never repped
first, in learning-path order, otherwise the one whose last rep is oldest. A rep
counts once it has a check or some active time. `?exercise=<id>` links open a
workout directly, for example `http://localhost:8000/lab/?exercise=0060` or
`?exercise=mlp` for the MLP lab. None of JupyterLab's menus, tabs, status bar
or side panels are shown; `jupyter-lite.json` disables them (ADR 0003).

### The Bench

Each workout opens on the Bench, one focus at a time:

- **Side rail:** **← All workouts** (home), the stage seals (the lab's
  checkpoints), this rep's active-time timer, the rep summary, the kernel
  status with interrupt and restart, the **Learning path**, and the theme
  toggle. The page follows the system light or dark setting until you pick a
  theme, and remembers that choice.
- **Left pane:** **Brief** (the goal, each stage's contract, Predict, Watch out
  and Why sections, typeset math, and links to hints and the reference
  solution), **Drills** (prediction cards), **Explore** (visual panels with
  their own Run control; their loading code stays behind **Show code**) and
  **Reps** (every rep with its splits, help used and notes; resume, **Export
  rep**, **Export all** and **Import reps**). The last tab you used is
  remembered per workout.
- **Right pane:** the answer cell, the only cell that is checked, with Run, the
  stage to check, Check and Stop above it. After a check, a grid of verifier
  squares, grouped by stage (or checkpoint), selects the first failing square;
  its detail panel shows the error, timing, captured case and **Load case into
  scratch**. A scratch cell opens below on request; **Run scratch** reruns the
  answer and then the scratch cells. **Expand** gives the answer the whole page
  beside the rail (remembered across workouts).

Below 1100px the rail, problem and answer stack into one column.

For dojo workouts, checks start at the baseline stage and the stage menu adds
follow-up and production verifiers cumulatively (ADR 0002). For the MLP lab, the
stage menu checks the independent `init`, `forward`, `backward`, `train`,
`inference`, `evaluate`, `state` and `diagnose` checkpoints in any order; the
grid keeps each checkpoint's latest result, and a pass that used a reference
component (`REFERENCE_OVERRIDES = {"loss_and_grads"}`) says so and never counts
as independent credit. Each check creates a fresh Pyodide worker. Browser checks
submit only the answer cell, so imports and helpers an answer needs must be in
that cell.

### Keyboard

| Shortcut | Action |
| --- | --- |
| `Shift+Enter` | run the cell in the notebook kernel |
| `⌘/Ctrl+Enter` | check the answer at the selected stage |
| `⌘/Ctrl+.` | stop a running check |
| `⌘/Ctrl+Shift+Enter` | run the answer, then each scratch cell (also **Run scratch**) |
| `⌘/Ctrl+Shift+E` | expand the answer to the whole page, or show the problem again |
| `⌘/Ctrl+K` | open the workout switcher (filter by id, title or topic; arrows and Enter) |
| `Tab` | accept a completion; NumPy and your own names appear as you type (`Ctrl+Space` asks) |
| Arrow keys | move within the tab list or the verifier squares |

Shortcuts don't fire while you type a drill prediction, except `⌘/Ctrl+K`.

The **Shortcuts** panel (beside **Run scratch**) has an opt-in **Emacs keys**
toggle: `Ctrl+A/E/F/B/N/P/D/K/Y` and `Alt+F/B/D` in code editors. With it on,
`Ctrl+K` inside an editor deletes to the end of the line, and the workout
switcher opens with `Ctrl+K` from outside an editor.

### Reps and splits

Returning to a workout means starting a **New rep**, which saves the current
answer to the rep log and resets the answer cell to the starter. The timer counts
only active time: it pauses while the tab is hidden or after two minutes without
input, and stops once the final stage (or every checkpoint) passes. A stage's
**split** is the active time when it first passed in that rep; the rail shows it
next to your best, and the passed stage's seal is stamped (without animation if
you prefer reduced motion). After the final stage a one-line summary gives the
time, the difference from your best, hints used, and a Next up link. Reps saved
before timing existed show unknown splits as "—".

Rep changes save the answer you are leaving before opening the next one. If a
change fails, the gym restores the previous rep; loading and recovery time do
not count as practice. A check that finishes after you switch still belongs to
the rep you submitted. After a failed check, start with its highlighted example
and try it in scratch. After editing, the old result stays visible until you
check again.

### Drills

A drill is a short prediction exercise: read the prompt and probe code, type your
prediction, **Run** the probe in the notebook kernel, compare, then **Reveal
reasoning**. Drills are never checked; predictions are saved with the rep and
included in exports.

### Authoring briefs and drills

Question scripts opt in to a structured brief with a `# Goal:` line. Then:

- `# Goal:`, `# Predict:`, `# Watch out:` and `# Why:` each start a section; it
  continues over the following comment lines until a blank comment line (`#`),
  code, or the next marker.
- Stage headings (`# Baseline — …`, `# Follow-up — …`, `# Production — …`) scope
  the sections after them. The MLP lab's `labs/mlp/brief.py` uses a heading per
  checkpoint (`# Forward — …`).
- Math is `$…$` inline or `$$…$$` display, within one line; `\$` is a literal
  dollar, and dollars inside backticks are code.
- A `# %% [drill]` cell is a drill: its first comment line is the title, the
  remaining comment lines the prompt, and its code the probe. Its reasoning
  lives in the eval script as `# Drill <n>:` (1-based, in drill order), so the
  learner's script never gives the answer away; the MLP lab keeps its drills in
  `labs/mlp/brief.py` and their reasoning in `labs/mlp/reasoning.py`, outside
  the checker bundle. Drill cells are inert in the native workflow.

The catalog build rejects unknown or misspelt markers, unbalanced `$`, a stage
the workout doesn't have, and drills without reasoning (or reasoning without a
drill), naming the script and line. Scripts without `# Goal:` keep the plain
prose brief.

Exploration cells contain a short loader, while their Python setup lives in
`browser_gym/dist/gym-assets/lessons/`. Older saved notebooks with embedded setup
code open with that code collapsed. Exploration cells are never checked.

Run the MLP native reference checks from the repository root:

```sh
python -m browser_gym.labs.mlp.checks --candidate reference --checkpoint all
```

## Release-qualified offline build

The release artifact is built with:

```sh
make -C browser_gym offline-build
```

`offline-build` runs the catalog build, builds the JupyterLab extension, builds
the JupyterLite site, copies the checker and exercise assets, rewrites the
checker worker to load local runtime files, disables PyPI fallback, and writes
`browser_gym/dist/gym-assets/offline-manifest.json`. The generated offline site
loads Pyodide from `./static/pyodide/pyodide.mjs`, loads piplite wheels from
`./pypi/all.json`, and serves the checker worker's wheels from
`./gym-assets/pypi/`.

The tested offline scope is a previously loaded site. A browser must first load
the site while the local HTTP server is reachable so the JupyterLite service
worker, Pyodide runtime, wheels, notebooks, checker bundles, and Gym extension
assets are cached. After that, the qualified scope is reload of that same site
with browser networking disabled, then running the `0059` probe cell, tensor
card, widget update, isolated check, and rep export/reimport. This does not
claim first-load-without-network, arbitrary package installation, cross-browser
sync, cloud backup, or native Python execution in the browser.

Runtime and package identities for the tested artifact:

- Browser automation: Google Chrome `153.0.8010.12` through Playwright `1.63.0`
  on `DOJO_TEST_PORT=8116`.
- JupyterLite: `jupyterlite-core==0.8.5` and
  `jupyterlite-pyodide-kernel==0.8.6`.
- Frontend packages: JupyterLab `4.5.x`, `@jupyterlab/application` `^4.5.11`,
  `@jupyterlab/notebook` `^4.5.11`, and `@jupyter-widgets/jupyterlab-manager`
  from the JupyterLite build.
- Pyodide: npm package `pyodide@314.0.6`, Python `3.14.2`, ABI `2026_0`,
  platform `emscripten_5_0_3`, npm integrity
  `sha512-BKDTJyIqFxC4BExLqeRS3f5xvXZIjOt8C3zGLN/Cc7tFxSwvKVhVkchQQ2AGLOtR4YrVOIFxbV8poyDOOmWwxQ==`.
- Qualified local packages: NumPy `2.4.6`, micropip `0.11.1`,
  sortedcontainers `2.4.0`, Hypothesis `6.168.5`, comm `0.2.3`,
  ipywidgets `8.1.9`, jupyterlab_widgets `3.0.17`, and
  widgetsnbextension `4.0.16`.

Important release hashes from the generated manifest:

- `static/pyodide/pyodide.mjs`:
  `69e3f6ccec3e14b465df60be577ca62f536251406b9a00cce019eac5252a2495`
- `static/pyodide/pyodide.asm.wasm`:
  `3a0a00dfeaa348ac20f9ef09904233d32d33f644339662d4af368f8a2010f37a`
- `static/pyodide/python_stdlib.zip`:
  `80c5be6babfe03297069703410c3c29404dcf2525d2b128746bae5536f94831f`
- `pypi/hypothesis-6.168.5-cp314-cp314-pyemscripten_2026_0_wasm32.whl`:
  `bc5cc310f9f86ec62f0d0dd7eea5a4788f18ec793b70ee2c7163b916768e1057`
- `pypi/ipywidgets-8.1.9-py3-none-any.whl`:
  `f2b8cbcaae10252b809fbe4d7470db75c09b769a32cbf816d20e5ca6d3c5a79d`
- `gym-assets/checker-worker.js`:
  `4b17b5084b4b729145c8ad0e6c7918e657c0176a0fab77cec8e73605632375bd`
- `gym-assets/catalog.json`:
  `06f4153c34f21744f7a70151fa967421f36daa339f6ee2e42570bcb16dd5beb4`

The complete per-asset hash list is in
`browser_gym/dist/gym-assets/offline-manifest.json` after every offline build.
It covers the Pyodide runtime, wheels, checker and catalog, lessons, and every
file of the gym extension, including its bundled Newsreader and JetBrains Mono
fonts, plus MathJax's fonts, so nothing loads from a font CDN, and
`make -C browser_gym offline-build` verifies the result against it. Because
`dist/` is generated, rebuild the offline artifact and use that manifest as the
release checksum source. To check a built `dist/` again later, naming any
missing or altered asset:

```sh
python browser_gym/offline_build.py --verify
```

## Persistence, export, and native route

Reps are stored in local browser storage and JupyterLite contents for
convenience. Browser storage can be cleared by the browser, is not synced across
profiles or machines, and should not be treated as durable backup. Use **Export
rep** or **Export all** for portable JSON archives; imported archives verify
that each receipt's checked source still hashes to the stored SHA-256.

Each browser check runs in a fresh Pyodide worker. The exploration notebook
kernel is persistent until interrupted or restarted. **Interrupt** requests a
cooperative kernel interrupt when the browser has the required isolation support;
**Restart kernel** is the hard recovery path and discards only live kernel state,
not saved reps, notebooks, notes, or receipts.

The native route remains the repository's Python workflow. Use the root
`make test`, root `make eval`, and the MLP command above when browser storage,
JupyterLite, or the local offline scope is not the right execution environment.

Run focused checks with `make -C browser_gym test`. To run the Playwright suite
for this ticket on the release port, use `DOJO_TEST_PORT=8116 npm --prefix
browser_gym/extension run test:e2e -- recovery_offline.spec.ts` after
`offline-build`. The site build requires Node.js, npm, and Python 3.11 or newer.

### Inspect and replay check examples

After a check, the first failing square is selected. Its detail shows the
error, then **Where it failed**: the line of the check whose assertion failed,
with the lines around it, and, when your answer raised, your answer's line
first. **The failing example** is the last call the check made before it
failed, with its inputs, the expected output and yours, followed by the check's
own variables at the failing line; the check's earlier calls fold away below.
Select any square to see its captured inputs, expected and actual outputs,
exceptions, array shapes and dtypes. A square's status refers
to the entire verifier, not each individual call; some calls intentionally raise
validation errors. The detailed check log and receipts are expandable.

**Load case into scratch** installs examples in the notebook kernel and adds a
small scratch cell for the failing example. Use `gym_cases`, `gym_passed`, and `gym_failed` to browse
examples, `gym_case(0)` for an independent copy, and `gym_try(0)` to call your
current notebook function with copied inputs. Run the answer cell first.
`gym_source_hash` identifies the submitted source behind the captured results.
Reload the examples after restarting the kernel. Loading examples does not
execute the submitted answer or change what is sent to the checker.

Samples are saved with check receipts and restored with reps. Capture keeps
the first three and last three calls per function per check. Arrays larger than
256 elements and unsupported objects are marked preview-only; those inputs
cannot be replayed. Expected outputs are computed from the canonical reference
when the captured inputs can be reconstructed. MLP checks also retain their
checkpoint evidence and reference-assistance classification.

## GitHub Pages

The live dojo is at https://phi9t.github.io/torch_dojo/lab/.
Pushes to `main` run `.github/workflows/pages.yml`: tests, a self-contained
JupyterLite build, and GitHub Pages deployment. The Pages publishing source
must be **GitHub Actions**. Only `browser_gym/dist` is published.

For a local build under the same repository prefix:

```sh
DOJO_BASE_URL=/torch_dojo/ make -C browser_gym offline-build
```

`DOJO_BASE_URL` defaults to `/` for the usual local server. It must be an
absolute directory path with a trailing slash. The e2e specs navigate relative to
Playwright's base URL, so the whole suite also runs against a site served under
the prefix. Serve a directory in which `torch_dojo` is the prefixed `dist` (for
example a symlink `torch_dojo -> browser_gym/dist`), then run
`DOJO_TEST_PORT=<port> DOJO_PREFIX=/torch_dojo npm --prefix browser_gym/extension
run test:e2e`. The Pages workflow's `tests/pages-smoke.cjs` opens the deployed
home page, a workout's brief and explore panel, a passing check, and a reload.

GitHub Pages does not supply
cross-origin isolation headers; use **Restart kernel** if a cooperative
interrupt is unavailable.

## Motif reviews

The home page includes a searchable motif library and a spaced-review queue.
Cards link back to qualified workouts; native-only workout numbers are labelled
Jupyter / PyTorch. Reviews have their own local archive and do not affect reps
or checks. See the [motif review guide](../docs/practice/motif-review.md) for
ratings, notebook practice, and browser/native archive transfer.
