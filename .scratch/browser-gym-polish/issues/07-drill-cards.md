# 07: Drill cards

**What to build:** A **Drills** tab joins Brief, Explore and Reps. Each drill (see the glossary) is a card: a short prompt, a field for the learner's prediction, a Run button that executes the drill's probe in the kernel, and Reveal reasoning, which shows the author's explanation. Drills are never checked, but the predictions are saved with the rep. 0059 gets the first drills.

**Blocked by:** 06

**Status:** done

- [x] Question scripts can contain `# %% [drill]` cells. The matching reasoning lives in the eval script under `# Drill <n>:` blocks, so it stays hidden from the learner's script.
- [x] The catalog emits drills shaped `{title, prompt_html, probe_source, reasoning_html}`, and the notebook gains a `drill` cell role.
- [x] The build rejects a drill with no reasoning, reasoning with no drill, and duplicate numbers.
- [x] Reveal reasoning works whether or not the drill has been run, and is never required before running.
- [x] Predictions are saved per drill in the rep's `predictions`, survive reload, and appear in export.
- [x] Drill cells are inert in the native workflow: `make test` and `make eval` pass with 0059's drills present.
- [x] pytest covers drill extraction and its rejections. e2e covers predict → run → reveal on a 0059 drill and that the prediction persists.

## Comments

- Drill cells are numbered from 0 in notebook metadata and `predictions`; authors write `# Drill 1:` for the first drill. Imports inside drill cells are not carried into the answer cell.
- Each card is a group named by its title (via `aria-labelledby`, since JupyterLab rewrites a cell's own `aria-label` as outputs change). The prediction field blocks Lumino shortcuts with `data-lm-suppress-shortcuts`; ticket 08's ⌘K must therefore listen at document level, not as a Lumino keybinding, to keep the spec's ⌘K exception.
- Predictions save 300 ms after typing pauses, with one timer per drill so typing in one drill never cancels another's save.
- Workouts without drills hide the Drills tab, and arrow keys skip it.
- Found while requalifying: the startup rep restore replaces the notebook model, so tests must wait for Check to be enabled before typing (fixed in `brief.spec.ts`); a rep notebook opened outside the Bench now has drill cells before the answer, so the `attempts.spec.ts` fallback picks the code cell just before scratch.
- Native check: `0059_stable_cross_entropy_eval.py` passes 9/9 with drills present. `make test` has 4 errors only from modules needing `torch`, which the browser-gym venv lacks; unrelated.
