# Browser gym polish: a focused workout page

Status: ready-for-agent

Sources: the grilling session of 2026-10-08, the terms in `GLOSSARY.md` (workout, drill, rep, split, answer cell), [ADR 0003](../../docs/adr/0003-strip-jupyterlab-inside-jupyterlite-for-the-browser-gym.md), and the layout prototype on branch `prototype/workout-page` (verdict in its `VERDICT.md`; published preview https://claude.ai/artifact/1gpsQhCr3gjtUFFXWu6xhc, variant B).

## Problem Statement

I come back to the browser gym often for refreshers and drills on ML coding fundamentals. Today it feels like using an IDE, not working out:

- Full JupyterLab chrome (menu bar, document tabs, status bar, side icons) takes a fifth of the screen. The gym's only identity is a sideways tab label.
- The brief is a wall of same-weight paragraphs. Math appears as plain text (`mean(a_i-b_j) = mean(a)-mean(b)`) unless the build happens to know that exact equation.
- Exploration cells show their loading code (`exec(await pyfetch(...))`) to the learner.
- The answer cell looks like every other cell.
- The check panel shows a large empty banner before any check has run.
- The toolbar mixes pill buttons with unstyled text links.
- Nothing shows progress across returns. I can't see whether I'm getting faster at a workout, and passing a stage feels like nothing happened.
- The brief, drills, exploration, answer, check results, hints and attempts all stack on one long scroll, so I lose the problem as soon as I start typing.

## Solution

Each workout opens as a calm, mathematical page with one focus at a time (prototype variant B, "Bench"):

- **Side rail:** stage seals with this rep's split times and my best times, plus a quiet active-time rep timer.
- **Left pane:** tabs: **Brief**, **Drills**, **Explore**, **Reps**. One tab is visible at a time.
- **Right pane:** the answer cell and nothing else that competes with it:
  - the **Check** and **Run** actions
  - a grid of verifier squares grouped by stage
  - a detail panel for the selected verifier with its captured case
  - a collapsed scratch cell
- **Look:**
  - JupyterLab's chrome is gone. Only its Python cell editor and its rendered markdown/HTML outputs remain.
  - Serif prose, properly typeset math, a monospace code face, one muted ink-teal accent.
  - A cinnabar seal stamped on each passed stage.
  - Light and dark themes that follow the system, with a toggle.
- **Brief:** question scripts can mark brief sections (`# Goal:`, `# Predict:`, `# Watch out:`, `# Why:`) and write `$…$` math. Unmarked scripts keep working.
- **Drills:** shown as cards. I type a prediction, run the probe, compare, then reveal the reasoning.
- **Reps:** returning to a workout means starting a **new rep**, which resets the answer cell to the starter. Reps record splits (active time to first pass at each stage) and hints used, so I can see myself get faster.
- **Home page:** lists every workout grouped by topic with my best time, rep count, highest stage and last rep. It suggests **Next up** (the workout I've gone longest without), and `⌘K` jumps anywhere.

## User Stories

### Opening and navigating

1. As a returning learner, I want the site root to open a home page of all workouts, so that I choose what to drill instead of landing in one fixed workout.
2. As a returning learner, I want workouts grouped by topic on the home page, so that I can find the area I want to refresh.
3. As a returning learner, I want each workout on the home page to show my best production time, rep count, highest stage passed, and how long since my last rep, so that I can see where I stand at a glance.
4. As a returning learner, I want a **Next up** suggestion at the top of the home page that picks the workout I've gone longest without a rep, so that I spread my practice without planning it.
5. As a returning learner, I want workouts I have never done to count as the longest since a rep, so that new workouts get suggested too.
6. As a learner, I want `?exercise=<id>` links to keep opening that workout directly, so that existing bookmarks and the Pages links still work.
7. As a learner inside a workout, I want `⌘K` / `Ctrl+K` to open a workout switcher I can filter by ID or title and pick with the keyboard, so that I can jump without going home.
8. As a learner, I want the switcher to show how long ago I last did each workout, so that I can make the same choice as on the home page.
9. As a learner, I want a way back to the home page from any workout, so that I can browse again.

### The workout page

10. As a learner, I want no JupyterLab menu bar, document tabs, status bar, file browser, launcher, or side icon bars, so that the page feels like a workout and not an IDE.
11. As a learner on a screen at least 1100px wide, I want the problem on the left and my answer on the right, so that I can read the problem while I type.
12. As a learner on a narrower screen, I want the layout to stack into one column without horizontal page scroll, so that it still works on a small laptop window.
13. As a phone visitor, I want the page not to break, so that I can at least read a workout, even though solving there is not a goal.
14. As a learner, I want the left pane split into **Brief**, **Drills**, **Explore**, and **Reps** tabs with only one visible at a time, so that each view has one focus.
15. As a learner, I want the tab I last had open remembered per workout, so that returning puts me where I was.
16. As a learner, I want the tabs to be keyboard-operable like a standard tab list, so that I can move between them without the mouse.
17. As a learner, I want each tab label to show a count where useful (number of drills, explore panels, reps), so that I know what's inside before opening it.
18. As a learner, I want the workout ID, title, and topics shown above the tabs, so that I always know which workout I'm in.

### Brief

19. As a learner, I want the brief to show the **Goal** first and set apart, so that I know what to build before anything else.
20. As a learner, I want math in the brief properly typeset, inline and display, so that formulas read like mathematics.
21. As a learner, I want **Watch out** sections visibly marked as traps, so that I notice the mistake the workout is designed to catch.
22. As a learner, I want **Why** sections set apart as explanation, so that I can skip or read them deliberately.
23. As a learner, I want each stage's contract (baseline, follow-up, production) in the brief, with the current stage emphasised and later stages de-emphasised, so that I focus on the stage I'm on.
24. As a learner, I want the brief to say which stage each section belongs to and whether I've passed it, so that I can orient myself.
25. As a learner, I want the hints and the reference solution reachable from the brief, so that help is near the problem statement.
26. As a learner, I want code identifiers in prose shown in the code face, so that names are unambiguous.

### Drills

27. As a learner, I want each drill shown as a card with a prompt and its probe code, so that I know exactly what to predict.
28. As a learner, I want a prediction field on each drill, so that I commit to an answer before seeing the result.
29. As a learner, I want **Run** on a drill to execute its probe in the notebook kernel and show the real output beside my prediction, so that I can check my intuition.
30. As a learner, I want **Reveal reasoning** on a drill after running it, so that I can learn why when I was wrong.
31. As a learner, I want drills never graded and never sent to Check, so that predicting stays low-pressure.
32. As a learner, I want my drill predictions saved with the current rep, so that I can compare them across reps.

### Explore

33. As a learner, I want exploration panels to show only their rendered output (tensor cards, plots, sliders), with no loader code, so that I see the idea and not the plumbing.
34. As a learner, I want an explicit **Run** control on an exploration panel that has not run yet, so that I can start it when the kernel is ready.
35. As a learner, I want exploration widgets themed to match the page in both light and dark, so that plots don't flash white in dark mode.

### Answer cell and checking

36. As a learner, I want the answer cell clearly labelled as the only cell that is checked, so that I know where my solution goes.
37. As a learner, I want the answer cell editor themed in the gym's palette and code face, in both themes, so that the editor is pleasant to work in for long sessions.
38. As a learner, I want `Shift+Enter` to run the answer cell in the notebook kernel, so that I can try my code without checking.
39. As a learner, I want `⌘/Ctrl+Enter` to check my answer against the current stage, so that checking is one keystroke.
40. As a learner, I want the Check button to name the stage it will check (for example "Check follow-up"), so that I know what I'm about to run.
41. As a learner, I want to choose a different stage or checkpoint to check, so that I can recheck an earlier stage or jump ahead deliberately.
42. As a learner, I want `⌘.` / `Ctrl+.` to stop a running check, so that I can abandon a slow check without the mouse.
43. As a learner, I want a quiet running indicator while a check is in its fresh worker, so that I know it's working without a big banner.
44. As a learner, I want check results shown as a grid of squares, one per verifier, grouped by stage, so that I can see the whole picture at once.
45. As a learner, I want stages I haven't reached shown as hatched, inert squares, so that I see what's coming without noise.
46. As a learner, I want the first failing square selected automatically after a check, so that I land on what to fix.
47. As a learner, I want the detail panel for a selected verifier to show its name, status, timing, error, and captured case (inputs, expected, actual), so that I understand the failure.
48. As a learner, I want **Load case into scratch** in the detail panel to keep working as today, so that I can replay a failing case in the kernel.
49. As a learner, I want nothing shown in the results area before my first check except a one-line prompt, so that an empty state doesn't shout.
50. As a learner, I want stopped and timed-out checks to keep their source-bound receipts and saved-versus-discarded explanation, so that recovery behaviour does not regress.
51. As a learner, I want a small readout line with worker runtime, NumPy version, and checker version, so that check provenance stays visible but quiet.
52. As a learner, I want the scratch cell available but collapsed by default under the answer, so that I can experiment without clutter.

### Reps, splits, and stamps

53. As a returning learner, I want **New rep** to save my current answer to the rep log and reset the answer cell to the starter, so that I can drill the workout from scratch.
54. As a learner, I want a rep timer in the side rail that counts only active time, pausing while the tab is hidden or I've been idle for more than two minutes, so that a lunch break doesn't count against me.
55. As a learner, I want the timer to stop once the final stage is passed, so that the rep's time is final.
56. As a learner, I want the timer dimmed and optionally hidden, so that it motivates without pressuring.
57. As a learner, I want my split recorded the first time I pass each stage in a rep, so that I can see how long each stage took.
58. As a learner, I want each split shown in the side rail next to my best for that stage, faster or slower, so that I can see improvement immediately.
59. As a learner, I want a passed stage's seal stamped with a short ink-stamp animation, so that passing feels like an event without confetti or sound.
60. As a learner who prefers reduced motion, I want the stamp shown without animation, so that the page respects my settings.
61. As a learner, I want a one-line rep summary after the final stage, with production time, difference from my best, and hints used, plus a Next up link, so that a finished rep has a clear ending.
62. As a learner, I want the **Reps** tab to list every rep of this workout with date, splits per stage, and hints used, with my best rep marked, so that I can see my history.
63. As a learner, I want to resume an earlier rep from the Reps tab, so that the existing resume behaviour stays available.
64. As a learner, I want using a hint or revealing the solution recorded against the rep, so that assisted reps are distinguishable from clean ones.
65. As a learner, I want reps and splits included in **Export all** archives and restored on import, so that my history survives a browser wipe.
66. As a learner with attempts saved before this change, I want them to appear as reps with unknown splits rather than disappear, so that I don't lose history.
67. As a learner, I want notes on a rep, so that I can jot what tripped me up.

### Look and feel

68. As a learner, I want light and dark themes that follow my system setting, so that the page is easy on the eyes by default.
69. As a learner, I want a theme toggle that overrides the system and is remembered, so that I can choose.
70. As a learner, I want prose in a calm serif, code and data in a monospace face, and one muted accent, so that the page feels mathematical and quiet.
71. As a learner using the offline build, I want all fonts and typesetting assets bundled with the site, so that the page looks the same with no network.
72. As a keyboard user, I want every control to have a visible focus state, so that I always know where I am.
73. As a screen-reader user, I want verifier squares, seals, and the timer labelled with their meaning, so that state isn't only conveyed by colour.

### Labs

74. As a learner doing the MLP lab, I want the side rail to show its checkpoints instead of the three stages, so that the lab fits the same page.
75. As a learner doing the MLP lab, I want the verifier grid grouped by checkpoint, with checkpoint evidence and reference substitutions visible in the detail panel, so that nothing the current lab panel shows is lost.
76. As a learner doing the MLP lab, I want splits recorded per checkpoint, so that reps work the same way.

### Authoring

77. As a workout author, I want to mark brief sections in question-script comments with `# Goal:`, `# Predict:`, `# Watch out:`, and `# Why:`, so that the brief gets structure without leaving the `# %%` script format.
78. As a workout author, I want to write `$…$` and `$$…$$` math in question-script comments, so that I don't need the build to know each equation in advance.
79. As a workout author, I want an unmarked question script to render as plain prose sections as today, so that migrating the 100+ other scripts is optional.
80. As a workout author, I want a cell marked as a drill in the question script to become a drill card (prompt from its comments, probe from its code), so that drills keep running natively in Emacs too.
81. As a workout author, I want drill reasoning to live in the eval script, so that the native question script doesn't spoil the answer.
82. As a workout author, I want the build to reject malformed markers, unbalanced math delimiters, and drills without reasoning, so that mistakes fail at build time and not in the browser.
83. As a workout author, I want topics on every browser workout's catalog entry, so that the home page can group them.

## Implementation Decisions

### Shell: stay inside JupyterLite (ADR 0003)

- JupyterLite's `disabledExtensions` setting turns off the menu bar, document tabs, status bar, file browser, launcher, running-sessions panel, table of contents, and command palette UI.
- The gym extension owns the whole layout and builds it from Lumino widgets: home page, side rail, tabbed left pane, right pane.
- **Intended approach:** two views of the same notebook model, sharing one kernel session. Each view shows only certain cells, chosen by their `dojo.role` metadata:
  - the left view shows instruction, drill and exploration cells for the active tab;
  - the right view shows the answer cell and the scratch cell.

  If multiple views prove unworkable, the fallback is a single notebook view that relocates cells by role. The test points stay the same either way.
- The JupyterLab notebook cell editor and output renderers are kept and themed by overriding JupyterLab's CSS variables with the gym's design tokens, so one token set drives both themes.

### Design tokens (from the prototype, variant B)

| Role | Choice |
| --- | --- |
| Prose | Newsreader |
| Code and data | JetBrains Mono |
| Math | the typesetter's bundled math fonts |
| Accent | muted ink-teal |
| Passed-stage seal | cinnabar (used for nothing else) |
| Pass / fail | semantic green and red, separate from the accent |
| Light page | cool paper ground |
| Dark page | deep green-grey ground |

- All tokens are defined once with light and dark values, and the theme is chosen by system preference or an explicit toggle.
- Fonts ship as local assets in the offline build and are listed in the offline manifest.

### Math typesetting

- Briefs and drills keep using JupyterLab's bundled math renderer. The prototype used KaTeX only for convenience.
- The catalog build's fixed lookup table of five equations is replaced by author-written `$…$` / `$$…$$`, which the build passes through escaped and wrapped as math. The table can stay as a fallback for unmigrated scripts until they are converted.

### Question-script markers

- A comment line starting with `# Goal:`, `# Predict:`, `# Watch out:`, or `# Why:` starts a section of that kind. Following comment lines continue it until a blank comment line or a new marker.
- Stage headings already recognised by the build (`Baseline —`, `Follow-up —`, `Production challenge —`) keep their meaning.
- A `# %% [drill]` cell header marks a drill. This follows the existing `# %% [answer]` convention, so tagged-cell selection (ADR 0001) keeps working, and the native judge ignores drills.
- Drill reasoning lives in the eval script, in a comment block marked `# Drill <n>:` where `<n>` is the drill's order in the question script.
- The catalog entry gains structured brief data: an ordered list of sections, each `{kind, stage?, html}`, and a list of drills, each `{title, prompt_html, probe_source, reasoning_html}`.
- The generated notebook gains a `drill` cell role holding the probe code.

### Rep data (extends the existing attempt store; no new store)

- "Attempt" is renamed **rep** in every learner-facing label ("New rep", "Rep log", "Resume rep 2"). Internal names may stay.
- The rep record gains:
  - `activeSeconds`: accumulated active time, saved whenever the record is updated;
  - `predictions`: a map from drill index to text.
- Each check receipt gains `activeSecondsAtCheck`.
- A stage's split is the `activeSecondsAtCheck` of the rep's first passing receipt for that stage. It is derived, not stored separately.
- Older records and archive imports without these fields load with `activeSeconds = 0` and unknown splits.
- Active time stops accruing while the page is hidden, after 2 minutes without input, and once the final stage or checkpoint has passed.
- "Best" for a stage is the minimum split for that stage across the workout's completed reps.

### Home page and Next up

- The site root without an `exercise` parameter opens the home page. `?exercise=<id>` keeps opening that workout directly.
- Topics come from catalog `concepts`. Workouts with empty concepts (0061–0066) get topics added in their catalog metadata.
- Next up picks the workout whose most recent rep is oldest. Workouts never repped rank first, in learning-path order.

### Keyboard

| Shortcut | Action |
| --- | --- |
| `Shift+Enter` | run cell (kernel) |
| `⌘/Ctrl+Enter` | check the current stage |
| `⌘/Ctrl+K` | workout switcher |
| `⌘/Ctrl+.` | stop a running check |
| Arrow keys | move within the tab list |

Shortcuts must not fire while typing in a drill prediction field, except `⌘/Ctrl+K`.

### Verifier grid shape (from the prototype; trimmed)

```ts
// One square per verifier, grouped by stage (or checkpoint for labs).
type SquareState = 'pass' | 'fail' | 'not-run' | 'locked'; // locked = stage beyond the checked one
interface VerifierSquare { stage: string; name: string; state: SquareState; elapsedMs?: number; case?: CapturedCase }
// After a check: select the first failing square, else none.
```

### Order of delivery

1. 0059 is the pilot workout, since it uses every element.
2. The MLP lab is second, since it uses checkpoints.
3. The remaining browser workouts are migrated to markers afterwards.

## Testing Decisions

Good tests check behaviour you can see from outside, in a browser or in the build's output: what the learner sees and can do, and what the catalog build emits. They never assert on CSS class names, widget internals or private helpers. Test the high-value paths end to end rather than unit-testing layout code.

**1. The built site (Playwright e2e)**

The primary test point. Prior art: `first_exercise.spec.ts`, `attempts.spec.ts`, `recovery.spec.ts`, `cases.spec.ts`, `mlp.spec.ts`. Coverage:

- JupyterLab's chrome is absent, as a guard against plugin-ID drift in `disabledExtensions`.
- The home page groups workouts, shows Next up, and opens a workout.
- `⌘K` filters and jumps between workouts.
- Each tab shows its own content.
- Drill flow: predict, Run, see output, Reveal reasoning; the prediction persists with the rep.
- Explore panels show no loader code and still render widgets.
- Checking with `⌘Enter` fills the verifier grid and selects the first failing square, whose detail shows the captured case.
- Load case into scratch still works.
- New rep resets the answer to the starter, and the Reps tab lists the previous rep.
- Splits use Playwright's fake clock: active time excludes hidden-tab time, and the first pass records a split.
- The stamp appears on pass.
- The theme toggle overrides the system preference and is remembered.
- The offline build still passes the qualified recovery scope.
- Stop/timeout receipts are unchanged.

About 60 existing lookups use renamed labels ("Check solution", "New attempt", "Attempt notes", "Resume attempt N"), and those tests are updated as part of this work.

**2. The catalog build (pytest)**

Prior art: `tests/test_build_catalog.py`. Coverage:

- Markers become typed brief sections, and `$…$` passes through escaped and wrapped as math.
- `# %% [drill]` cells become drill entries paired with their eval-script reasoning.
- Unmarked scripts still produce today's prose brief.
- The build rejects unbalanced math, unknown markers, drills without reasoning, and reasoning without a drill.
- Every browser workout has topics.
- Drill cells are never part of submitted source.

**3. The attempt store (vitest)**

Prior art: `src/attempts.spec.ts`. Coverage is limited to the new persisted fields:

- Old records and archives without `activeSeconds` / `activeSecondsAtCheck` / `predictions` load.
- Exported archives keep them, and import verifies them alongside the existing SHA-256 receipt check.
- The split for a stage is derived from the first passing receipt.

The native workflow's checks (`make test`, `make eval`, and the MLP native reference) must keep passing unchanged. That's the guarantee that markers and drill cells don't break ADR 0001's tagged-cell contract.

## Out of Scope

- A spaced-repetition review queue. Next up uses only "longest since last rep"; the rep data is shaped so a queue can be added later.
- Rewriting brief prose for all ~100 non-browser dojo scripts. Markers are opt-in, and only the 13 browser workouts are migrated.
- A standalone app that doesn't use JupyterLite (rejected in ADR 0003).
- Keeping a "show full JupyterLab" escape hatch.
- Optimising for solving on phones.
- Gamification beyond the seal stamp and splits: no XP, streaks, confetti, or sound.
- Cross-device sync or cloud backup of reps. Export/import stays the durability story.
- Changes to the checker worker, verifier semantics, cumulative stages (ADR 0002), or recovery semantics beyond relabelling.

## Further Notes

- **Decisions made while writing this spec** (open to challenge):
  - the `# %% [drill]` cell header;
  - drill reasoning in the eval script under `# Drill <n>:`;
  - the two-views shell approach;
  - keeping JupyterLab's bundled math renderer instead of adding KaTeX.
- **Risk:** plugin IDs in `disabledExtensions` can change across JupyterLite upgrades. The e2e chrome-absence check is the guard (ADR 0003 consequence).
- The learner prefers focus over density. When a detail isn't covered here, show one thing at a time and keep the controls around it minimal (prototype verdict).
- The `GLOSSARY.md` terms *workout*, *drill*, *rep*, *split* and *answer cell* are the vocabulary for UI copy and tickets.
