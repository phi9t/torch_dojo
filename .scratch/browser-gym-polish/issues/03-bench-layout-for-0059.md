# 03: Bench layout for 0059

**What to build:** Opening 0059 shows the Bench layout chosen in the prototype. A narrow side rail on the left gives the workout's identity and stage seals. The left pane has **Brief**, **Explore** and **Reps** tabs, and one tab is visible at a time. The right pane holds only the answer cell, Check and Run, and a collapsed scratch cell. The current check panel and rep list move into the new places with their behaviour unchanged. The grid and timer come in later tickets.

The prototype that settled this layout is on the `prototype/workout-page` branch (variant B, "Bench"; see its VERDICT.md).

**Blocked by:** 02

**Status:** done

- [x] Both panes are views of the same notebook model, filtered by each cell's `dojo.role`. The left shows instruction and exploration cells; the right shows answer and scratch cells. If two views prove unworkable, use the spec's fallback (one view that relocates cells) and record why in this ticket's Comments.
- [x] Brief, Explore and Reps form an accessible tablist: arrow keys move between tabs, and the chosen tab is remembered per workout.
- [x] Explore panels show only their output with a Run control; their code stays available behind a disclosure.
- [x] The scratch cell is collapsed by default, opens with one click, and its state never reaches the checked source.
- [x] Shift+Enter runs the focused cell, ⌘/Ctrl+Enter checks the answer cell, and ⌘/Ctrl+. stops a running check.
- [x] At narrow widths the two panes stack (rail above, then left, then right) with no horizontal page scroll. Solving on a phone is out of scope, but reading the brief on one works.
- [x] Existing 0059 e2e behaviours (run, check starter/reference/unstable, hints, reveal, notes, export) pass in the new layout.

## Comments

- Two views work. The answer pane is the opened `NotebookPanel`, and the problem pane is `docManager.cloneWidget` of it: same model, same kernel. Each view hides cells by pane (`src/cellPanes.ts`, by `dojo.role` with a fallback on the generated opening line for reps saved before roles). Two workarounds were needed:
  - Notebook windowing is off (`overrides.json`, `windowingMode: none`), so hidden cells are real widgets.
  - Each view disposes the output area of every code cell it never shows. Otherwise both views render every output, which duplicates element ids and runs exploration scripts twice.
- Workout id, title and topics sit above the tabs (spec story 18), not in the rail. The rail has the brand, the stage seals, kernel controls, Learning path and the theme toggle. The seals show passed/current/upcoming now; splits and the stamp animation are ticket 05.
- The catalog now states `stage_order` (`cumulative` for dojo stages, ADR 0002; `independent` for MLP checkpoints).
- Hints stay under the answer for now. Linking them from the brief is ticket 06.
- `cases.spec.ts` and `qualification_0061.spec.ts` were already failing on HEAD because the 0059 verifiers and the 0061 reference changed. Their expectations were refreshed: the mutant now fails 2 baseline verifiers, and the empty-cluster mutant was rewritten for the vectorized reference.
