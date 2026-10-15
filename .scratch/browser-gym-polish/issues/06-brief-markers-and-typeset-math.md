# 06: Brief markers and typeset math

**What to build:** Authors can structure a workout's brief with `# Goal:`, `# Predict:`, `# Watch out:` and `# Why:` comment markers, and can write math as `$…$` and `$$…$$`. The catalog build turns these into typed brief sections. The Brief tab renders them as a calm, typeset page: the current stage is emphasised, math is typeset by JupyterLab's bundled math renderer, and links lead to the hints and the reference solution. 0059 is the first workout converted.

**Blocked by:** 03

**Status:** done

- [x] The catalog emits brief sections shaped `{kind, stage?, html}` with kind one of goal / predict / watch-out / why / prose.
- [x] The build rejects, with a message naming the script and line: an unknown marker, an unbalanced `$`, or a marker for a stage the workout doesn't have.
- [x] Scripts with no markers still produce today's brief exactly. The fixed equations table remains only as a fallback for them.
- [x] 0059's brief uses markers and `$…$` math, and the native `make test` / `make eval` still pass for it.
- [x] pytest covers marker parsing, math passthrough, rejection of malformed input and the no-marker fallback. e2e checks that the 0059 Brief tab shows the typed sections and typeset math in both themes.

## Comments

- A script opts in with a `# Goal:` line. Before that rule, any `$` or `# Predict:` turned a script structured, which would have changed or broken 0042–0047, 0067, 0068, 0071, 0077 and 0080. Today only 0059 opts in, and every other qualified brief is byte-identical to before.
- Only misspellings of a marker are rejected (`# Watchout:`, `# goal:`), with a "did you mean" hint. Ordinary prose labels like `Tiny example:` stay prose. `\$` is a literal dollar, and dollars inside backtick code are not math. Math must open and close on one line.
- Stage sections carry their stage as a class (`dojo-stage-<stage>`) because the notebook's HTML sanitizer drops data attributes. The Brief tab emphasises the stage being checked, marks passed ones, and quiets later ones. This only works for marked briefs; tickets 10–11 convert the rest.
- Hints and the reference solution are reachable from a "Stuck?" line under the brief. It drives the same controls under the answer, so assistance is still recorded on the rep.
- Marked scripts no longer get the legacy bare-identifier and equation-table heuristics; authors write `code` and `$math$` themselves.
