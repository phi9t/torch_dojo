# 01: Learner-facing copy says "rep", not "attempt"

**What to build:** Everywhere a learner reads or clicks "attempt", the gym says "rep" instead (see the **Rep** entry in the glossary). This small, mechanical change goes first so every later ticket uses the new words. Stored data, export formats and internal names are unchanged.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] Buttons, headings, labels and aria-labels read New rep, Rep log, Resume rep N, Rep notes and Export rep. No learner-facing "attempt" remains.
- [x] Every e2e lookup that used an "attempt" label uses the new label, and the full e2e suite passes.
- [x] Records exported before the rename still import, and records exported after it still import into an older build.
