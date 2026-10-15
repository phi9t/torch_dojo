# 03: Parsing: 0035, 0072, 0078, 0079

**What to build:** The text and parsing exercises join the gym: the byte-pair-style tokenizer, CSV records, the CSV consumer, and the two-operator calculator.

**Blocked by:** 01

**Status:** done

- [x] Each has a metadata entry (topic "parsing"), a typed brief, and 2–3 drills (merge order and ties, quoted fields and embedded newlines, what a consumer should receive for a corrupt row, operator precedence and associativity).
- [x] Each is in the qualification e2e table and the smoke spec.
- [x] Native evals and question scripts behave as before; pytest passes.

## Comments

- Drills: 0035 (Counter tie order vs the lexicographic tie-break, overlapping pairs, replaying merges in learned order), 0072 (`split(',')` and quoted fields, newlines inside quotes, doubled quotes), 0078 (dropping bad rows, `zip` hiding width errors, per-field conversion), 0079 (left fold vs precedence, the final term, non-ASCII digits).
- Native verifier counts unchanged: 0035 6/6, 0072 7/7, 0078 6/6, 0079 6/6. Review fix: 0079's drill reasoning no longer implies `isdigit` accepts underscores.
