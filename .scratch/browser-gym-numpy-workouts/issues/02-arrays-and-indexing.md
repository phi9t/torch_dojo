# 02: Arrays and indexing: 0067, 0071, 0076, 0077

**What to build:** The remaining array and indexing exercises join the gym the way 0068 did: run-length random access, grid connected components, max-pool winning coordinates, and sparse COO products.

**Blocked by:** 01

**Status:** done

- [x] Each has a metadata entry (topic "arrays"), a typed brief, and 2–3 drills on its most common misconception (off-by-one run boundaries, 4- vs 8-connectivity, NCHW coordinate order, duplicate COO entries, and so on).
- [x] Each is in the qualification e2e table (reference passes at production, starter fails, one mutant fails for its reason) and in the smoke spec.
- [x] Native evals and question scripts behave as before; pytest passes.

## Comments

- Drills: 0067 (exclusive run ends, `searchsorted` side, `bool` as a count), 0071 (4 vs 8 neighbours, mark on push vs pop, recursion depth), 0076 (window-local vs original coordinates, divmod by `kw`, tie order), 0077 (fancy-index `+=` drops duplicates, rows vs columns, truncated fractional indices).
- Native verifier counts unchanged: 0067 6/6, 0071 6/6, 0076 7/7, 0077 6/6.
- 0077's candidate is a class, `SparseCOO`. The catalog's reference solution used to take only a top-level function named after the candidate, so it was empty for 0077, and 0061's reference was missing its helper. It now takes every top-level definition before the verifiers cell, with the imports they use.
