# 02: Preserve checked source

**What to build:** After checking an answer, a learner can see precisely what source was checked and whether later edits have made that result stale.

**Blocked by:** 01: Edit and check `0059` in the browser

**Status:** resolved

- [x] Each check creates an append-only receipt containing the exact source snapshot and hash, result, timestamp, and exercise, checker, runtime, package, and fixture identities.
- [x] Editing a solution cell after a passing check marks that pass as stale without changing the receipt or the saved source.
- [x] Rechecking creates a new receipt, and the result panel identifies the source hash for each check.
- [x] An attempt retains its notebook and notes alongside its receipts after a reload.

## Comments

Resolved with local attempt receipts that preserve checked source/hash, checker/runtime/package/fixture identity, persisted notebook/notes, and stale-pass labeling after solution edits. Verified with focused unit tests, browser acceptance, and `make -C browser_gym test`.
