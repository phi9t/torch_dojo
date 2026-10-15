# 15: Recover from stopped work

**What to build:** A learner can stop a runaway check or exploration and continue working without losing prior results.

**Blocked by:** 02: Preserve checked source

**Status:** resolved

- [x] A check deadline and Stop action terminate a runaway checker and produce a timeout or stopped receipt for the submitted source.
- [x] A subsequent check starts in a fresh worker and succeeds when the answer is valid; previous receipts remain intact.
- [x] Exploration offers an interrupt when available and a clear kernel restart path when it is not.
- [x] Restart explains that live kernel state is discarded while saved attempts, notebooks, notes, and receipts remain available.
