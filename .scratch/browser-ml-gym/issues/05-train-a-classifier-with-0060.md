# 05: Train a classifier with `0060`

**What to build:** A learner moves from stable loss to a classifier step, sees training behavior, and receives an isolated check of their implementation.

**Blocked by:** 03: Hints, new attempts, and export; 04: Explore `0059` visually

**Status:** resolved

- [ ] The `0060` notebook presents its canonical contract, editable answer, hints, and independent check with attempt receipts.
- [ ] A bounded training run shows a loss curve and a tensor shape or gradient probe; changing a control does not launch training automatically.
- [ ] The notebook links the completed `0059` concept to the classifier step without requiring a prior pass to open it.
- [ ] Reference passes; starter and a plausible wrong classifier step fail for the intended reasons.

## Resolution

Implemented in `browser_gym` as a generated `0060` notebook/checker derived
from the canonical `dojo/0060_classifier_step.py` and
`dojo/0060_classifier_step_eval.py` pair. The browser lesson includes
progressive hints, reference reveal data, source-bound receipts through the
existing attempt store, tensor shape and gradient probes, and an explicit
button-triggered bounded training loss curve. `0060` opens directly with
`/lab/?exercise=0060`; no prior `0059` pass is required.

Validation covered reference pass, starter failure, and a plausible wrong
valid-count classifier step failing
`verify_production_weight_unequal_microbatches`.
