# 08: Diagnose a broken MLP

**What to build:** A learner investigates a planted MLP defect, sees evidence of the failure, and demonstrates that their correction resolves it.

**Blocked by:** 07: Complete the MLP experiment

**Status:** resolved

- [x] The lab offers a reproducible broken run with visible symptom, relevant numerical diagnostics, and a clear repair task.
- [x] The learner can compare a broken run with a corrected run using plots and checkpoint results.
- [x] An isolated check rejects the planted defect and accepts the corrected implementation; the receipt preserves the checked source and any reference substitutions.

## Resolution

- Added the MLP `diagnose` checkpoint around the planted `omitted_batch_mean` defect, including finite-difference diagnostics, broken-vs-corrected training comparison data, checkpoint comparison evidence, and a constrained diagnosis report contract.
- Exposed the repair task through the MLP notebook UI and source-bound browser receipts, including explicit source substitution coverage for `loss_and_grads`.
- Verified native checks, extension checks, browser e2e on `DOJO_TEST_PORT=8108`, the `browser_gym` target, and the root Python suite.
