# 09: Connect calibration with `0069`

**What to build:** A learner examines confidence and reliability for model predictions and solves the calibration metrics exercise with the same browser workflow.

**Blocked by:** 05: Train a classifier with `0060`; 07: Complete the MLP experiment

**Status:** resolved

- [x] The `0069` notebook exposes the canonical metric contract, answer cells, hints, check results, and attempt receipts.
- [x] A confidence or bin-count control updates a reliability visualization and a tensor or behavior probe.
- [x] The learning path connects `0059`, `0060`, MLP, and `0069` through visible navigation and shared concepts.
- [x] Reference passes; starter and a plausible incorrect metric implementation fail for the intended reasons.

## Resolution

Implemented the generated `0069` Browser ML Gym lesson, catalog metadata, checks, receipts, and calibration probe UI. The notebook now exposes the canonical `calibration_metrics` contract with answer and hint cells, reliability and confidence visualizations, bin-count controls that do not mutate submitted source, and visible path navigation across `0059`, `0060`, MLP, and `0069`.

Verification:

- `python -m pytest browser_gym/tests -q`
- `npm --prefix browser_gym/extension run typecheck`
- `make -C browser_gym build`
- `DOJO_TEST_PORT=8109 npm --prefix browser_gym/extension run test:e2e -- qualification.spec.ts -g 0069`
- `DOJO_TEST_PORT=8109 make -C browser_gym test`
- `python -m unittest discover -s tests -v`
- `PYTHONPATH=. python dojo/0069_calibration_metrics_eval.py`
