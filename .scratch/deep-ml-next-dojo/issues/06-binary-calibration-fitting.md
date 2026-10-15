# 06: Practice binary probability calibration fitting

**What to build:** A learner can fit a Platt or isotonic calibrator on one set of scores and predict probabilities for separate query scores. Reserve exercise 0094. The approved parent spec governs the common question/eval script contract.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The question/eval pair uses top-level cells, one editable answer section, a probability-prediction probe, and cumulative baseline, follow-up, and production verifiers.
- [x] Baseline fits Platt calibration with stable logistic loss and the specified positive L2 term on slope and intercept.
- [x] Follow-up fits isotonic calibration by pooling adjacent violations after combining tied scores, interpolates between fitted scores, and clamps outside their range.
- [x] Production handles constant scores and one-class labels, validates finite binary data, leaves inputs unchanged, and predicts without query labels.
- [x] The reference passes direct evaluation; saved candidate stage selection works; raw-score passthrough and a calibrator that ignores score ties fail checks. Add the exercise to the dojo index with a source link and original wording.

## Resolution

Merged `0094_binary_calibration.py` and its eval with the #94 index row. Direct reference evaluation passed 5/5 verifiers. Saved candidate baseline selection ran exactly 2 checks and rejected the blank answer as expected. The known Platt optimum rejects raw-score passthrough; the isotonic tie case checks weighted pooling and interpolation.
