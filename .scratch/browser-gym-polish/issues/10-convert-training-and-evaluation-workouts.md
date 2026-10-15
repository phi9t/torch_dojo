# 10: Convert briefs and drills: training and evaluation workouts

**What to build:** The browser workouts on training and evaluation (0060, 0061, 0069, 0073, 0074) get structured briefs with markers and `$…$` math, and two or three drills each that target the workout's most common misconception, so they read like 0059.

**Blocked by:** 07

**Status:** done

- [x] Each workout's brief uses Goal / Predict / Watch out / Why where they fit, with no marker forced where it adds nothing.
- [x] Each drill's probe runs in the browser kernel in well under a second, and its reasoning explains why, not only what.
- [x] The native `make test` / `make eval` and the catalog build pass for every converted workout.
- [x] e2e smoke: each converted workout opens with typed brief sections and at least one drill that runs.

## Comments

- Each workout has a Goal, stage-scoped Predict / Watch out (and Why for 0073) where they fit, `$…$` / `$$…$$` math for the Brier, ECE and AP formulas, and three drills: 0060 (bias reduction axis, in-place update of the caller's weights, weighting microbatch means), 0061 (buffered `+=` vs `np.add.at`, `bincount` without `minlength`, stale labels), 0069 (edge bin, Brier sum vs mean, confidence vs true-class probability), 0073 (paired vs independent resampling, linear-interpolated percentiles, trips vs frames), 0074 (GT ids across images, recall denominator, precision envelope).
- Native eval verifier counts are unchanged: 0060 6/6, 0061 8/8, 0069 6/6, 0073 6/6, 0074 6/6. Every probe runs in about 0.1 s natively and its reasoning quotes the real output.
- Review fixes: 0069's drill reasoning corrected (which rule needs the clamp at confidence 1; Brier scores every class), and Watch out sections in 0069 and 0073 end before the contract text that follows.
- The e2e smoke (`converted.spec.ts`, committed with ticket 11) opens every converted workout, checks the typed brief, and runs and reveals its first drill.
