# 05: Timed reps with splits and stamps

**What to build:** Each rep is timed. The rail shows a quiet active-time timer. When a check first passes a stage, its split (see the glossary) is recorded, and that stage's seal is stamped in cinnabar, alongside the best time and a faster/slower delta. When the final stage passes, the timer stops and a one-line summary appears. The Reps tab lists past reps with their splits, hints used and notes, and marks the best rep.

**Blocked by:** 01, 03

**Status:** done

- [x] Active time counts only while the page is visible and the learner is not idle (no input for 2 minutes), and stops after the final stage passes.
- [x] The rep record gains `activeSeconds` and `predictions` (predictions are filled in by the drills ticket). Each check receipt gains `activeSecondsAtCheck`.
- [x] A rep's split at a stage is the `activeSecondsAtCheck` of its first passing receipt at that stage. The best time is the minimum split per stage across reps.
- [x] Records saved before this change load with `activeSeconds = 0` and unknown splits, shown as "—" and never as 0:00. They never become a best time.
- [x] Export then import is lossless for the new fields, and older exports still import.
- [x] The stamp animation respects prefers-reduced-motion. There is no sound, confetti, XP or streak.
- [x] vitest covers the store's new fields, the split and best-time rules, and migration. e2e uses a fake clock to cover the timer pausing on hidden/idle, splits per stage, the stamp, the summary and the Reps tab.

## Comments

- The timer adds one-second ticks only while the page is visible, there was input within the last 2 minutes, and the rep is unfinished. A tick adds at most 2 seconds, so a sleeping laptop doesn't count. When the rep finishes, the timer settles on the finish time from the finishing check, and that time is saved exactly.
- Best is the minimum known split per stage across the workout's other reps (this ticket's wording), not only completed reps (the spec's wording). Unknown splits (`null`) never become a best.
- Receipts saved without a `stage` get one inferred from their `verify_<stage>` names, so seals, splits and squares agree on old reps.
- The store also gained `savePrediction`, which ticket 07's drill cards use.
- The summary's Next up link points at the learning path's next workout for now. Ticket 08 replaces it with the home page's Next up rule.
