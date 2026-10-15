# Torch Dojo

Terms for the project's practice flow and interactive execution environment.

## Language

**Insula**:
An isolated environment that runs project tools against a mutable workspace.

**Mutable workspace**:
The writable view of the Torch Dojo repository shared between the host and an insula.

**Scenario**:
An ordered notebook practice run whose cells share the same working state, with setup and cleanup around the run. Reviewing an earlier result does not execute its cell again.

**Workout**:
One self-contained practice problem a learner opens, solves, and checks; the browser gym presents each question script as a workout. A lab is a longer workout checked in independent checkpoints instead of cumulative stages.
_Avoid_: lesson, exercise (in learner-facing text)

**Drill**:
A short predict-then-run practice inside a workout, such as predicting a shape before running an indexing expression; drills are never checked, but the learner's predictions are kept with the rep.

**Rep**:
One timed run at a workout that starts from the starter answer and keeps its own checks, hints used, and notes; returning to a workout means starting a new rep.
_Avoid_: attempt (in learner-facing text)

**Split**:
The active time from the start of a rep to its first passing check at a given stage; time while the page is hidden or idle does not count.

**Answer cell**:
The browser gym's form of a question script's answer section: the one notebook cell submitted when a learner checks a workout.
_Avoid_: solution cell

**Question script**:
A guided exercise that a learner temporarily edits, runs, and revises while solving a problem.

**Eval script**:
The companion to a question script that contains a reference implementation and verifiers.

**Reference version**:
A complete runnable implementation that teaches a particular approach to an exercise and states the scope in which it is correct.

**Reference progression**:
An ordered set of reference versions that explains and demonstrates improvements in an exercise's implementation.

**Answer section**:
The part of a question script reserved for the learner's implementation and its supporting definitions.

**Setup section**:
A part of a question script containing top-level definitions needed by the learner's answer during an evaluation run.

**Verifier**:
A check of one behavior or property of a learner's implementation.

**Eval case**:
A named, inspectable example with inputs, an expected outcome, and the behavior it exercises.

**Eval data**:
The concrete eval cases available to a learner for exploration and replay.

**Eval result**:
An inspectable record of what an implementation produced for an eval case and how it compared with the expected outcome.

**Baseline stage**:
The smallest complete version of an exercise's stated behavior.

**Follow-up stage**:
A changed constraint or edge case that builds on the baseline answer.

**Production stage**:
The exercise's explicit reliability and scale contract for a reusable implementation.

**Practice scene**:
A shared sensor observation used across related autonomy workouts, with explicitly named coordinate frames and observation support.

**Motif**:
A reusable problem-solving pattern that appears across workouts, with a cue for recognising when to use it.

**Recall card**:
A short prediction about a motif, followed by an explanation and a probe the learner can run. The learner rates how readily they recalled it; this is separate from a workout check.

**Review queue**:
Recall cards ordered by when they are due for practice, followed by cards not yet reviewed. A learner's recall rating determines when a card returns.
