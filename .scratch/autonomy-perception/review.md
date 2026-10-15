# Autonomy perception integration review

Status: resolved

## Initial independent reports

- Standards: zero findings.
- Spec: two P2 findings, independently reproducing pre-review QA observations.

### P2: half-open membership and rounded upper cell indices

An interior float32 point with x=-finfo.tiny in a box ending at zero was
physically inside, but subtraction against origin=-1 rounded its computed
cell x to1. Pooling split two actual same-voxel rows; decoration used incorrect
centroids/centers; the copied capstone helper emitted an out-of-range cell.

Correction: decide physical membership first, validate finite/int64-representable
computed coordinates, then cap the already-included integer coordinates at
grid_shape-1. The rule is shared by all decoration/grouping reference versions
and the capstone's editable setup. Exact/outside boundary rows remain excluded.
Literal float32/64 fixtures establish one voxel, counts/inverse IDs, BEV mean
and centroid offsets±0.25. Decimal represented-input scalar arithmetic prevents
the independent oracle from reproducing the same cancellation bug. This oracle
models represented-input scalar arithmetic for the fixtures; it does not claim
universal agreement with working-dtype floor at every near-internal boundary.
The independent literal outer-boundary expectations establish this regression.

### P2: adapter accepts singular feature-coordinate affine

The adapter constructor accepted finite metadata with a valid homogeneous
bottom row but singular2x2 linear block, while the feature sampler rejects it.

Correction: require nonzero linear-block determinant before constructor state
changes. The shared initializer covers every complete adapter version; the
question states invertibility explicitly. A named dependent-row affine fixture
expects constructor ValueError; float32 and float64 are checked.

## Red/green and qualification evidence

- Decoration regression:7/8 →8/8 default checks.
- Pooling regression:5/7 →7/7 default checks.
- Capstone regression:6/8 →8/8 default checks, including the prior unsafe
  out-of-range scatter error.
- Adapter regression:6/7 →7/7 default checks.
- All12 affected advertised reference/stage scopes passed. The inference-only
  adapter remains baseline-forward scoped, and its metadata constructor
  rejection was separately replayed.
- All12 filled actual-question saved-candidate/stage combinations passed with
  excluded teaching/reference poison and an executed setup marker. The
  capstone answer uses its actual question setup, not an answer-side helper
  copy, so its saved setup correction is qualified directly.
- Actual Emacs29.3/emacs-jupyter/ipykernel ran14 sequential stateful cells on
  the owned kernel. Four pre-fix results/errors remained inspectable while
  new definitions passed fresh float32/64 boundary and singular-affine cases.

Reproducible artifacts under target/autonomy-perception/verification/review-fix:
run.py, report.json, qualification.log, steps.json, connect.el, emacs.log and
kernel-report.json. Temporary state remains project-local; no /tmp or remote
push. The owned kernel remained ready through qualification and was then shut down by root.

## Independent rechecks

### Standards

Clean. The fixes introduce no documented-standard violations or actionable
smells. Boundary corrections stay in independent scripts' geometry helpers;
literal fixtures and scalar oracles stay inside verifier sections. Teaching
probes remain untagged, and adapter validation uses the shared initializer.

Remaining Standards findings: 0.

### Spec

Both P2 findings resolved in c5d9309. The reviewer independently verified all
three decoration versions' centroid offsets of +.25 and -.25 in float32/64;
all three capstone versions' one cell, count 2 and BEV [3,0,0]; the actual
question setup's cell [0,0,0,0]; and all adapter constructors' singular-affine
rejection. Question contracts and dedicated verifiers cover both changes.
Literal boundary expectations establish the regression independently, and
the Decimal oracle's arithmetic scope is documented.

Remaining Spec findings: 0.

## Final integration gate

Root merged c5d9309 and independently ran both required gates to completion:
all 113 reference scripts passed (exit 0), and 49 repository tests passed with
seven optional skips (exit 0). Ticket13's final acceptance is closed. Logs are
retained under target/autonomy-perception/final-eval.log and final-test.log.
The owned qualification kernel exited successfully. All 15 clean worktrees
created for this implementation were removed; unrelated worktrees remain.
