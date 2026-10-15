# Progressive references review

Baseline: `cdbaec737e9a1f9ea63ff8f632fee6c334bbfec7`.
Initial reviewed integration: `d7b9468f9432f8beb5479b7312f821c4302b143c`.
Diff: `git diff cdbaec7...HEAD`.

## Standards

No documented-standard violations. The scripts retain top-level cells and a
single answer section; infrastructure remains outside `dojo`, and stages
remain cumulative.

One baseline judgement call: possible duplicated verification code in
`0101_sparse_dispatch_eval.py`. The case factory repeated the token/slot affine
oracle used for backward verification. Reuse `_pointwise_oracle` under
`torch.no_grad`, keeping the empty-input branch explicit. Complete reference
algorithms are intentionally independent and should stay that way.

## Spec

One P2 finding: Python integer scalars were compared through `math.isclose`,
which can convert distinct large integers to the same float. Confirmed:
`compare_values(2**60 + 1, 2**60)` returned no mismatch, and changing a case's
count by one incorrectly passed comparison with the original expectation.

This violates the requirement: “If a learner changes inputs, comparisons must
use expectations for those changed inputs … do not silently compare with an
expectation computed for old data.” Compare integral scalars exactly and add
public-interface regressions for output comparison and stale-input detection.

The remaining implementation requirements passed review: complete versions,
independent oracles, supported stages, storage and gradient checks, reference
isolation, fresh replay, and qualified Emacs/Jupyter and saved-candidate loops.

## Resolution

Both findings were resolved in `70204f9`. The scalar policy compares Python
and NumPy integer/boolean scalars exactly and preserves floating tolerances.
The case factory now shares the independent pointwise MoE oracle under
`torch.no_grad`, with an explicit empty-input branch.

Affected verification: 49 repository tests ran successfully with seven
optional environment skips; all three MoE versions passed 12 production
verifiers and ten replay cases each. The Spec reviewer independently reran
all seven eval-data tests. Both reviewers confirmed the fixes and reported
no remaining actionable findings.

Initial findings: Standards 0 documented violations and 1 maintainability
smell; Spec 1 correctness finding (P2).

Final findings: Standards 0; Spec 0.
