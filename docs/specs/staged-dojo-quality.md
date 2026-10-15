# Staged dojo quality

## Problem statement

The 81 dojo question/eval pairs offer a working edit-and-check loop, but many question scripts give only a toy probe and a single implementation step. Their eval scripts typically have a few checks and do not make follow-up constraints visible as a progression. A learner can pass without practicing the shape, numerical, error, memory, and integration choices that make the algorithm useful beyond a toy input.

## Solution

Each exercise keeps one editable answer section and one neighboring eval script. A learner first solves a baseline contract, then revises the same answer for follow-up constraints, then meets a problem-specific production contract. Each stage has a named, cumulative set of verifiers. The question script explains the contract, gives a prediction or derivation prompt, and provides a runnable check at each stage. The eval script holds the reference and native Python verifiers; the judge only selects stages and reports failures.

Production means a robust reusable algorithmic kernel, not a service deployment. Relevant requirements include input validation, numerical stability, dtype/device/shape semantics, reproducibility, bounded memory, performance, and decisive testing. A requirement belongs in an exercise only when it teaches a real issue for that algorithm. Sequential algorithms may use loops; GPU tests run only when the environment supports them.

## User stories

1. As a learner, I can pass the baseline before handling harder cases, so I can build confidence from a complete small solution.
2. As a learner, I can run a named follow-up check against my current kernel definition, so I can revise without restarting the notebook.
3. As a learner, I can run a production check that includes earlier stages, so passing means I did not regress the baseline.
4. As a learner, I can inspect explicit shapes, boundary rules, invalid-input policy, and worked probes before coding.
5. As a learner, I can edit top-level helper definitions and answer code in Emacs/Jupyter without a `main()` function.
6. As a learner, I can run saved-file evaluation with a stage flag, so results match the interactive check without executing teaching probes.
7. As a learner, I can distinguish source material from authored extensions, so I do not mistake practice constraints for recovered questions.
8. As an exercise author, I can write ordinary `verify_*` functions with stage names and shared helpers, so the eval remains a readable script.
9. As an exercise author, I can run all stages against the reference as the quality gate.
10. As an existing learner, I can use untouched exercise pairs exactly as before while migration proceeds.

## Implementation decisions

- Stage names are `baseline`, `followup`, and `production`, in that order.
- A staged verifier name begins `verify_baseline_`, `verify_followup_`, or `verify_production_`. A requested stage runs that stage and every preceding stage.
- Existing `verify_*` functions retain their current behavior until migrated.
- Interactive checks select a stage explicitly. Direct reference evaluation runs every check; `--candidate` defaults to baseline for staged exercises and accepts `--stage`.
- The answer section remains a single tagged cell. The learner improves its implementation in place; teaching and check cells stay ordinary top-level script cells.
- Pilot exercise families are attention, voxel pooling, and CSV parsing. They test model state/masks, sparse geometry/memory, and streaming grammar/errors.
- The migration of the remaining pairs is organized into small independently verifiable batches, with each pair retaining a passing reference.

## Testing decisions

- Test stage selection at the judge's public `check`/`evaluate` interface, including cumulative order, legacy pairs, invalid stages, reference defaults, and candidate defaults.
- Test each pilot's reference against all stages. Verifiers should check behavior and independent invariants rather than the implementation text.
- Use focused examples and Hypothesis where generated cases reveal boundary, permutation, or chunk-splitting mistakes.
- Keep the all-reference suite green after each migration batch.

## Out of scope

- A deployable service, distributed execution, and general benchmarking infrastructure.
- Forcing every exercise to use the same production checklist or an arbitrary performance threshold.
- Running GPU-dependent checks on machines without a supported GPU.

## Further notes

The existing ADR preserving interactive scripts and tagged answer/setup/verifier cells remains in force. The fieldbook's source and evidence limits remain separate from the authored dojo contracts.
