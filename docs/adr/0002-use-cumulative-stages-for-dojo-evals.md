# Use cumulative stages for dojo evals

## Decision

A question script keeps one learner-edited answer section. The learner revises it through baseline, follow-up, and production stages. The paired eval script uses ordinary Python verifier functions named `verify_baseline_*`, `verify_followup_*`, and `verify_production_*`; the judge selects all verifiers up to the requested stage. Untagged `verify_*` functions retain their existing behavior during migration. Direct reference evaluation runs all checks, while saved candidate evaluation starts at baseline unless a stage is requested.

## Context

The dojo is used as top-level `# %%` scripts in Emacs and Jupyter. Separate answer modules or generated test-case registries would make it harder to edit one working solution interactively. A single final-only check would obscure the learner's progress and make a production constraint fail before the baseline is understood.

## Consequences

Each exercise author must state the new contract before adding a stage verifier. Later stages cannot silently weaken earlier behavior. The judge needs only stage selection; reference implementations and all exercise-specific tests stay in their eval scripts. A production label describes the exercise's explicit reusable-kernel contract, not unrestricted deployment readiness.
