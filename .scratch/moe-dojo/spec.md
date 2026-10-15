# Portable MoE dojo workouts

Status: resolved

## Objective

Turn the eight-workout learning path in `docs/research/moe-implementations-primary-survey.md` into original, runnable dojo question/eval pairs numbered 0095–0102. Keep the existing Emacs/Jupyter script workflow: top-level `# %%` cells, one `# %% [answer]`, staged `check(...)` calls, and a neighboring eval script with a standalone reference plus `# %% [verifiers]`. Judge machinery stays outside `dojo/`.

## Scope

1. Router variants: selected softmax, full-softmax truncation, selected renormalization, sigmoid affinity with selection-only bias.
2. Shared branch: absent, ungated, gated.
3. Route ledger: requested/kept counts by expert/rank and imbalance.
4. Balancing: differentiable soft scores, detached hard counts, local/global scope.
5. Post-batch bias feedback.
6. Capacity admission: per-expert vs shared-rank capacity.
7. Expert-contiguous dispatch and weighted combine.
8. A tiny end-to-end MoE training update and diagnostics.

All exercise contracts must explicitly specify shapes, invalid/empty behavior, tie policy where relevant, and source/evidence limits. The verifiers should test a plausible wrong implementation at each stage; production means a reusable local kernel, not distributed or GPU throughput. Reference evals must run with available repo dependencies and candidate baseline/stage path must load the answer cell. Update README index/count and run repo gates. Do not edit unrelated untracked files. No `/tmp`.

## Out of scope

Distributed collectives, CUDA kernels, exact replication of private model training loops, hierarchical routing, RL route replay, V4 hash routing.

## Acceptance

All 8 references pass baseline/followup/production; all 8 candidate stubs load and fail for an answer-related reason; `make eval` and `make test` pass; README links all pairs; local `main` contains the result.

## Qualification

All eight pairs passed their staged references. Candidate stubs loaded through the saved-file path and failed on their editable answers. The full 102-reference eval and repository tests passed after review fixes. Standards review found no documented violation; spec review's capacity, dispatch, and training diagnostics findings were fixed and rechecked.
