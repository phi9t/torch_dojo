# 03: Explore MoE routes, expert batching, and gradients

Status: resolved
Blocked by: 01 — Explore softmax cases and progressive references
Parent: Progressive dojo references

## What to build

Apply the exploration loop to sparse MoE dispatch. Let learners inspect token/expert route rows, execute selected experts, identify lost or duplicated contributions, and inspect gradient behavior. Teach the progression from a directly traceable implementation to expert batching and reuse of the sorted route layout.

## Acceptance criteria

- [x] The question script exposes named cases and runnable cells for inspecting inputs, expert modules, expected route permutation, weighted token output, and an individual comparison result. Learners can revise one answer and replay a failed case.
- [x] Eval data includes stable expert ordering, duplicate token routes, uneven loads, unselected experts, empty batches, and malformed/nonfinite expert replies. Shared fixture construction supplies the same concrete data to verifiers and exploration.
- [x] Retrieval and replay construct fresh expert modules and tensors while preserving intended within-case aliasing. Prior parameter gradients, input mutations, module state, or tensor graphs do not contaminate a later replay.
- [x] Complete directly traceable and batched references coexist in separate interactive cells. A further version reuses the grouped route layout when it demonstrates a real reduction in repeated work. Each version states its contract, rationale, computational/memory costs, and tradeoffs.
- [x] Comparisons that change expert batch shapes use deterministic pointwise experts and explicitly state that scope. Stochastic or batch-coupled expert behavior is not silently treated as equivalent.
- [x] Versions claiming the production contract preserve stable permutation, repeated-route accumulation, sparse selected-expert execution, autograd, and the existing empty/malformed-output policies. Qualify every version against its declared supported stage.
- [x] Independent forward and backward checks cover inputs, mixture weights, and selected expert parameters. Ordinary Python verifiers retain deeper properties that cannot be expressed as a single input/output comparison.
- [x] An opt-in experiment records the workload and device and demonstrates the change in expert invocation work or route grouping, alongside measured timing where appropriate. It makes no distributed or GPU-kernel throughput claim.
- [x] Deliberately wrong duplicate-route overwrite or grouping implementations yield inspectable mismatches, and a gradient mistake is caught by the relevant verifier. Reference isolation and fresh replay checks pass.
- [x] Reuse the shared mechanics from ticket 01; introduce only the extensions required by real module-state and structured-output cases, with tests through the learner-facing interface.

## Comments

Can run independently of ticket 02 after ticket 01. Keep expert algorithms and route semantics local to this eval script; shared retrieval/replay mechanics stay outside the exercise directory.

## Integration evidence

Integrated `9d9efb6` by fast-forward into `codex/progressive-references-integration`.

- `TMPDIR="$PWD/target/t" PYTHONPATH=. target/eval-venv/bin/python -m unittest discover -s tests -p test_sparse_dispatch_exploration.py -v`: 3 tests passed. The deliberate detached candidate failed both gradient verifiers as expected.
- `TMPDIR="$PWD/target/t" PYTHONPATH=. target/eval-venv/bin/python dojo/0101_sparse_dispatch_eval.py`: 12/12 default reference checks passed.
- Initial pytest invocation could not run because this environment has no pytest; qualification used the repository's unittest runner.

Ticket remains pending ticket 04's combined saved-candidate and Emacs/Jupyter qualification.

### Combined qualification

Ticket 04 completed the remaining MoE dispatch acceptance checks. All advertised
saved-candidate stages (`baseline`, `followup`, `production`) pass through
`evaluate(..., candidate=True)` using project-local copies containing raising
poison in excluded question teaching/exploration cells and excluded eval
reference/exploration cells. All declared reference-version stages pass through
`check`: dense attention at followup, streaming attention at production, and
all three dispatch versions at production. Existing independent numerical,
gradient, and resource verifiers establish their respective contracts.

Emacs 29.3 with emacs-jupyter executed 12 sequential requests against an owned
Jupyter/ipykernel session for all three pilots. It retained failing case data
and results, redefined the current answer, and passed fresh selected replays
without losing the earlier failure. The paged demonstration diagnosed an
untrimmed nonfinite tail; the dispatch demonstration diagnosed overwritten
duplicate contributions and then ran backward on the fresh live output.

Combined repository gate: 48 tests ran successfully, seven optional environment checks
skipped; all 102 reference scripts pass. Six pilot scripts parse and diff
checks pass. Evidence artifacts are under the ignored project-local
`target/progressive-references/verification/qualification/`; the workflow and
limitations are documented in `docs/practice/interactive-eval.md`.
