# 02: Explore paged attention from dense to streaming

Status: resolved
Blocked by: 01 — Explore softmax cases and progressive references
Parent: Progressive dojo references

## What to build

Apply the working exploration loop to paged attention. Let learners inspect the physical KV pages and logical page table, run the current answer, retain mismatches, and compare complete dense and streaming reference versions. Demonstrate why the memory improvement matters without changing the attention objective.

## Acceptance criteria

- [x] The question script retrieves named eval data into the kernel and provides runnable cells to inspect one page-table case, execute the current answer, retain its result, and replay a chosen failure after revision.
- [x] Shared fixture construction exposes ordinary query, key/value blocks, page tables, context lengths, and independent expected outcomes. Include repeated/reordered pages, partial pages, empty context, unused nonfinite cells, and invalid referenced data or indexing.
- [x] Complete dense and online-softmax streaming references coexist in separate interactive cells. Each explains page interpretation, intermediate quantities, supported contract, numerical behavior, computational work, temporary storage, and limitations.
- [x] The dense reference qualifies for the stages it supports; it does not claim the streaming production-storage contract. The strongest streaming reference remains the default direct-evaluation target and passes every required stage.
- [x] Comparisons cover forward parity, extreme score ranges, gradients, and input immutability using independent expectations. Preserved stage verifiers remain ordinary Python functions.
- [x] The streaming implementation visits a block at a time without gathering the full logical context. Verify this storage claim separately from output parity, and distinguish forward temporary storage from autograd-retained training state.
- [x] Opt-in memory/time exploration records workload, dtype, device, and method. Analytic storage estimates are labeled separately from observed peak memory; supported GPU measurements synchronize correctly. Add further reference versions only for demonstrated improvements.
- [x] Retrieval and replay retain structured tensor data without invoking reference code, reuse the shared mechanics from ticket 01, and do not introduce exercise-specific machinery into the judge.
- [x] A plausible wrong page-order or partial-tail implementation yields an inspectable failing case. Fresh replay, saved-answer isolation, and all reference versions' declared checks pass.

## Comments

Can run independently of ticket 03 after ticket 01 establishes the common exploration mechanics.

### Integration evidence

Integrated `0e10df7` on `codex/progressive-references-integration`. With
`TMPDIR` set to the project-local `target/t`, the two tests in
`tests/test_paged_exploration.py` pass, and direct execution of
`dojo/0093_paged_attention_eval.py` passes all eight checks, including
streaming storage, gradient parity, validation, and input immutability.

Final saved-candidate isolation and the combined Emacs/Jupyter workflow
qualification remain assigned to ticket 04. This ticket stays
`ready-for-agent` until that final acceptance evidence is recorded.

### Combined qualification

Ticket 04 completed the remaining paged attention acceptance checks. All advertised
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
