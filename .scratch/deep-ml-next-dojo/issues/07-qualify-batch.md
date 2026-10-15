# 07: Qualify the six-exercise batch

**What to build:** The six new dojo pairs form a coherent, discoverable batch whose exercise references, stage checks, and source attribution hold together on the integrated tree.

**Blocked by:** 01: Practice pairwise reward-model training; 02: Practice grouped RLOO advantages and policy loss; 03: Practice on-policy MDP trajectory sampling; 04: Practice a DDPM reverse sampling step; 05: Practice attention over paged KV cache; 06: Practice binary probability calibration fitting

**Status:** resolved

- [x] The dojo index lists exactly one question/eval pair for each reserved number 0089–0094 and explains how the batch extends existing exercises.
- [x] All six question scripts remain interactive top-level scripts with one tagged answer section, and all six eval scripts hold references and ordinary staged verifiers outside the shared judge.
- [x] Source links resolve, the prose and tests are original, and no Deep-ML prompt, solution, or hidden test is transplanted.
- [x] A review identifies at least one plausible wrong candidate per exercise and confirms its verifier fails for the intended reason, including numerical and boundary policies.
- [x] The all-reference eval suite, saved candidate stage checks, shared judge tests, compilation, and diff checks pass on the integrated tree.

## Comments

Implemented six authored question/eval pairs, 0089–0094, and indexed them in the README. Each question has one editable `# %% [answer]` section and top-level interactive cells; each eval keeps its reference and staged verifiers in the script. All six Deep-ML source URLs returned HTTP 200 on 2026-10-06.

Reference and mutation reviews covered one plausible wrong implementation per pair. Review fixes strengthened low-precision DDPM schedule arithmetic and Platt fitting with very large finite scores. Saved-file candidate checks exercised baseline, follow-up, and production stages for all six pairs. On the integrated tree, `make eval PYTHON=target/eval-venv/bin/python` passed all 94 references; `make test PYTHON=target/eval-venv/bin/python` passed 37 tests with 7 optional UI skips. Compilation and `git diff --check` passed.
