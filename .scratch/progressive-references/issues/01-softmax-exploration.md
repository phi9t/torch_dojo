# 01: Explore softmax cases and progressive references

Status: resolved
Blocked by: None (can start immediately)
Parent: Progressive dojo references

## What to build

Deliver the complete practice loop for softmax: retrieve concrete eval data into the kernel, inspect a named case, call the current answer directly, retain the comparison, fix the answer, and replay fresh inputs. Pair this with complete teaching and improved reference versions. Establish shared retrieval, replay, comparison, and measurement mechanics outside the exercise directory only as needed by this working slice.

## Acceptance criteria

- [x] The question script exposes a top-level eval_data variable and separate runnable cells demonstrating case inspection, manual answer execution, comparison, and selected-case replay. Keep one editable answer section, the existing cumulative checks, and no main function.
- [x] Eval cases have stable names, stage, purpose, named inputs, expected outcome or expected exception, and a comparison policy. Generated cases include reproducible seed/configuration details. Concrete data remains ordinary Python and tensor objects.
- [x] The eval script owns ordinary case factories in its reference-free verifier context. Verifiers and exposed data share fixture constructors. Retrieval executes neither reference definitions nor teaching probes nor the direct runner.
- [x] Individual and aggregate eval results are inspectable: case name, inputs used, actual output or exception, expected outcome, pass/fail, and useful numerical/shape/dtype diagnostics. Expected exceptions are distinguished from unexpected implementation errors.
- [x] Manual execution compares the observed output against the expectation for the exact input snapshot. Changed inputs require matching expectations; stale expected values are not silently reused.
- [x] Retrieval and automated replay build fresh mutable inputs and matching expectations, preserve intentional within-case aliases, and avoid reusing previous autograd graphs or module state.
- [x] A complete vanilla shifted-softmax reference and a complete reference with wider accumulation live in distinct interactive cells, with derivation, supported contract, complexity, numerical behavior, and tradeoffs. Improved core algorithms do not delegate to earlier versions.
- [x] The vanilla version respects the current large-logit baseline. Show a decisive low-precision long-reduction limitation and the improvement from wider accumulation; clearly label any deliberately unstable demonstration.
- [x] Qualify each reference version against its declared supported stage and compare forward/backward behavior on their common domain using independent expectations. Keep the strongest supported reference as the ordinary direct-evaluation target.
- [x] A small opt-in comparison experiment records workload, dtype, device, warmup, measurement method, and variability. Framework performance comparisons are labeled separately from authored solutions and learner restrictions. No speed threshold is a correctness gate.
- [x] Tests through learner-facing interfaces cover stage filtering, fresh replay after mutation/gradient use, expected exceptions, unexpected errors, useful mismatches, reference isolation, and saved-answer exclusion of exploration cells. Existing check calls keep their boolean semantics.
- [x] A deliberately wrong softmax answer produces a retained, reproducible failing case; revising that answer makes the same case pass. Focused pilot checks and shared-runner tests pass before the slice is considered complete.

## Comments

First ready ticket. This is a vertical slice establishing the pattern through an actual workout, rather than an isolated infrastructure change. The small common case structure must remain readable and must not introduce a function-name registry or generated unit-test format.

Implemented the complete softmax exploration loop and reference progression. Six learner-facing exploration tests and eleven existing judge tests pass. Shifted reference qualifies followup; wider accumulation qualifies production. Opt-in half-precision demonstration and timing comparison run successfully.
