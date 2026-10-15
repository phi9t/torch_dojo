# Progressive dojo references

Status: resolved

## Problem Statement

Learners can revise their answers through baseline, follow-up, and production stages, but the eval scripts generally expose only one final reference. That reference often combines the mathematical algorithm, numerical choices, validation, and efficiency decisions. Reading it does not show how to develop a simple implementation into a reliable and efficient one.

The user wants a vanilla implementation that is particularly good for teaching, followed by gradually deeper improvements in numerical stability, performance, memory, and efficiency. Each reference version must be a complete runnable implementation in its own interactive cell, so learners can execute and compare versions directly in Emacs/Jupyter.

Learners also need access to concrete eval inputs and outcomes in their live kernel. A pass/fail stream currently leaves the data inside verifier functions, making it awkward to fetch one example, run the current answer, inspect the result, and decide what to fix.

## Solution

Give each pilot eval script a reference progression. Begin with the simplest useful implementation of the stated mathematics. Add versions that each introduce one meaningful improvement, explain its mechanism, and demonstrate its benefit and tradeoffs. Keep the reference progression beside the ordinary verifiers and the learner's separate question script.

Expose named eval cases through an ordinary Python case factory in the eval script. The question script retrieves them into a top-level eval_data variable. Learners can inspect a case's inputs and expected outcome, call their own implementation directly, compare the output, and retain individual or aggregate eval results in the kernel. Shared retrieval, replay, and comparison mechanics live outside the exercise directory.

The three proposed pilots are softmax, paged attention, and sparse MoE dispatch. They exercise numerical reductions, context-sized temporary storage, sparse expert work, and gradients. Qualify the pilot pattern before extending it across the bank.

## User Stories

1. As a learner, I want a readable vanilla reference, so that I can connect equations to operations and trace a small example by hand.
2. As a learner, I want each reference version to be complete and runnable, so that I can execute it without reconstructing patches or running earlier algorithm definitions.
3. As a learner, I want a separate cell for each version, so that I can explore the progression in my existing Emacs/Jupyter session.
4. As a learner, I want each version's supported inputs and limitations stated, so that I know what its successful checks establish.
5. As a learner, I want each transition to explain the problem it solves, so that an optimization has a reason beyond being shorter or more complicated.
6. As a learner, I want a counterexample to the earlier limitation, so that I can understand the numerical or memory issue concretely.
7. As a learner, I want parity checks between versions on their common domain, so that I can distinguish an implementation improvement from a changed objective.
8. As a learner, I want gradient checks where relevant, so that a fast forward implementation does not silently break training.
9. As a learner, I want measurements with shapes, dtype, device, and method recorded, so that I can assess when an efficiency improvement is useful.
10. As a learner, I want to run explorations explicitly, so that ordinary answer evaluation stays quick.
11. As a learner, I want to keep revising one answer section, so that exploring references does not fragment my working solution.
12. As a learner, I want to see what an optimization costs in readability or memory, so that I can choose an implementation for a particular workload.
13. As an author, I want to test reference versions through the existing checking seam, so that learner and reference evaluation exercise the same contracts.
14. As an author, I want unsupported early-version constraints made explicit, so that a teaching version is not advertised as production-ready.
15. As an author, I want shared measurement mechanics outside the exercise directory, so that the eval scripts focus on the algorithms and comparisons.
16. As a maintainer, I want existing saved-answer and live-kernel behavior preserved, so that the reference progression does not change how users practice.
17. As a learner, I want the eval data in a top-level variable, so that I can browse and manipulate concrete examples in Emacs/Jupyter.
18. As a learner, I want a consistent case structure, so that I can find the inputs, expected outcome, stage, and purpose without reading a verifier's implementation.
19. As a learner, I want to call my answer directly on one case, so that I can inspect intermediate tensors and debug with ordinary Python.
20. As a learner, I want an eval result containing actual and expected values, so that I can understand a mismatch and rerun it after a fix.
21. As a learner, I want to retrieve fresh case data, so that input mutation, gradients, or module state from an earlier attempt do not corrupt a replay.
22. As a learner, I want to collect results for several cases and inspect only failures, so that I can focus on the behavior that still needs work.
23. As a learner, I want deterministic generated samples with recorded generation details, so that I can explore more shapes and edge cases while reproducing a failure.
24. As an author, I want verifiers and the exposed eval data to share case construction, so that users explore the same inputs used by the checks.

## Implementation Decisions

- Preserve the interactive question/eval pairing and cumulative learner stages recorded in the existing ADRs. Question scripts retain one editable answer section and optional setup sections, without a main function.
- Reference versions live in separate top-level cells in the eval script, before its verifier section. Each has a distinct descriptive name and can coexist in the kernel with the other versions.
- Reference versions contain complete algorithms. They may share ordinary imports and clearly documented contract helpers. An improved version must not delegate its core algorithm to a preceding version or require mutable state created by an earlier teaching probe.
- A reference progression and a learner stage are different concepts. The number of reference versions is problem-specific. Qualify each reference version against the highest existing stage it actually supports.
- The strongest supported reference remains the ordinary direct-evaluation target. Saved-candidate evaluation continues to select the learner's answer; reference definitions and exploration cells must remain excluded from live learner checks.
- Explanations accompany each version: supported contract, derivation or intermediate values, change from the preceding version, complexity and temporary storage, numerical behavior, and tradeoffs.
- Vanilla means the simplest correct implementation on its declared domain. A deliberately unstable or wrong implementation may be shown as a clearly labeled failure demonstration, not as a reference claiming to satisfy a stage that requires stability.
- Softmax starts with explicit maximum subtraction and reductions because the current baseline already requires large-logit stability. A later version investigates wider accumulation for low precision or long reductions. The framework's fused implementation may be measured as a separately labeled comparison; learner restrictions continue to apply to authored reference solutions.
- Paged attention starts with a dense logical-sequence implementation that makes page ordering and the attention equation visible. A later version visits one block at a time with online maximum, mass, and weighted numerator. Further numerical or recomputation versions belong only when their benefit is demonstrated. Distinguish forward temporary storage from autograd-retained training state.
- Sparse MoE dispatch starts with a directly traceable routing/reduction implementation, followed by expert batching and reuse of the sorted route layout. Comparisons that change expert invocation batch shapes use deterministic pointwise experts. Batch-coupled or stochastic experts can have different semantics and must not be silently claimed equivalent.
- Preserve stable route ordering, duplicate-route accumulation, gradients, and the current empty and malformed-output policies in reference versions that claim the production contract.
- Question scripts add prompts to predict limitations and investigate the next improvement. They point to the reference progression without inserting reference answers into the answer section.
- Use the existing check/evaluate interface as the primary correctness seam. Complete reference versions and learner answers are real adapters at that seam. Eval data is an inspectable collection of concrete examples; verifier execution remains ordinary Python functions. Avoid a function-name registry, self-registration machinery, or generated unit-test structure.
- Keep problem-specific inputs, objectives, tolerances, and reference comparisons in the eval script. Reusable timing, device synchronization, and memory-measurement mechanics belong outside the exercise directory if shared mechanics are needed.
- Explorations and performance measurements are explicit opt-in work. Ordinary reference eval remains a correctness check. Separate setup, warmup, repeated measurements, and device synchronization; report workload, dtype, device, and variability.
- Claim performance improvements only for measured workloads. Do not require an arbitrary speed threshold or assume every vectorized version wins on tiny inputs. Label analytic storage estimates as estimates and distinguish them from observed peak memory.
- Add a small common structure for eval cases: stable name, stage, explanatory purpose, named inputs, expected outcome or expected exception, and comparison policy. Generation details include a seed and relevant configuration when a case is generated. Do not store a reference implementation in the case.
- Concrete tensors, arrays, scalars, and nested Python structures remain accessible as ordinary objects. Preserve shapes, dtype, device, and autograd requirements; avoid converting all inputs into a display-only serialized format.
- Eval scripts own the case factories and exercise-specific comparison policies. Load these from the reference-free verifier context already used for learner checks. Retrieving data must not execute reference definitions, the direct runner, or teaching probes.
- Question scripts include an ordinary top-level exploration cell that retrieves eval data into a variable, selects one named case, runs the learner's current answer, and keeps the comparison result. These exploration cells are excluded from saved-answer loading and do not need a main function.
- Case retrieval and automated replay construct fresh inputs and their matching expected outcomes. A replay must not reuse a tensor graph or mutable module state from an earlier attempt. Preserve intentional alias relationships within one case where they are part of the contract.
- If a learner changes inputs, comparisons must use expectations for those changed inputs. Recompute with an independent oracle or rebuild the case from its generation parameters; do not silently compare with an expectation computed for old data.
- Eval results retain the case name, inputs used, actual outcome or exception, expected outcome, pass/fail, and diagnostic details. Numerical diagnostics include useful shape/dtype discrepancies and error information according to the exercise's stated comparison policy. Use tolerances rather than exact equality where required.
- Support both manual execution and an optional convenience replay across selected cases. Results are returned as inspectable Python data. Existing check calls retain their boolean behavior and cumulative stage semantics.
- Expected-exception cases show the intended failure separately from an unexpected implementation error. A correct rejection is a successful result, not an unexplained red traceback.
- Deterministic examples and selected generated samples are inspectable. Property verifiers, gradients, resource assertions, and stateful checks remain ordinary Python where a single input/output record is insufficient. Do not claim a finite eval-data collection represents every generated property trial.
- Shared case construction is the source of fixture data for both verifiers and exploration. Exercise-specific verification remains independent of the learner answer and avoids using reference versions as the sole correctness oracle.
- Start with the same three pilots and expand the schema only when their real differences require it. Larger allocations, device transfers, and benchmarks stay explicit opt-in actions.

## Testing Decisions

- Reuse ordinary stage verifiers and the existing public checking interface. Keep independent mathematical or framework oracles in verifiers rather than calling another reference version as the only proof of correctness.
- Qualify every reference version against its declared supported stage. Exercise versions on a common domain for forward parity and, where relevant, backward parity.
- Check that live-kernel and saved-answer checks still load only the approved imports, setup, answer, and verifier sections. A reference progression must not leak into learner evaluation.
- For softmax, cover row-shift invariance, extreme logits, dimension handling, noncontiguous inputs, float64 gradients, and a decisive low-precision long-reduction example. A failure demonstration reports its limitation explicitly.
- For paged attention, cover repeated and reordered pages, partial pages, empty context, unused nonfinite cells, referenced-value validation, numerical score range, and gradients. Verify the streaming version's stated forward-storage property separately from its numerical output.
- For MoE dispatch, cover stable permutation, repeated token routes, sparse expert execution, empty batches, malformed expert replies, and gradients through input, mixture weights, and selected expert parameters. State the expert semantics used in comparisons.
- Use Hypothesis when generated shapes, page tables, route patterns, or permutations reveal a real remaining risk. Avoid tests that merely mirror an implementation.
- Run correctness checks before measurements. Do not put measured speedups into correctness assertions. Measurements are reproducible experiments with recorded conditions.
- Prior art includes the bank's existing cumulative verifiers, dense paged-attention oracle, sparse/dense MoE gradient parity, and saved-answer isolation tests.
- Run the full reference eval gate, the repository test gate, focused pilot qualification, and script syntax checks before landing the pilot.
- Test data retrieval and replay through their learner-facing interface: names and stage filtering, fresh mutable inputs, actual/expected retention, structured mismatches, correct expected exceptions, and unexpected candidate errors.
- Prove that retrieving eval data does not execute the reference section, teaching probes, or runner. Check that saved-answer evaluation skips question-script exploration cells.
- Use deliberately wrong answers to confirm the exposed comparisons identify the relevant failing case and retain enough information to reproduce it. A passing reference alone does not qualify the debugging experience.
- Test replay after an implementation mutates inputs or accumulates gradients. A fresh case must recover the original fixture and expected outcome while retaining intentional within-case aliases.

## Out of Scope

- Migrating all 102 exercise pairs before qualifying the pilot pattern.
- A fixed number of reference versions imposed on every problem.
- CUDA or distributed kernel implementations, GPU throughput promises, or a general benchmark platform.
- Automatically ranking implementations by speed, hiding reference algorithms in shared abstractions, or replacing the interactive scripts with ordinary modules.
- Implementing the separate failure-feedback, evaluation-session, or whole-bank qualification recommendations from the architecture report.

## Further Notes

The user confirmed complete runnable reference versions on October 7, 2026. No extra approval is needed for that representation choice. The proposed implementation order is a complete softmax pilot, then independently complete paged-attention and MoE-dispatch pilots, then combined qualification and documentation. The existing three learner stages remain contract checkpoints; reference progressions expose the implementation journey.

The user subsequently requested inspectable, schematized eval data and manual execution against that data. This extends the pilot to include the exploration loop alongside progressive references. The first slice should demonstrate retrieving an actual softmax case, calling the current kernel answer, retaining a useful mismatch, revising the answer, and replaying fresh inputs. Later slices exercise structured paged-attention inputs and MoE module/gradient state.

## Resolution

Implemented all four tickets and landed locally on main. Softmax, paged
attention, and sparse dispatch now expose complete progressive references
plus inspectable case data, manual comparisons, and fresh selected replay.
The scripts remain top-level Emacs/Jupyter workouts with one learner answer
cell and reference-free candidate/verifier loading.

Qualification covers seven reference versions, nine saved-candidate stage
combinations, actual Emacs/Jupyter edit/fix/replay across all three pilots,
102 reference scripts, and 49 repository tests (seven optional skips).
Standards and Spec review findings were fixed and independently confirmed.
See `review.md`, ticket 04's evidence, and `docs/practice/interactive-eval.md`.
All feature worktrees were removed; concurrent unrelated work was preserved.
No remote operations were performed.
