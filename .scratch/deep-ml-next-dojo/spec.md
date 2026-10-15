Status: resolved

# Next ML/AI dojo batch from the Deep-ML live catalog

## Problem Statement

The live Deep-ML catalog contains useful ML/AI coding topics that the dojo does not yet teach. The research note identifies 54 source-linked questions, but a flat import would repeat existing exercises and dilute the learner's edit-and-check flow. Learners need a small next batch that extends current post-training, generative-model, inference, and evaluation exercises with clear numerical contracts and meaningful follow-ups.

## Solution

Add six original, bite-sized question/eval pairs, selected from the strongest uncovered leads: pairwise reward-model training, REINFORCE leave-one-out (RLOO), on-policy trajectory sampling, one DDPM reverse step, paged attention, and binary probability calibration. Each question remains a top-level Emacs/Jupyter script with one editable answer section and baseline, follow-up, and production checks. Its neighboring eval script contains a runnable reference and ordinary Python verifiers. The learner can revise one answer in a live kernel or evaluate the saved script without running teaching probes.

This batch extends the existing dojo; it does not reproduce Deep-ML's wording, tests, or solutions. The original survey remains a bank of later leads.

## User Stories

1. As a learner, I want a short prediction or hand-calculation prompt before coding, so that I can reason about the mechanism first.
2. As a learner, I want one answer section per question, so that I can revise the same working solution through all stages.
3. As an Emacs/Jupyter learner, I want top-level cells and helper definitions, so that I can run and inspect steps interactively.
4. As a shell learner, I want saved-file evaluation to skip teaching probes, so that my edited answer is checked reproducibly.
5. As a learner, I want baseline checks to be small and decisive, so that I can tell whether I have the core algorithm.
6. As a learner, I want follow-up checks to change one meaningful constraint at a time, so that I can see why the initial solution is insufficient.
7. As a learner, I want production checks to state validation, numerical, and scale behavior explicitly, so that passing means the kernel is reusable under its stated contract.
8. As a learner, I want cumulative stages, so that later passes cannot hide a baseline regression.
9. As a learner, I want the reward-model exercise to distinguish training a scalar reward model from updating a preference policy with DPO.
10. As a learner, I want a stable pairwise reward loss and strict preference accuracy, so that ties, margins, and extreme score gaps have clear meanings.
11. As a learner, I want RLOO advantages grouped by prompt, so that another prompt's responses cannot enter a leave-one-out baseline.
12. As a learner, I want the RLOO exercise to distinguish fixed rollout data from the current trainable policy, so that gradients reach only the intended log-probabilities.
13. As a learner, I want to sample an MDP using an injected RNG, so that I can reproduce action choices, transitions, and trajectory statistics.
14. As a learner, I want terminal and horizon behavior defined, so that state visits and discounted returns have unambiguous denominators.
15. As a learner, I want one DDPM reverse step with a known hand-computable example, so that I can separate posterior mean from added noise.
16. As a learner, I want the final denoising step and seeded stochastic steps checked separately, so that noise is applied only when required.
17. As a learner, I want paged attention to agree with dense attention on a tiny cache, so that block indexing has a trustworthy reference.
18. As a learner, I want partial final blocks, nonsequential physical pages, and invalid table entries covered, so that the page table is more than a contiguous-cache disguise.
19. As a learner, I want the production paged-attention task to bound temporary memory, so that the exercise teaches why paging is useful.
20. As a learner, I want Platt and isotonic calibration fitted only on calibration data, so that evaluation examples do not leak into fitting.
21. As a learner, I want ties, constant scores, and held-out reliability checks covered, so that calibration does not appear to work only on easy toy samples.
22. As a learner, I want every exercise to link its motivating source and identify authored extensions, so that I do not mistake dojo constraints for an external prompt.
23. As an exercise author, I want to keep reference implementations and verifier helpers in eval scripts, so that the shared judge remains focused on stage selection and reporting.
24. As an exercise author, I want checks that fail plausible wrong implementations, so that the reference passing alone is not the quality bar.
25. As a maintainer, I want the all-reference suite and shared judge suite to stay green, so that the new questions do not disrupt current practice.

## Implementation Decisions

- Use the next available four-digit exercise numbers when implementation begins. Keep the established question/eval naming convention and one tagged answer section per question. Ordinary imports, setup cells, teaching probes, and check calls stay at top level; no `main()` function or answer module is introduced.
- Use the existing `check` and `evaluate` seam. Stage names and cumulative behavior follow the existing baseline, follow-up, and production contract. Verifiers may share top-level imports and helpers inside the verifier section. No new judge, template, or task registry is needed.
- Each question explains its inputs, shapes, output, reductions, boundary cases, and invalid-input policy before its checks. Each eval exposes a reference implementation for direct execution and at least one verifier per stage. Exercise-specific generated cases are welcome when they catch a real invariant; a dependency on Hypothesis is not required merely to increase case count.
- **Pairwise reward model:** accept finite chosen and rejected score vectors of equal positive length plus a finite nonnegative margin. Return a differentiable mean logistic loss and strict chosen-greater-than-rejected accuracy. Baseline uses zero margin; follow-up applies the margin and tests ties; production uses stable softplus, preserves input tensors, and rejects malformed inputs. Accuracy is computed from unshifted scores and is not part of the gradient.
- **RLOO:** accept current sequence log-probabilities, fixed scalar rewards, and integer prompt-group IDs. Optionally accept frozen rollout-policy and reference sequence log-probabilities with a nonnegative KL coefficient for reward shaping. If the coefficient is positive, require both frozen log-probability vectors and shape each reward by subtracting the coefficient times rollout-minus-reference log-probability. Return per-completion advantages and a scalar mean policy-gradient loss. A response's baseline is the mean shaped reward of *other* responses in its own group. Singleton groups have zero advantage. Mean reduction is per completion across the batch. Rewards and shaping quantities are fixed data; only current log-probabilities receive policy-loss gradients. An empty batch returns empty advantages and graph-connected zero loss. Baseline covers one group, follow-up covers interleaved unequal groups and KL shaping, and production covers singleton, empty-batch, nonfinite, and no-mutation cases.
- **On-policy trajectory sampling:** accept finite stochastic transition and policy tables, transition rewards, one start state, terminal states, a discount, positive episode count and horizon, and an injected NumPy generator. Transition and policy rows must be nonnegative and sum to one within a stated tolerance. Return ordered episodes of transition records (state, action, next state, reward, terminal flag), state-visit frequencies, state-action counts, and mean discounted episode return. Count the initial state once per episode and each reached successor once; never choose an action from a terminal state. A terminal start creates an empty transition trajectory, one initial-state visit, and zero return. Normalize visitation by all counted visits and mean return by episode count. Baseline uses deterministic transitions, follow-up uses stochastic transitions and termination, and production validates distributions, reproducibility, count conservation, and horizon truncation.
- **DDPM reverse step:** use a beta schedule indexed explicitly by a one-based scalar denoising step shared by all elements of an input tensor. Compute the standard noise-prediction posterior mean and posterior variance; at the final step add no random noise and do not consume the generator. Accept a supplied noise tensor for deterministic checks and an injected generator when noise must be drawn. Baseline checks a small known mean, follow-up checks final-step and stochastic behavior, and production covers batch shapes, schedule bounds, finite inputs, dtype/device agreement, and input immutability. Preserve gradients through the model prediction and noisy input where the tensor contract permits them.
- **Paged attention:** accept a query per head, physical key/value blocks, a logical-to-physical block table, and context length. Return one attention output per head with stable softmax. A context length of zero returns zeros. Repeated physical page IDs are allowed and read each time they occur logically; invalid indices or excess context length raise `ValueError`. Baseline checks one and two blocks against dense attention; follow-up checks shuffled pages and a partly filled final block; production processes blocks without reconstructing the full logical KV sequence, retains differentiability for query and referenced cache values, and validates shapes, dtype/device, and no mutation. Sequential block iteration is allowed.
- **Binary calibration:** expose one candidate entry point that fits on nonempty calibration scores and binary labels, then predicts for separate query scores under a named Platt or isotonic method. Platt fitting minimizes stable logistic loss with a fixed L2 coefficient of `1e-3` on both slope and intercept, keeping the two-parameter fit finite on separable or one-class data. Isotonic fitting uses weighted pool-adjacent-violators after combining tied scores; prediction linearly interpolates between distinct fitted scores and clamps outside their range. A single distinct score yields a constant prediction. Baseline checks Platt on a small example, follow-up checks isotonic monotonicity and tied scores, and production covers constant scores, one-class labels, finite outputs, no mutation, and the absence of test-label leakage. Held-out calibration metrics are verifier inputs, not extra labels passed into fitting.
- Keep each pair independently implementable. RLOO may refer learners to the existing REINFORCE and GRPO exercises; paged attention may refer to dense attention and KV cache exercises; calibration may refer to the current metric evaluator. Such references are teaching links, not runtime imports between exercise scripts.
- Add the six pairs to the dojo index and describe their relationship to the current exercises. Keep source attribution to the Deep-ML problem pages and paper-level sources where necessary, while using original explanations, examples, contracts, reference code, and checks.

## Testing Decisions

- Test at the existing public exercise seam: run each eval script directly against its reference, then run saved-file candidate checks at each stage. Keep the repository's all-reference eval and shared judge tests green.
- Each verifier checks observable behavior or a mathematical invariant, not source structure. Hand-computed probes and independent dense/naive references take priority over copying the reference implementation into a test.
- Reward-model checks include an extreme wrong-way score gap, a nonzero reference margin, a tie whose accuracy is zero, and policy-score gradients. A constant or swapped implementation must fail.
- RLOO checks include two interleaved prompts with different reward scales, a singleton, unequal group sizes, a shaped-reward case, and gradient isolation. Global reward normalization or inclusion of a response in its own baseline must fail.
- Trajectory checks include a deterministic hand trace, a terminal start, seeded stochastic replay, changed-seed behavior, normalization of visits, and conservation between trajectory records and reported counts. A greedy sampler or a sampler that acts after termination must fail.
- DDPM checks include a known posterior mean, zero noise at the last step, seeded noise at an earlier step, schedule-index boundaries, and finite gradients. A sampler using `beta_t` in place of the posterior variance must fail.
- Paged-attention checks include out-of-order physical pages, a partially filled last page containing adversarial unused values, a repeated page ID, empty context, dense output/gradient parity, and a larger context. Review must verify block-bounded temporary storage; an optional memory measurement may support this, but the default eval must not depend on a fragile process-memory threshold.
- Calibration checks compare Platt fitting with an independent small convex reference, verify isotonic monotonicity and tie aggregation, and evaluate on held-out labels that fitting never receives. Tests should distinguish predicted probabilities from raw scores and reject accidental query-label use.
- Before considering the batch complete, review each pair for one plausible incorrect learner solution that existing verifiers reject, inspect numerical edge policies, and run the whole reference suite and shared tests once after integration.

## Out of Scope

- Turning all 54 catalog leads into scripts in this batch.
- Copying Deep-ML prompts, tests, solutions, or platform interfaces.
- A full RLHF trainer, reward-model dataset pipeline, distributed rollout service, complete diffusion model, GPU PagedAttention kernel, or deployed calibration service.
- Changing the shared judge, introducing per-exercise task registries, or requiring GPU hardware for the default eval suite.
- Marking any learner attempt complete in a practice tracker.

## Further Notes

The source survey is the [Deep-ML live catalog follow-up](../../docs/research/deep-ml-live-catalog-2026-10.md). The six topic leads are [reward-model loss](https://www.deep-ml.com/problems/484), [RLOO](https://www.deep-ml.com/problems/1266), [on-policy sampling](https://www.deep-ml.com/problems/529), [DDPM reverse sampling](https://www.deep-ml.com/problems/396), [PagedAttention](https://www.deep-ml.com/problems/492), and [calibration fitting](https://www.deep-ml.com/problems/1466). Deep-ML's open-problem repository carries an Educational Use Only license; this spec calls for original dojo material.

The existing ADRs preserve interactive scripts and cumulative stages. “Production” means a reliable kernel under the stated exercise contract, not a production service. This spec deliberately stops at a buildable six-exercise batch; later catalog leads can be selected after learners try these and the quality of the new pairs is reviewed.
