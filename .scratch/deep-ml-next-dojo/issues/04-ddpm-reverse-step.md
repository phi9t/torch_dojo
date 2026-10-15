# 04: Practice a DDPM reverse sampling step

**What to build:** A learner can compute one noise-prediction DDPM posterior step and distinguish its mean from the optional stochastic term. Reserve exercise 0092. The approved parent spec governs the common question/eval script contract.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The question/eval pair uses top-level cells, one editable answer section, a numerical probe, and cumulative baseline, follow-up, and production verifiers.
- [x] Baseline checks the posterior mean against a hand-calculated example with an explicitly one-based schedule index.
- [x] Follow-up checks the no-noise final step and a seeded earlier step, including that the final step does not consume the generator.
- [x] Production validates schedule bounds and beta range, shape/dtype/device/finite agreement, leaves inputs unchanged, and preserves gradients through the noisy input and predicted noise.
- [x] The reference passes direct evaluation; saved candidate stage selection works; a candidate using raw beta instead of posterior variance fails a decisive check. Add the exercise to the dojo index with a source link and original wording.

## Comments

Merged exercise #92 and its original, source-linked README entry. The shared eval environment passed all 5 reference verifiers. Saved blank candidate baseline selection ran 1 check and failed as expected. The supplied-noise check rejected a raw-beta-variance mutant; production checks cover invalid inputs, gradients, and immutability.
