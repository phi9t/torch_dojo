# 07: Complete the MLP experiment

**What to build:** A learner trains, evaluates, and restores the MLP, using reference components when needed to continue beyond an unfinished checkpoint.

**Blocked by:** 03: Hints, new attempts, and export; 06: Build MLP forward and backward checkpoints

**Status:** resolved

- [x] Independent checkpoints cover mini-batch training, batched inference, evaluation, and saving/restoring parameters and optimizer state.
- [x] Explicit Run actions produce bounded loss, decision-boundary, confidence, and gradient-norm views on fixed data splits.
- [x] A learner can explicitly substitute a supplied reference component to continue; receipts record each substitution and do not credit that component as independently solved.
- [x] Reference checkpoints pass in native and browser checks, while selected wrong answers fail with useful evidence.

## Resolution

Implemented the complete NumPy MLP lab checkpoints: `init`, `forward`,
`backward`, `train`, `inference`, `evaluate`, and `state`. The browser MLP
lesson now exposes explicit bounded Run actions for loss, decision boundary,
confidence, and gradient norms on the fixed splits. MLP checker receipts include
reference substitutions and `owned_checkpoint_passed` so substituted components
are recorded without independent credit.

Verification:

- `python -m pytest browser_gym/labs/mlp/test_checks.py browser_gym/tests/test_build_catalog.py -q`
- `python -m pytest browser_gym/tests browser_gym/labs/mlp -q`
- `npm --prefix browser_gym/extension run typecheck`
- `npm --prefix browser_gym/extension test`
- `make -C browser_gym build`
- `DOJO_TEST_PORT=8107 npm --prefix browser_gym/extension run test:e2e -- mlp.spec.ts`
- `DOJO_TEST_PORT=8107 make -C browser_gym test`
- `python -m pytest -q`
