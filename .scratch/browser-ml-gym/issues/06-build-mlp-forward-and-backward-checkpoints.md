# 06: Build MLP forward and backward checkpoints

**What to build:** A learner initializes a small NumPy MLP and solves separate forward and backward checkpoints while inspecting shapes and gradients.

**Blocked by:** 01: Edit and check `0059` in the browser; 04: Explore `0059` visually

**Status:** resolved

- [x] The lab has canonical native Python source and checks plus a browser notebook derived from them.
- [x] Fixed seeds and a bundled two-dimensional dataset make initialization, forward, and backward results repeatable.
- [x] Each checkpoint can be checked independently in an isolated worker, with shape and gradient evidence displayed in the notebook.
- [x] Reference solutions pass; starter and a broken gradient fail; a documented native check command works.

## Comments

Resolved with canonical MLP fixture, starter, reference, mutant, and checkpoint
checker modules under `browser_gym/labs/mlp/`. The browser catalog now derives
`notebooks/mlp_forward_backward.ipynb` and `checkers/mlp_forward_backward.json`
from those sources, and `/lab/?exercise=mlp` exposes independent `init`,
`forward`, and `backward` checks with shape and gradient evidence.
