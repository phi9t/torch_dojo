# 01: Baseline 2D submanifold rulebook workout

**What to build:** Add the baseline version of the `0121_submanifold_conv` workout so a learner can implement a 2D, 3x3 submanifold sparse convolution as a rulebook lookup plus per-offset gather, matrix multiply, and add. The workout should teach that active output rows are exactly the input rows, inactive neighbours contribute zero, and the center offset maps each active site to itself.

Blocked by: gather-scatter-drills completion (drills 0114-0120 come first)

Status: resolved

- [x] The question script follows the dojo interactive-script contract: ordinary top-level cells, exactly one answer cell containing the two learner functions, and a baseline `check(...)` call.
- [x] The baseline opens with a predict-first cell about a hole in a row of active sites and whether submanifold convolution creates an output there.
- [x] The eval script provides a standalone reference implementation and baseline verifiers that check a hand-sized 2D rulebook fixture.
- [x] The baseline output matches an independent dense-convolution oracle sampled at active input sites, with cross-correlation offset order rather than flipped convolution order.
- [x] The candidate starter loads through saved-file evaluation and fails the baseline for an answer-related reason.

## Comments

Resolved in ticket branch. Verification evidence:

- `PYTHONPATH=. uv run python dojo/0121_submanifold_conv_eval.py` passes both baseline checks.
- `PYTHONPATH=. uv run python dojo/0121_submanifold_conv_eval.py --candidate` loads the saved starter and fails because the answer functions return `None`, confirming an answer-related failure.
