# 02: Follow-up 3D, batches, order, and gradients

**What to build:** Extend the same workout functions so the learner's implementation handles inferred spatial dimensionality, batches, empty active sets, and autograd while preserving the submanifold contract. A completed answer should keep input row order unchanged, prevent batch bleed, and match the dense oracle for both forward values and gradients.

Blocked by: 01: Baseline 2D submanifold rulebook workout

Status: resolved

- [x] The follow-up stage reuses the same answer cell and functions rather than introducing a separate 3D workout or alternate API.
- [x] The follow-up opens with a predict-first cell showing that an active site with an all-zero feature vector is still an active output row.
- [x] Verifiers cover 3D coordinates, multiple batches, empty input, unsorted coordinates, and preservation of input row order.
- [x] Asymmetric weights catch flipped kernels and transposed offset ordering.
- [x] Backward checks show gradients with respect to features and weights match the dense oracle.

## Comments

- 2026-10-11: Implemented ticket 02 on `work/submanifold-conv-dojo/ticket-02`. The reference implementation in `dojo/0121_submanifold_conv_eval.py` now infers spatial dimensionality from `coords`, keeps batch in the active-row key, preserves input row order, handles empty active sets, and applies rulebook gathers through differentiable matrix multiplies. The question script keeps starter `pass` bodies in the answer cell.
- 2026-10-11: Added follow-up eval cases for 3D unsorted multibatch coordinates, an all-zero active feature row, empty 3D input, asymmetric weights, and feature/weight gradients against the dense oracle.
- 2026-10-11: Verification passed for the reference with `PYTHONPATH=. uv run python dojo/0121_submanifold_conv_eval.py --stage followup`. The saved starter candidate loaded and failed for answer-return reasons with `PYTHONPATH=. uv run python dojo/0121_submanifold_conv_eval.py --candidate --stage followup`.
