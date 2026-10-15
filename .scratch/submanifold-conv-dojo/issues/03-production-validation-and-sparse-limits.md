# 03: Production validation and sparse limits

**What to build:** Bring the workout to its production contract by making invalid sparse-coordinate inputs fail before unsafe lookup or dense allocation, while keeping ordinary strided feature tensors and supported floating dtypes working. The production stage should make the flat-key and rulebook limitations explicit without adding performance gates.

Blocked by: 02: Follow-up 3D, batches, order, and gradients

Status: resolved

- [x] The production stage opens with a predict-first cell showing why neighbouring flat keys must not alias across spatial or batch boundaries.
- [x] Verifiers reject duplicate coordinates, out-of-range coordinates, and batch or spatial key-space overflow with `ValueError` before lookup or dense allocation.
- [x] Verifiers catch key aliasing at low and high spatial boundaries, including wrapped x coordinates and out-of-range y, z, or batch coordinates.
- [x] Strided feature tensors, float32 and float64 values, empty inputs, and zero-feature active rows remain valid under the production checks.
- [x] The workout includes a sizing discussion for rulebook memory and frames optional custom-backward or timing ideas as exploratory notes, not correctness gates.

## Comments

- Resolved by adding production-stage validation for duplicate, out-of-range, rank-mismatched, and overflowing sparse metadata before rulebook lookup or dense oracle allocation. The production eval also keeps valid strided float32 features, float64 values, empty inputs, and all-zero active feature rows passing.
- Evidence: `PYTHONPATH=. uv run python dojo/0121_submanifold_conv_eval.py --stage production` passes 8/8. The saved starter candidate loads with `--candidate --stage production` and fails for answer-stub reasons.
