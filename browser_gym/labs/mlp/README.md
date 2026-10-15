# NumPy MLP Modeling Lab

This lab is the browser-gym MLP capstone. The canonical learner source is
`starter.py`, the working reference is `reference.py`, deterministic train,
validation, and test splits live in `fixtures.py`, and `checks.py` owns the
native verifier logic used by the browser checker bundle.

Run the reference through every native checkpoint from the repository root:

```sh
python -m browser_gym.labs.mlp.checks --candidate reference --checkpoint all
```

Useful focused checks:

```sh
python -m browser_gym.labs.mlp.checks --candidate starter --checkpoint forward
python -m browser_gym.labs.mlp.checks --candidate mutants --checkpoint backward
python -m browser_gym.labs.mlp.checks --candidate mutants --checkpoint diagnose
```

The independent checkpoints are `init`, `forward`, `backward`, `train`,
`inference`, `evaluate`, `state`, and `diagnose`. The training checkpoint runs
one bounded mini-batch SGD epoch with momentum on the fixed training split.
Inference checks batch-size-invariant probabilities, evaluation checks test
metrics without parameter mutation, and state checks deep save/restore of both
parameters and optimizer velocity.

The `diagnose` checkpoint is the planted-defect task. It compares the supplied
`loss_and_grads` implementation to directional finite differences, shows the
fixed broken-vs-corrected validation-loss curves, and requires
`diagnose_mlp_defect()` to return a short diagnosis dictionary naming
`omitted_batch_mean`, the failed invariant, the supporting numerical evidence,
the repair, and the backward regression check. The planted mutant intentionally
omits the mini-batch mean scale from `dlogits`; the corrected implementation
divides `dlogits` by the batch size before computing every gradient block.

Later checkpoints can be checked with an explicit supplied reference component
while keeping credit separate. Add a top-level declaration such as:

```python
REFERENCE_OVERRIDES = {"loss_and_grads"}
```

Submitted browser source cannot import `browser_gym.labs.mlp.reference`
directly. The checker only honors supplied components named in
`REFERENCE_OVERRIDES`, and receipts record those substitutions with
`independent_credit = false`.

Receipts record the substituted component and report
`owned_checkpoint_passed = false` only when the checked checkpoint itself was
substituted.

`save_state` returns a snapshot dict with `params` and `optimizer` keys.
`params` contains copied `W1`, `b1`, `W2`, and `b2` arrays. `optimizer`
contains an integer `step` and a `velocity` dict with one copied array per
parameter block. `restore_state` returns `(params, state)` using arrays that are
independent from the snapshot.
