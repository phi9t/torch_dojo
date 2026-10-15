# Autonomy and perception practice

Start with the [ten-session queue](sessions.tsv), then filter the [103-drill bank](drills.tsv) by priority, duration or question script. Record each rep in [attempts.tsv](attempts.tsv). The [fieldbook](handbook.html) contains full contracts, worked solutions and follow-up explanations. The [source ledger](sources.tsv) lists eight primary technical references.

## Practice flow

Read the stated contract before starting a drill. A dojo link points to a related runnable question script; its exact shapes, framework and edge policies may differ from the fieldbook.

The NumPy workouts use notebook-style `# %%` cells: predict, plan, implement, probe and check. Each adjacent `_eval.py` owns the reference implementation and verifiers; `torch_judge/` contains the reusable runner. From the repository root:

```bash
PYTHONPATH=. python dojo/0059_stable_cross_entropy_eval.py
PYTHONPATH=. python dojo/0059_stable_cross_entropy_eval.py --candidate
```

Start with ML01 and ML02: stable softmax cross-entropy and one linear-classifier update. Use finite float64 logits, integer targets, mean reduction and a vectorized batch computation. Test extreme logits, per-row shifts, repeated samples, gradient row sums and central finite differences. For `Z = XW + b`, calculate `dW = X.T @ dZ` and `db = dZ.sum(0)`. An ignored-target extension must define all-ignored behavior and weight unequal microbatches correctly.

Use the first rep to identify a gap. Review the solution after recording the result, then try a changed case on another day. Reading a reference does not establish recall.

## Review focus

- Clustering: empty-cluster and tie policies; labels consistent with returned centers.
- Attention: explicit shape transitions, mask meaning and backward checks.
- Geometry: named frames, boundary conventions and round trips.
- Matching and voxel pooling: collision policies and repeated-index accumulation.
- Evaluation: a defined estimand, paired resampling, data slices and simulator validation.
- Systems: quantified constraints, discriminating experiments, versioned artifacts and rollback.
- CUDA: legal synchronization and device timing.

## Technical references

The source ledger covers NumPy broadcasting, indexed accumulation and window views; PyTorch attention, cross-entropy and CUDA; and the EMMA and Sim Agents research references. These explain concepts and APIs. The drill contracts and worked solutions are authored teaching material.
