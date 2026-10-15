# Explore, fix, and replay

The progressive reference pilots are softmax (`0002`), paged attention (`0093`), and sparse MoE dispatch (`0101`). Their question and eval files remain cell-based scripts: open them in Emacs, connect a Jupyter kernel, and execute selected `# %%` cells. Keep your working implementation in the question's single `[answer]` cell. Run its definition again after every edit.

## Inspect one failure

From `dojo/`, execute the imports and answer cells, then try the following cells. The variables stay in your kernel and can be inspected between executions.

```python
from torch_judge import get_eval_data, compare, replay, check

eval_file = "0002_softmax_eval.py"
eval_data = get_eval_data(eval_file)
case = eval_data["extreme_logits"]
case.purpose, case.stage, case.generation
case.inputs, case.expected
```

```python
actual = my_softmax(**case.inputs)
result = compare(case, actual)
result.passed, result.diagnostics
result.inputs, result.actual, result.expected
```

Keep `result` while you edit the answer. Rerun the answer cell, then replay the cases you want to fix:

```python
results = replay(eval_file, my_softmax,
                 names=["extreme_logits", "noncontiguous_axis"])
failures = {name: outcome for name, outcome in results.items()
            if not outcome.passed}
failures
```

Replay calls the current function you pass. It reconstructs each selected case independently, including fresh module state and leaf tensors. The earlier `result` still records the old failure. Omit `names` to replay the whole concrete collection, or use `stage="followup"` to include baseline and follow-up cases.

Use `check(eval_file, my_softmax, stage="production")` for the complete verifier contract. Concrete cases are selected examples; they do not stand in for all property trials, gradient checks, resource checks, or stateful checks. Passing their forward comparisons alone does not establish correct autograd.

For paged attention, inspect `reordered_repeated_partial` and `unused_nonfinite_tail`. For MoE, inspect `duplicate_routes`, then `gradient_routes` and `shared_expert`. The latter intentionally uses two expert IDs that refer to the same module; fresh retrieval preserves that relationship.

## Exceptions and changed inputs

An expected rejection is a successful outcome when the exception type matches the contract:

```python
case = get_eval_data("0093_paged_attention_eval.py")["invalid_page"]
try:
    actual = paged_attention(**case.inputs)
except Exception as error:
    result = compare(case, exception=error)
else:
    result = compare(case, actual)
result.passed, result.expected_exception, result.exception, result.diagnostics
```

Replay captures expected and unexpected exceptions automatically. It returns records rather than stopping at the first candidate error.

If you edit `case.inputs`, its old expectation no longer establishes correctness for those inputs. `compare` reports that inputs changed. Retrieve a fresh case to return to the original fixture. To explore a different fixture, build a new `EvalCase` with independently computed expectations for your new inputs; changing the displayed `case.expected` does not replace the original oracle snapshot.

Result tensors are detached snapshots for inspection, not the live computation graph. To debug backward, call your answer on a fresh case, run `actual.square().sum().backward()` on the live output, and inspect the case's leaf gradients. For MoE, unpack the output/permutation tuple before backward. Retrieve fresh data before another backward attempt.

For manual comparison, `result.inputs` and `result.observed_inputs` record the values observed when you compare; `result.expectation_inputs` records the original oracle fixture. Manual comparison cannot reconstruct pre-call inputs after a mutating implementation. Replay additionally records its pre-call inputs in `result.inputs`, so its records expose both sides of a mutation. Parameter and module-state changes can also invalidate expectations. Ordinary module attributes and arbitrary side effects require problem-specific verifiers.

## Read the reference progression

Each reference version is a complete function in a separate eval-script cell. Execute the imports, any documented contract helpers, and the version's cell. You do not need to run earlier reference algorithms or the final eval runner.

| Pilot | Version | Highest qualified learner stage | What changes |
|---|---|---|---|
| Softmax | `softmax_shifted` | `followup` | Explicit maximum subtraction and normalization; input-dtype reduction |
| Softmax | `softmax_wide` | `production` | Float32 working reductions for float16/bfloat16, then restore output dtype |
| Paged attention | `paged_attention_dense` | `followup` | Gather logical tokens, trim the final page, apply dense attention |
| Paged attention | `paged_attention_streaming` | `production` | Carry online maximum, normalization mass, and weighted value sum block by block |
| Sparse dispatch | `dispatch_traceable` | `production` | Trace one weighted reply per route |
| Sparse dispatch | `dispatch_batched` | `production` | Invoke each selected expert once with its routes |
| Sparse dispatch | `dispatch_grouped` | `production` | Reuse contiguous groups from the stable sorted route layout |

Reference versions teach implementation choices. Learner stages define cumulative contracts. There need not be one version per stage, and an early version must not claim constraints it does not meet. In particular, dense paged attention does not meet the production forward-storage contract. The shifted softmax's float16 mass can overflow on 70,000 equal logits, even though maximum subtraction prevents exponential overflow.

All three MoE versions assume deterministic pointwise experts for their common-domain parity. Changing invocation batch sizes can change BatchNorm, dropout, or batch-coupled expert behavior. These portable implementations establish routing and gradient semantics; they are not fused serving kernels.

After executing the appropriate eval-script cells, select a reference explicitly:

```python
check("0002_softmax_eval.py", softmax_shifted, stage="followup")
check("0002_softmax_eval.py", softmax_wide, stage="production")
```

The strongest version remains the normal direct eval target. Your saved answer is still selected by `--candidate`; reading reference cells does not replace your question's answer.

## Request experiments explicitly

The eval scripts define opt-in experiments. Their definitions are inert until called:

```python
explore_long_half_reduction()  # softmax: demonstrate reduction overflow
benchmark_softmax()           # softmax: shifted/wide/framework comparison
explore_paged_storage()       # paged attention: timings and analytical storage
explore_dispatch_work()       # MoE: expert-call work and timings
```

Run them in the corresponding eval-script kernel context after the needed definitions. Read the workload, dtype, device, sample timings, and method in the returned report. Timings use warmup and repeated synchronized wall-clock measurements. No universal speedup is promised. Paged storage figures are analytical working-array estimates, not measured allocator peaks; they exclude physical inputs, framework workspaces, and autograd-retained state. Streaming forward temporaries can stay bounded while backward's saved state grows with context.

## Author cases and verifiers

Keep imports at the eval script's top level. Put `make_eval_data`, independent oracle helpers, comparison policies, and ordinary `verify_*` functions inside the `[verifiers]` cell. Close that cell before the runner. Retrieval loads imports and the verifier context; it does not run reference definitions, exploration cells, or the runner.

A factory returns a dictionary keyed by each case's stable `name`:

```python
# %% [verifiers]
def make_eval_data():
    logits = torch.tensor([1000., 1001.], dtype=torch.float64)
    return {
        "extreme_logits": EvalCase(
            name="extreme_logits",
            stage="baseline",
            purpose="Maximum subtraction avoids exponential overflow.",
            inputs={"x": logits, "dim": -1},
            expected=torch.softmax(logits, dim=-1),
            comparison=compare_values,
        ),
    }
```

Import `torch`, `EvalCase`, and `compare_values` before this cell. The default comparison checks nested structure, tensor/array shape and dtype, tensor device, and numerical values. Configure tolerances with an ordinary function or `functools.partial`; integer and boolean arrays/tensors compare exactly. Python and NumPy integer scalars also compare exactly without float conversion, and boolean scalars must match booleans rather than integer `0`/`1`. Integer scalars must match integers; floating scalars retain the configured numerical tolerances. For a rejection case, set `expected_exception=ValueError` (or a tuple of allowed exception types) and supply the invalid inputs.

Factory inputs remain real Python objects, not serialized display records. Make every retrieval fresh. Use local injected generators and record the seed/configuration in `generation`. Preserve intentional aliases within a case; do not reuse graphs or accumulated parameter gradients across retrievals. Build expectations independently of the learner implementation and reference versions. Framework operators, hand-computed results, and separately derived mathematics can serve as oracles.

Ordinary verifiers should share the factory for concrete fixtures. Keep gradient assertions, Hypothesis trials, resource checks, and stateful protocols as ordinary Python functions when a single input/output case would be misleading. A finite factory cannot enumerate every generated property trial. Add a new complete reference version only for a demonstrated limitation, qualify its supported stage through `check`, and compare versions on their common domain with independent forward/backward expectations.

Question scripts keep one `[answer]` cell and optional `[setup]` cells. Saved candidates execute imports plus setup/answer cells in script order. Put required constants and helper definitions in tagged cells. Leave teaching probes, concrete exploration data/results, benchmarks, and final checks in untagged cells so saved-answer loading excludes them. Live-kernel use deliberately allows running any cells you choose.
