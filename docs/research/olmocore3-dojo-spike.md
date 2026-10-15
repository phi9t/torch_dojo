# Olmo-core 3: dojo workout spike

Source checked October 6, 2026: Ai2's 168-page technical report, [*Supercharging Olmo-core for Efficient and Scalable MoE Training*](https://allenai.org/papers/olmocore3), and its [public Olmo-core implementation](https://github.com/allenai/OLMo-core). This is a proposal for original practice exercises, **not** a claim that Ai2 publishes or endorses these prompts. The paper's throughput measurements require particular B300 clusters, parallel layouts, and routing workloads; the CPU-sized exercises below test the underlying contracts, not those performance claims ([report §14 and Appendix B](https://allenai.org/papers/olmocore3#page=103)).

**Spike verdict:** Start with the three routing workouts below. Each has a small input, a hand-checkable result, and a distinct failure case. A CPU probe reproduced the report's six-token result: requested counts `(2,4)`, mean soft scores `(0.6,0.4)`, auxiliary loss `14/15`, and requested max/mean load `4/3`. This confirms that the proposed diagnostic can catch a lower surrogate loss with worse actual balance. It does not validate the report's cluster throughput or training trajectory. No dojo exercise is implemented by this spike.

## Recommended first slice

| Priority | Original workout | Paper mechanism | Why it belongs here |
| --- | --- | --- | --- |
| 1 | **Requested routes, kept routes, and workload diagnostics** | [§10.1–10.2, Eqs. 19–22](https://allenai.org/papers/olmocore3#page=68) | Establishes the state that all later exercises need. |
| 2 | **Shared rank capacity** | [§10.3–10.4, Eqs. 23–29](https://allenai.org/papers/olmocore3#page=70) | A compact deterministic admission algorithm with sharp edge cases. |
| 3 | **Token gerrymandering diagnostic** | [§10.5 and §10.7, Eqs. 30, 38–45](https://allenai.org/papers/olmocore3#page=72) | A surprising, independently checkable failure of a familiar surrogate objective. |
| 4 | **Post-batch router bias feedback** | [§10.6, Eqs. 35–37](https://allenai.org/papers/olmocore3#page=74) | Adds state, update timing, and gradient separation to routing. |
| 5 | **Collapse ungated shared experts** | [§20.1, Eqs. 74–78](https://allenai.org/papers/olmocore3#page=119) | A small PyTorch reparameterization with forward and backward parity. |
| 6 | **Routed/shared merge scale** | [§21, Eqs. 80–86](https://allenai.org/papers/olmocore3#page=121) | Connects a model contract to initialization variance and gradients. |

These are new mechanisms beyond [dojo/0028_moe.py](../../dojo/0028_moe.py), which already covers top-k selection, normalized selected weights, sparse dispatch, and ties. Avoid a generic “implement an MoE” repeat. The [INT8 quantization exercise](../../dojo/0036_int8_quantization.py) and [paged attention exercise](../../dojo/0093_paged_attention.py) also make a generic quantization or buffer-indexing workout less compelling. The exercise concepts, rather than an analogy to MXFP8 or NVSHMEM, drive this shortlist.

## Contracts and decisive checks

### 1. Route accounting and four distinct imbalance views

**Authored API:** Given selected expert IDs `[T,K]`, ownership `expert_to_rank[N]`, and a keep mask `[T,K]`, return integer requested/kept counts per expert, requested/kept counts per destination rank, and dropped counts. Require each token's selected expert IDs to be distinct and in range. The report defines one `(token, expert)` pair as a route; rank totals are sums over experts owned by the destination. The kept counts determine dispatched rows and expert compute, while requested counts feed the current load-balancing controller ([§10.1 and Table 16](https://allenai.org/papers/olmocore3#page=68)).

**Baseline:** `bincount` requested expert IDs; sum by owner. **Follow-up:** Apply the keep mask and report `requested - kept`, `sum(requested)=T*K`, `sum(kept)+sum(dropped)=T*K`. **Production:** Compute max/mean imbalance separately for requested experts, kept experts, requested ranks, and kept ranks, with an explicit zero-total policy. A capped hot rank can appear balanced in the *kept* histogram while requested demand was badly skewed ([§10.2, Eq. 22 and Table 17](https://allenai.org/papers/olmocore3#page=69)). Tests: empty input, multiple sources, one rank owning multiple experts, and a hot requested expert whose excess routes are dropped.

### 2. Shared destination-rank capacity

**Authored API:** Inputs are a nonnegative requested-count matrix `[source_rank, expert]`, expert ownership, and integer capacity per destination rank. Return a kept-count matrix, with `0 <= kept <= requested` and each destination's sum at most its capacity. For a deterministic policy, iterate destination ranks, then local experts in ID order, then source ranks in ID order; admit a prefix of each source/expert split until the shared destination budget is exhausted. This order follows the report's current rowwise keep matrix, but a CPU exercise can state it directly without NVSHMEM ([§10.3](https://allenai.org/papers/olmocore3#page=70)).

**Baseline:** Single destination with a shared integer budget. **Follow-up:** Multiple source/destination ranks and deterministic partial-split admission. **Production:** Compare with private per-expert budgets whose sum equals the shared budget **after rounding**. With requests `(5,1)` and private capacities `(3,3)`, private budgets keep 4/6 on the rank while shared capacity 6 keeps 6/6; the paper's two-rank example adds a second balanced `(3,3)` rank ([Figure 41](https://allenai.org/papers/olmocore3#page=71)). Verify `dropped_shared <= dropped_private` for every nonnegative histogram at matched aggregate capacity, since `max(0,sum(x)-sum(c)) <= sum(max(0,x-c))` ([§10.4, Eq. 29](https://allenai.org/papers/olmocore3#page=71)). This is a statement about **number** of dropped routes, not model quality or which tokens survive. An empty histogram, zero capacity, source-order changes, and integer-rounding mismatches should be tested.

### 3. Token gerrymandering: loss is not a balance certificate

**Authored API:** For normalized scores `p[T,N]` and top-k IDs `[T,K]`, compute detached hard frequencies `f_i=count_i/(TK)`, mean soft mass `P_i=mean_t p[t,i]`, scalar `L=N*sum_i(f_i*P_i)`, and requested max/mean imbalance `I=N*max(f)`. Treat selected IDs as the hard routing result; verify their score ordering, with a documented tie policy. Gradients pass through `P`, not the hard counts. These definitions are [Eqs. 30, 38–41](https://allenai.org/papers/olmocore3#page=75); the [public loss implementation](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/moe/loss.py) is a useful cross-check.

**Baseline:** Uniform balanced scores give `L=1`, `I=1`. **Follow-up:** Reproduce the report's six-token top-1 example with probability rows `(1,0)` twice and `(0.4,0.6)` four times. It has counts `(2,4)`, `f=(1/3,2/3)`, `P=(3/5,2/5)`, `L=14/15<1`, and `I=4/3>1`; compare with three `(0.8,0.2)` and three `(0.4,0.6)` rows, which retain `P=(3/5,2/5)` but have balanced hard counts and `L=1` ([§10.7.3, Eqs. 43–45 and Figure 43](https://allenai.org/papers/olmocore3#page=77)). The exact `(1,0)` row is a hand-computable probability example, not the softmax of finite logits. **Production:** Check `L=1+N*(f-u)·(P-u)` numerically; compare local versus combined-batch balancing and verify soft-score gradients using finite logits and finite differences while selections remain unchanged. Keep hard counts detached. Explicitly separate the scalar from direct hard-count diagnostics. The paper reports a checkpoint-branch experiment where the auxiliary scalar fell while actual imbalance rose; the six-token exercise demonstrates the algebra, not the empirical trajectory ([Figure 42](https://allenai.org/papers/olmocore3#page=76)).

### 4. Router bias as a post-batch controller

**Authored API:** Given unbiased router logits `z[T,N]`, persistent bias `b[N]`, top-k, and step size `gamma`, select on `z+b` but compute selected mixture weights from `z`. Accumulate requested IDs over a batch; only after the final microbatch update `b_i += gamma*sign(mean(counts)-counts_i)` and reset the counts. This is the report's auxiliary-loss-free feedback rule ([§10.6, Eqs. 35–37](https://allenai.org/papers/olmocore3#page=74)) and matches the [router's `post_batch` update](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/moe/v2/router.py#L407-L429).

**Baseline:** Underused expert bias rises; overused bias falls. **Follow-up:** Two microbatches update once after accumulation, not twice. **Production:** Verify detached persistent state, stable deterministic ties, unbiased output weights, and zero-count/all-equal-count behavior. A small `gamma` or oscillatory sequence can illustrate that this is a controller, not a proof of convergence. The report does **not** establish this controller's model-quality or stability effect in Olmo-core 3 ([§10.6 limitation](https://allenai.org/papers/olmocore3#page=74)).

### 5. Collapse always-active shared SwiGLU experts

**Authored API:** Given several ungated SwiGLU experts on the same `x[...,D]`, each with gate/up matrices `[D,m_j]` and down matrix `[m_j,D]`, construct one width `sum(m_j)` expert: concatenate gate/up weights on the intermediate axis and stack down weights on its input axis. `SiLU(xWg) * (xWu)` acts independently by intermediate coordinate, so the new output equals the sum of old outputs ([§20.1, Eqs. 74–76](https://allenai.org/papers/olmocore3#page=119)).

**Baseline:** Forward equality on unequal widths. **Follow-up:** Map each new weight-gradient slice back to an original expert and check equality under the same scalar loss. **Production:** Include biases with an explicit merge rule, non-contiguous input, and fixed per-branch output coefficients absorbed into down weights. A common input-dependent scalar multiplying the *whole* sum still collapses; distinct per-expert dynamic gates generally do not ([§20.1–20.2, Eqs. 77–78](https://allenai.org/papers/olmocore3#page=119)). Keep the claim to function/gradient equivalence; the report does not claim that one wide layout always runs faster.

### 6. Merge coefficients and initialization scale

**Authored API:** Given selected routed outputs `[T,K,D]`, selected probabilities `[T,K]` summing to one, and shared output `[T,D]`, return `sum_i K*p_i*f_i + g`. Check that every routed branch coefficient is one for uniform probabilities; this matches the exact partition of a dense MLP into `K` routed-width blocks plus one shared-width block ([§21.1, Eqs. 80–84](https://allenai.org/papers/olmocore3#page=122)).

**Baseline:** Uniform probabilities recover an additive dense partition. **Follow-up:** Compare routed unit-sum weights `p_i` with `K*p_i`, including routed and shared parameter-gradient scales. **Production:** With independent centered branch outputs and variance proportional to branch width, predict `R_var=(d_r*K^2*sum_i p_i^2+d_s)/(K*d_r+d_s)` and check a seeded simulation within statistical tolerance ([§21.2, Eqs. 85–86](https://allenai.org/papers/olmocore3#page=123)). This is an assumption-based initialization exercise; it is not a universal variance identity for trained, correlated experts.

## Defer or use as discussion follow-ups

- **Global balancing across gradient-accumulation microbatches** is interesting but needs care: the report distinguishes online prefix counts from the exact full optimizer-batch objective, which would require delayed backward or a prepass ([§10.5, Eqs. 31–34](https://allenai.org/papers/olmocore3#page=73)). It could be a production follow-up after workout 3 rather than its own introductory pair.
- **Topology-agnostic checkpoint resharding** has a clean small-array core: stable global tensor key, shape, canonical element order, and chunk offsets; reconstruct shards under a new EP layout and check exact reconstruction ([§12.1–12.3](https://allenai.org/papers/olmocore3#page=98)). It is a good systems elective after the routing set.
- **MXFP8, NVSHMEM rowwise EP, device-scheduled grouped GEMM, and overlap** are tightly tied to particular hardware/backend contracts; a toy CPU implementation would not validate the report's throughput claims ([§§5, 8–9, 11, 15](https://allenai.org/papers/olmocore3#page=30)). Use these as measurement and design questions, with matched operand values and end-to-end timing controls from [§19.3](https://allenai.org/papers/olmocore3#page=118).
- [§22](https://allenai.org/papers/olmocore3#page=125) discusses how the stack *could* support SFT, DPO, GRPO, and PPO, but explicitly reports **no integrated MoE post-training run or quality result**. It should not be presented as evidence for a new PPO/GRPO algorithm workout from this report.

## Suggested implementation order

Build workouts 1–3 first as separate `dojo/<4-digit>_<name>.py` and `_eval.py` pairs. Keep the top-level `# %%` teaching flow and `# %% [answer]` section, with baseline, follow-up, and production evals. Each eval can define its own small route-counting helper; each problem should state its own API and test the requested/kept distinction explicitly. Workouts 4–6 can follow if the first three are engaging and distinct from 0028. This ordering and these interfaces are **authored dojo design**, not prescriptions in the Ai2 report.
