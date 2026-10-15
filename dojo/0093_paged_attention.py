# %%
# 0093 · Attend over a paged KV cache
# Original dojo exercise prompted by https://www.deep-ml.com/problems/492
# See 0005_attention.py for dense attention and 0014_kv_cache.py for cached KV.

# %%
import math
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# A logical sequence reads physical pages [2, 0]. What happens if you
# accidentally read pages [0, 2]? Predict which value dominates when the
# query matches a key on page 2, then inspect this tiny page table.
toy_table = torch.tensor([2, 0])
print('logical block 0 reads physical page', toy_table[0].item())

# %%
# Inputs: query [H,D], key_blocks [P,B,H,D], value_blocks [P,B,H,Dv],
# integer block_table [L], and integer context_length C. H, D, B, Dv
# are positive; P may be zero only for an empty table. One query per head
# attends independently to the first C logical tokens. Logical block i
# reads physical block block_table[i], and can repeat another block's page.
# Scale scores by sqrt(D), softmax across the C tokens, and return [H,Dv].
# C=0 returns zeros. The last logical block may use fewer than B tokens;
# unused cells must have no effect. Every table entry must be in [0,P),
# including entries beyond the current context. Reject invalid shapes,
# index/capacity bounds, dtype/device, and nonfinite used values with
# ValueError. Floating query/KV tensors share dtype and device; table is
# integer on that device. Do not mutate inputs. Gradients must flow to
# query and the referenced KV entries.

# %% [answer]
def paged_attention(query: torch.Tensor, key_blocks: torch.Tensor,
                    value_blocks: torch.Tensor, block_table: torch.Tensor,
                    context_length: int) -> torch.Tensor:
    """Return independent per-head attention over logical paged tokens."""
    pass

# %%
# Baseline: one or two full pages match dense attention for each head.
check('0093_paged_attention_eval.py', stage='baseline')

# %%
# Follow-up: reorder and repeat physical pages. Mask the unused tail of
# a partial final block, even if its cells contain very large scores.
# An empty context has no weighted values and returns [H,Dv] zeros.
check('0093_paged_attention_eval.py', stage='followup')

# %%
# Production: validate the stated contract and use a running softmax
# maximum, mass, and weighted numerator while visiting one block at a time.
# Do not gather or concatenate the whole logical KV sequence. Forward
# temporary storage scales with one block and the output, not C. PyTorch
# autograd retains intermediates for backward, so training memory can
# still grow with context length; a bounded backward needs recomputation.
check('0093_paged_attention_eval.py', stage='production')

# %%
# Your workbench: inspect physical pages and their logical order before coding.
# Every retrieval builds fresh data; reference definitions never enter this
# kernel through get_eval_data. Baseline/followup/production filtering is cumulative.
eval_data = get_eval_data('0093_paged_attention_eval.py')
case = eval_data['reordered_repeated_partial']
case.inputs, case.expected, case.purpose

# %%
# Predict the final valid physical token from table and context_length.
# Run your current answer directly, then keep the record for inspection.
# If you changed fixture inputs, rebuild the case rather than using a stale
# oracle. Expected-rejection cases can be explored with replay below.
actual = paged_attention(**case.inputs)
result = compare(case, actual)
result.passed, result.diagnostics, result.actual, result.expected

# %%
# Try unused_nonfinite_tail when debugging a tail mask. Replay catches errors
# as inspectable results, and starts with fresh tensors after every revision.
results = replay('0093_paged_attention_eval.py', paged_attention,
                 names=['reordered_repeated_partial', 'unused_nonfinite_tail', 'invalid_page'])
failures = {name: result for name, result in results.items() if not result.passed}
failures

# %%
# After solving: open the companion eval script and run each complete reference
# cell. Dense qualifies through followup; streaming qualifies production.
# Why can dense be perfectly correct yet fail the production storage check?
# Trace (maximum, mass, weighted numerator) across two blocks by hand, then
# opt into explore_paged_storage() to compare estimates and measured timings.
# A forward storage bound says nothing about autograd-retained backward state.
