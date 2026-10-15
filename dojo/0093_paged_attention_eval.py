"""Reference and staged checks for attention over paged KV storage."""

# %%
import math
import torch
from functools import partial
from torch_judge import EvalCase, compare_values
from torch.utils._python_dispatch import TorchDispatchMode

# %%
def _validate_pages(query, key_blocks, value_blocks, block_table, context_length):
    if (not isinstance(query, torch.Tensor) or query.ndim != 2
            or not isinstance(key_blocks, torch.Tensor) or key_blocks.ndim != 4
            or not isinstance(value_blocks, torch.Tensor) or value_blocks.ndim != 4
            or not isinstance(block_table, torch.Tensor) or block_table.ndim != 1):
        raise ValueError('expected query [H,D], KV blocks [P,B,H,D/Dv], and table [L]')
    heads, width = query.shape
    pages, block_size, key_heads, key_width = key_blocks.shape
    if (heads == 0 or width == 0 or block_size == 0 or value_blocks.shape[:3] != (pages, block_size, heads)
            or key_heads != heads or key_width != width or value_blocks.shape[3] == 0):
        raise ValueError('incompatible or empty head/block dimensions')
    if (any(not tensor.is_floating_point() for tensor in (query, key_blocks, value_blocks))
            or len({tensor.dtype for tensor in (query, key_blocks, value_blocks)}) != 1
            or len({tensor.device for tensor in (query, key_blocks, value_blocks, block_table)}) != 1
            or block_table.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError('KV and query must share floating dtype/device; table must contain integers on that device')
    if (not isinstance(context_length, int) or isinstance(context_length, bool)
            or context_length < 0 or context_length > block_table.numel() * block_size):
        raise ValueError('context length exceeds the logical table capacity')
    for page_id in block_table:
        if int(page_id) < 0 or int(page_id) >= pages:
            raise ValueError('invalid physical page ID')
    if not torch.isfinite(query).all():
        raise ValueError('query must be finite')
    # Only referenced token values are validated by each complete algorithm.

# %%
# Reference 1: expose the logical sequence, then the attention equation.
# Each table entry names a physical page; repeats really repeat its tokens.
# Trim the final page before scoring. Max-subtracted framework softmax avoids
# exp overflow. O(C*H*(D+Dv)) work and context-sized gathered KV plus scores.
# Supports baseline/followup and numerical/gradient checks, but deliberately
# does not qualify for production's one-block forward-storage contract.
def paged_attention_dense(query, key_blocks, value_blocks, block_table, context_length):
    _validate_pages(query, key_blocks, value_blocks, block_table, context_length)
    if context_length == 0:
        return query.new_zeros((query.shape[0], value_blocks.shape[-1]))
    block_size = key_blocks.shape[1]
    keys, values = [], []
    for logical in range((context_length + block_size - 1) // block_size):
        page = int(block_table[logical])
        valid = min(block_size, context_length - logical * block_size)
        keys.append(key_blocks[page, :valid])
        values.append(value_blocks[page, :valid])
    keys, values = torch.cat(keys), torch.cat(values)
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError('referenced KV tokens must be finite')
    scores = torch.einsum('hd,thd->ht', query, keys) / math.sqrt(query.shape[1])
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum('ht,thv->hv', probabilities, values)

# %%
# Reference 2: online softmax, one logical block at a time.
# m is the largest score so far; l=sum(exp(score-m)); a=sum(exp(score-m)*v).
# When m grows, rescale the old l/a by exp(old_m-new_m) before adding a block.
# Same objective and gradients as dense; production includes validation.
# Work O(C*H*(D+Dv)), forward temporaries O(B*H*(D+Dv)+H*Dv).
# Autograd saves per-block intermediates: backward storage still grows with C.
# Uses input dtype; finite inputs whose dot products overflow need wider work
# or a stricter domain. This is portable PyTorch, not a fused serving kernel.
def paged_attention_streaming(query, key_blocks, value_blocks, block_table, context_length):
    _validate_pages(query, key_blocks, value_blocks, block_table, context_length)
    heads, width = query.shape
    block_size = key_blocks.shape[1]
    if context_length == 0:
        return query.new_zeros((heads, value_blocks.shape[-1]))
    maximum = query.new_full((heads, 1), -torch.inf)
    mass = query.new_zeros((heads, 1))
    weighted = query.new_zeros((heads, value_blocks.shape[-1]))
    for logical in range((context_length + block_size - 1) // block_size):
        page = int(block_table[logical])
        valid = min(block_size, context_length - logical * block_size)
        keys = key_blocks[page, :valid]
        values = value_blocks[page, :valid]
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise ValueError('referenced KV tokens must be finite')
        scores = torch.einsum('hd,thd->ht', query, keys) / math.sqrt(width)
        next_maximum = torch.maximum(maximum, scores.max(dim=1, keepdim=True).values)
        correction = torch.exp(maximum - next_maximum)
        weights = torch.exp(scores - next_maximum)
        weighted = weighted * correction + torch.einsum('ht,thv->hv', weights, values)
        mass = mass * correction + weights.sum(dim=1, keepdim=True)
        maximum = next_maximum
    return weighted / mass


# %%
# Opt in: compare a context-sized gather to bounded forward temporaries.
# These are ANALYTIC working-array estimates, not observed allocator peaks;
# they exclude physical input storage, validation, framework workspaces and
# autograd-retained state. Run inference_mode to measure the forward path.
# Timing includes Python/page-loop overhead, warmup and repeated synchronized
# wall-clock samples; no universal speedup is claimed for the streaming loop.
def explore_paged_storage(context_length=1024, block_size=32, heads=2,
                          key_width=16, value_width=16, dtype=torch.float64, device='cpu'):
    from torch_judge.measurement import measure
    pages = (context_length + block_size - 1) // block_size
    generator = torch.Generator(device=device).manual_seed(93)
    inputs = dict(
        query=torch.randn(heads, key_width, dtype=dtype, device=device, generator=generator),
        key_blocks=torch.randn(pages, block_size, heads, key_width, dtype=dtype, device=device, generator=generator),
        value_blocks=torch.randn(pages, block_size, heads, value_width, dtype=dtype, device=device, generator=generator),
        block_table=torch.arange(pages, device=device), context_length=context_length)
    itemsize = inputs['query'].element_size()
    workload = dict(context_length=context_length, block_size=block_size, heads=heads,
                    key_width=key_width, value_width=value_width, dtype=str(dtype))
    reports = {}
    with torch.inference_mode():
        expected = paged_attention_dense(**inputs)
        for name, implementation in [('dense', paged_attention_dense), ('streaming', paged_attention_streaming)]:
            torch.testing.assert_close(implementation(**inputs), expected)
            reports[name] = measure(lambda: implementation(**inputs), device=device, workload=workload)
    return dict(timing=reports, analytic_storage_bytes={
        'method': 'working-array estimate; not observed peak memory',
        'dense_gather_and_scores': itemsize * context_length * heads * (key_width + value_width + 2),
        'streaming_block_and_accumulators': itemsize * heads * (block_size * (key_width + value_width + 2) + 2 * value_width + 3),
        'autograd': 'not measured; saved per-block state can grow with context'})

# %%
paged_attention = paged_attention_streaming

# %% [verifiers]
def _dense(query, key_blocks, value_blocks, block_table, context_length):
    if context_length == 0:
        return query.new_zeros((query.shape[0], value_blocks.shape[-1]))
    keys = torch.cat([key_blocks[int(page)] for page in block_table], dim=0)[:context_length]
    values = torch.cat([value_blocks[int(page)] for page in block_table], dim=0)[:context_length]
    scores = torch.einsum('hd,thd->ht', query, keys) / math.sqrt(query.shape[1])
    return torch.einsum('ht,thv->hv', torch.softmax(scores, dim=-1), values)



def make_eval_data():
    generator = torch.Generator().manual_seed(93)
    q = torch.randn(2, 4, dtype=torch.float64, generator=generator)
    k = torch.randn(4, 3, 2, 4, dtype=torch.float64, generator=generator)
    v = torch.randn(4, 3, 2, 3, dtype=torch.float64, generator=generator)
    data = {}
    def add(name, stage, purpose, query, keys, values, table, length, error=None):
        inputs = dict(query=query, key_blocks=keys, value_blocks=values,
                      block_table=table, context_length=length)
        data[name] = EvalCase(name, stage, purpose, inputs,
            expected=None if error else _dense(**inputs).detach(), expected_exception=error,
            comparison=partial(compare_values, atol=1e-11, rtol=1e-11),
            generation={'seed': 93, 'block_size': 3, 'heads': 2, 'key_width': 4})
    add('two_full_pages', 'baseline', 'Trace two full logical blocks.', q, k, v, torch.tensor([0, 1]), 6)
    add('reordered_repeated_partial', 'followup', 'Physical page 3 appears twice; only one token of the last page is valid.', q, k, v, torch.tensor([3, 1, 3]), 7)
    add('empty_context', 'followup', 'An empty logical sequence returns zeros.', q, k, v, torch.empty(0, dtype=torch.long), 0)
    tail_k, tail_v = k.clone(), v.clone()
    tail_k[0, 1:] = float('nan'); tail_v[0, 1:] = float('inf')
    add('unused_nonfinite_tail', 'followup', 'Unused physical cells must never contribute.', q, tail_k, tail_v, torch.tensor([1, 0]), 4)
    add('gradient_pages', 'production', 'Repeated page gradients accumulate; page 2 stays unused.', q.clone().requires_grad_(), k.clone().requires_grad_(), v.clone().requires_grad_(), torch.tensor([3, 1, 3, 0]), 10)
    add('extreme_scores', 'production', 'A running maximum must rescale earlier blocks.', q * 100, k * 100, v, torch.tensor([3, 0, 1]), 8)
    add('invalid_page', 'production', 'Every table entry must name a physical page, even past context.', q, k, v, torch.tensor([0, 4]), 1, ValueError)
    bad = k.clone(); bad[1, 0, 0, 0] = float('nan')
    add('referenced_nonfinite', 'production', 'Reject nonfinite referenced keys.', q, bad, v, torch.tensor([1]), 1, ValueError)
    return data


def verify_followup_exposed_cases(paged_attention):
    for case in make_eval_data().values():
        if case.stage in ('baseline', 'followup'):
            torch.testing.assert_close(paged_attention(**case.inputs), case.expected, atol=1e-11, rtol=1e-11)


def verify_production_one_block_temporaries(paged_attention):
    # Operation-level tensor-size observation is a storage-contract check,
    # separate from output parity. It is not allocator peak-memory telemetry.
    # Physical input storage is excluded; newly returned operation tensors
    # must fit one block. This catches cat, stack, or context-sized indexing.
    case = make_eval_data()['two_full_pages']
    inputs = dict(case.inputs, block_table=torch.tensor([0, 1] * 32), context_length=192)
    limit = max(inputs['key_blocks'][0].numel(), inputs['value_blocks'][0].numel())
    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            outputs = result if isinstance(result, (list, tuple)) else (result,)
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    assert output.numel() <= limit, f'{func}: {output.numel()} elements exceeds one-block bound {limit}'
            return result
    # Table iteration returns unbind's many scalar views, also bounded.
    with Observe():
        actual = paged_attention(**inputs)
    torch.testing.assert_close(actual, _dense(**inputs), atol=1e-11, rtol=1e-11)

def verify_baseline_one_and_two_blocks(paged_attention):
    q = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    k = torch.tensor([[[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]],
                      [[[2., 0.], [0., 2.]], [[0., 2.], [2., 0.]]]], dtype=torch.float64)
    v = torch.tensor([[[[1.], [10.]], [[3.], [30.]]],
                      [[[5.], [50.]], [[7.], [70.]]]], dtype=torch.float64)
    for length, table in ((2, torch.tensor([0])), (4, torch.tensor([0, 1]))):
        got = paged_attention(q, k, v, table, length)
        assert got.shape == (2, 1)
        assert torch.allclose(got, _dense(q, k, v, table, length), atol=1e-12)


def verify_followup_shuffled_repeated_and_partial_pages(paged_attention):
    q = torch.tensor([[6., 0.], [0., 5.]], dtype=torch.float64)
    k = torch.tensor([[[[0., 1.], [1., 0.]], [[-1., 0.], [0., -1.]]],
                      [[[2., 0.], [0., 2.]], [[3., 0.], [0., 3.]]],
                      [[[1., 0.], [0., 1.]], [[4., 0.], [0., 4.]]]], dtype=torch.float64)
    v = torch.arange(12, dtype=torch.float64).reshape(3, 2, 2, 1)
    table = torch.tensor([2, 0, 2, 1])
    for length in (3, 5, 7):
        got = paged_attention(q, k, v, table, length)
        assert torch.allclose(got, _dense(q, k, v, table, length), atol=1e-12)


def verify_followup_adversarial_tail_and_empty(paged_attention):
    q = torch.tensor([[1., 0.]], dtype=torch.float64)
    k = torch.tensor([[[[0., 1.]], [[1000., 0.]], [[1000., 0.]]],
                      [[[1., 0.]], [[2., 0.]], [[3., 0.]]]], dtype=torch.float64)
    v = torch.tensor([[[[2.]], [[1e6]], [[1e6]]],
                      [[[3.]], [[5.]], [[7.]]]], dtype=torch.float64)
    table = torch.tensor([1, 0])
    assert torch.allclose(paged_attention(q, k, v, table, 4),
                          _dense(q, k, v, table, 4), atol=1e-12)
    zero = paged_attention(q, k, v, torch.empty(0, dtype=torch.long), 0)
    assert zero.shape == (1, 1) and torch.equal(zero, torch.zeros_like(zero))


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('invalid paged attention input accepted')


def verify_production_validation(paged_attention):
    q = torch.ones(2, 3)
    k = torch.ones(3, 2, 2, 3)
    v = torch.ones(3, 2, 2, 4)
    table = torch.tensor([2, 0])
    invalid = [
        (q[:, :0], k, v, table, 2),
        (q[0], k, v, table, 2),
        (q, k[:, :, :1], v, table, 2),
        (q, k, v[:, :, :, :0], table, 2),
        (q, k[:, :0], v[:, :0], table, 0),
        (q.double(), k, v, table, 2),
        (q, k, v.double(), table, 2),
        (q, k, v, table.float(), 2),
        (q, k, v, torch.tensor([-1, 0]), 2),
        (q, k, v, torch.tensor([0, 3]), 2),
        (q, k, v, table, 5),
        (q, k, v, table, -1),
        (q, k, v, table, True),
        (q * float('inf'), k, v, table, 2),
    ]
    for args in invalid:
        _reject(lambda args=args: paged_attention(*args))
    bad_k = k.clone()
    bad_k[2, 0, 0, 0] = float('nan')
    _reject(lambda: paged_attention(q, bad_k, v, table, 2))
    bad_v = v.clone()
    bad_v[2, 0, 0, 0] = float('inf')
    _reject(lambda: paged_attention(q, k, bad_v, table, 2))


def verify_production_dense_gradient_parity_and_no_mutation(paged_attention):
    case = make_eval_data()['gradient_pages']
    q, k, v, table = (case.inputs[name] for name in ('query', 'key_blocks', 'value_blocks', 'block_table'))
    original = [x.detach().clone() for x in (q, k, v, table)]
    got = paged_attention(q, k, v, table, 10)
    expected = _dense(q, k, v, table, 10)
    assert torch.allclose(got, expected, atol=1e-12)
    got.square().sum().backward(retain_graph=True)
    observed = [x.grad.clone() for x in (q, k, v)]
    for x in (q, k, v):
        x.grad = None
    expected.square().sum().backward()
    for x, grad in zip((q, k, v), observed):
        assert torch.allclose(x.grad, grad, atol=1e-11)
    assert torch.count_nonzero(k.grad[2]) == 0
    assert torch.count_nonzero(v.grad[2]) == 0
    for x, before in zip((q, k, v, table), original):
        assert torch.equal(x.detach(), before)


def verify_production_long_context_and_large_logits(paged_attention):
    torch.manual_seed(94)
    q = torch.randn(3, 5, dtype=torch.float64) * 100
    k = torch.randn(7, 8, 3, 5, dtype=torch.float64) * 100
    v = torch.randn(7, 8, 3, 2, dtype=torch.float64)
    table = torch.tensor([6, 2, 5, 1, 6, 0, 3, 4, 2, 1, 5, 0, 6, 4, 3])
    got = paged_attention(q, k, v, table, 117)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, _dense(q, k, v, table, 117), atol=1e-11)


# %%
from torch_judge import evaluate
if not evaluate('0093_paged_attention_eval.py', paged_attention):
    raise SystemExit(1)
