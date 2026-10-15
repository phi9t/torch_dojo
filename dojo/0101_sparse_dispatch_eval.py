"""Reference and staged checks for expert-contiguous local dispatch."""
# %%
import torch
from torch import nn
from torch_judge import EvalCase, compare, compare_values
from torch_judge.measurement import measure

# %%
# Shared contract checks, not routing algorithms. All three references enforce
# the production input/reply policy; tensor IDs are discrete, weights are not.
def _validate_dispatch(x, expert_ids, weights, experts):
    if (not isinstance(x, torch.Tensor) or not isinstance(expert_ids, torch.Tensor)
            or not isinstance(weights, torch.Tensor) or x.ndim != 2 or x.shape[1] == 0
            or expert_ids.ndim != 2 or expert_ids.shape[0] != x.shape[0]
            or expert_ids.shape[1] == 0 or expert_ids.dtype != torch.long
            or weights.shape != expert_ids.shape or not x.is_floating_point()
            or weights.dtype != x.dtype or weights.device != x.device
            or expert_ids.device != x.device or not torch.isfinite(x).all()
            or not torch.isfinite(weights).all() or not isinstance(experts, (nn.ModuleList, list, tuple))
            or len(experts) == 0 or any(not isinstance(e, nn.Module) for e in experts)
            or torch.any(expert_ids < 0) or torch.any(expert_ids >= len(experts))):
        raise ValueError("need compatible finite tensors and valid expert IDs")
    for expert in experts:
        for parameter in expert.parameters():
            if parameter.device != x.device or parameter.dtype != x.dtype:
                raise ValueError("expert parameter dtype/device mismatch")


def _validate_reply(reply, x, rows, width=None):
    if (not isinstance(reply, torch.Tensor) or reply.ndim != 2
            or reply.shape[0] != rows or reply.shape[1] == 0
            or reply.dtype != x.dtype or reply.device != x.device
            or not torch.isfinite(reply).all()
            or (width is not None and reply.shape[1] != width)):
        raise ValueError("expert reply must be finite Tensor [routes,F] with common F>0")
    return reply.shape[1]

# %%
# Version 1: route-by-route. Follow the stable permutation by hand; each
# reply contributes weights[t,k] * expert(x[t]). index_add sums duplicates.
# Supports production for deterministic pointwise experts. Calling each
# expert with batch=1 changes semantics for dropout/BatchNorm/batch coupling;
# no equivalence is claimed for those experts. O(R log R) sort, R expert calls,
# O(R*(D+F)) gathered/retained training storage, R=T*K. No dense all-expert pass.
def dispatch_traceable(x, expert_ids, weights, experts):
    _validate_dispatch(x, expert_ids, weights, experts)
    permutation = torch.argsort(expert_ids.reshape(-1), stable=True)
    if permutation.numel() == 0:
        reply = experts[0](x)
        _validate_reply(reply, x, 0)
        return reply, permutation
    tokens = torch.arange(len(x), device=x.device).repeat_interleave(expert_ids.shape[1])
    replies = []
    width = None
    for position in permutation:
        token = tokens[position]
        expert = experts[int(expert_ids.reshape(-1)[position])]
        reply = expert(x[token].unsqueeze(0))
        width = _validate_reply(reply, x, 1, width)
        replies.append(reply * weights.reshape(-1)[position])
    output = x.new_zeros((len(x), width))
    return output.index_add(0, tokens[permutation], torch.cat(replies)), permutation

# %%
# Version 2: batch routes for each selected expert. R calls become A calls,
# where A is the number of active experts. Batch shapes are meaningful:
# parity with version 1 assumes deterministic pointwise expert functions.
# Production contract; O(R log R + E*R) grouping via repeated ID scans,
# O(R*(D+F)) tensors plus each expert's activations. index_add preserves grads
# and repeated token contributions; GPU atomic order can introduce roundoff.
def dispatch_batched(x, expert_ids, weights, experts):
    _validate_dispatch(x, expert_ids, weights, experts)
    flat_ids = expert_ids.reshape(-1)
    permutation = torch.argsort(flat_ids, stable=True)
    if flat_ids.numel() == 0:
        reply = experts[0](x)
        _validate_reply(reply, x, 0)
        return reply, permutation
    tokens = torch.arange(len(x), device=x.device).repeat_interleave(expert_ids.shape[1])[permutation]
    sorted_ids = flat_ids[permutation]
    sorted_weights = weights.reshape(-1)[permutation]
    pieces = []
    width = None
    for expert_id, expert in enumerate(experts):
        positions = torch.where(sorted_ids == expert_id)[0]
        if positions.numel() == 0:
            continue
        reply = expert(x[tokens[positions]])
        width = _validate_reply(reply, x, len(positions), width)
        pieces.append(reply * sorted_weights[positions, None])
    output = x.new_zeros((len(x), width))
    return output.index_add(0, tokens, torch.cat(pieces)), permutation

# %%
# Version 3: reuse contiguous groups from the sorted route layout. One scan
# identifies active IDs/counts; packed x/weights are sliced at group offsets.
# Production contract, same deterministic pointwise scope. O(R log R + R + A)
# grouping instead of E repeated scans; O(R*(D+F)) working tensors and retained
# autograd activations. tolist() synchronizes CUDA metadata; this portable
# teaching implementation is not a fused GPU dispatcher or distributed kernel.
def dispatch_grouped(x, expert_ids, weights, experts):
    _validate_dispatch(x, expert_ids, weights, experts)
    permutation = torch.argsort(expert_ids.reshape(-1), stable=True)
    if permutation.numel() == 0:
        reply = experts[0](x)
        _validate_reply(reply, x, 0)
        return reply, permutation
    tokens = torch.arange(len(x), device=x.device).repeat_interleave(expert_ids.shape[1])[permutation]
    ids = expert_ids.reshape(-1)[permutation]
    packed_x = x[tokens]
    packed_weights = weights.reshape(-1)[permutation]
    active, counts = torch.unique_consecutive(ids, return_counts=True)
    pieces = []
    width = None
    offset = 0
    for expert_id, count in zip(active.tolist(), counts.tolist()):
        end = offset + count
        reply = experts[expert_id](packed_x[offset:end])
        width = _validate_reply(reply, x, count, width)
        pieces.append(reply * packed_weights[offset:end, None])
        offset = end
    output = x.new_zeros((len(x), width))
    return output.index_add(0, tokens, torch.cat(pieces)), permutation

# %%
sparse_dispatch = dispatch_grouped

# %% [verifiers]
def _linear_experts(dtype=torch.float64):
    # Initialization is overwritten; do not consume the learner's global RNG.
    with torch.random.fork_rng(devices=[]):
        experts = nn.ModuleList([nn.Linear(2, 2, bias=False, dtype=dtype) for _ in range(3)])
    with torch.no_grad():
        for index, expert in enumerate(experts):
            expert.weight.copy_(torch.eye(2, dtype=dtype) * (index + 1))
    return experts


def _compare_dispatch(actual, expected):
    return compare_values(actual, expected, atol=1e-11, rtol=1e-10)


def make_eval_data():
    """Fresh modules, leaf tensors and independent expectations every call.

    All ordinary experts are deterministic pointwise linear maps. Expected
    outputs use F.linear directly with parameters, then a token/slot sum;
    they do not call a reference dispatcher or alter expert call counters.
    """
    data = {}
    def add(name, stage, purpose, x, ids, weights, experts=None, reject=False):
        experts = _linear_experts() if experts is None else experts
        inputs = dict(x=x, expert_ids=ids, weights=weights, experts=experts)
        expected = None
        if not reject:
            with torch.no_grad():
                out = (_pointwise_oracle(x, ids, weights, experts) if len(x)
                       else x.new_empty((0, experts[0].out_features)))
                expected = out, torch.argsort(ids.flatten(), stable=True)
        data[name] = EvalCase(name, stage, purpose, inputs, expected,
                             expected_exception=ValueError if reject else None,
                             comparison=_compare_dispatch,
                             generation={'seed': 101, 'expert_semantics': 'deterministic pointwise'} if name == 'gradient_routes' else None)
    x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    add('stable_order', 'baseline', 'Stable expert groups: permutation [1,3,0,2].',
        x, torch.tensor([[2, 0], [2, 1]]), x.new_tensor([[.25, .75], [.5, .5]]))
    add('duplicate_routes', 'baseline', 'Token 0 requests expert 1 twice: contributions must add.',
        x.clone(), torch.tensor([[1, 1], [0, 1]]), x.new_ones((2, 2)))
    add('uneven_loads', 'followup', 'Expert 0 has five routes, expert 2 one; expert 1 is idle.',
        x.new_tensor([[1., 2.], [3., 4.], [-1., 2.]]),
        torch.tensor([[0, 0], [0, 2], [0, 0]]), x.new_ones((3, 2)))
    generator = torch.Generator().manual_seed(101)
    add('gradient_routes', 'followup', 'Backward through x, weights and selected expert parameters.',
        torch.randn(5, 2, dtype=x.dtype, generator=generator, requires_grad=True),
        torch.tensor([[2, 1], [0, 2], [2, 1], [0, 0], [2, 2]]),
        torch.randn(5, 2, dtype=x.dtype, generator=generator, requires_grad=True))
    with torch.random.fork_rng(devices=[]):
        shared = nn.Linear(2, 2, bias=False, dtype=x.dtype)
    with torch.no_grad(): shared.weight.copy_(torch.eye(2, dtype=x.dtype) * 2)
    add('shared_expert', 'followup', 'Two IDs intentionally alias one expert; replay must preserve that alias.',
        x.clone().requires_grad_(), torch.tensor([[0, 1], [1, 0]]),
        x.new_ones((2, 2)).requires_grad_(), nn.ModuleList([shared, shared]))
    add('unselected_expert', 'production', 'An unselected module raises if invoked; sparse execution keeps it idle.',
        x.clone(), torch.zeros((2, 1), dtype=torch.long), x.new_ones((2, 1)),
        nn.ModuleList([_linear_experts()[0], _UnselectedMustStayIdle()]))
    add('empty_batch', 'production', 'Expert 0 supplies F via an empty reply; no route contributions.',
        x[:0].clone(), torch.empty(0, 2, dtype=torch.long), x.new_empty((0, 2)))
    for name, expert in [('nonfinite_reply', _BadReply(2, float('nan'))),
                         ('malformed_reply', _MalformedReply(None)),
                         ('zero_width_reply', _BadReply(0, 0.))]:
        add(name, 'production', 'Selected expert must return finite Tensor [routes,F], F>0.',
            x.clone(), torch.zeros((2, 1), dtype=torch.long), x.new_ones((2, 1)), [expert], reject=True)
    return data


def _verify_cases(dispatch, names):
    for name in names:
        case = make_eval_data()[name]
        try:
            actual = dispatch(**case.inputs)
        except Exception as error:
            result = compare(case, exception=error)
        else:
            result = compare(case, actual)
        assert result.passed, (name, result.diagnostics)


def verify_baseline_exploration_examples(sparse_dispatch):
    _verify_cases(sparse_dispatch, ['stable_order', 'duplicate_routes'])


def verify_followup_exploration_examples(sparse_dispatch):
    _verify_cases(sparse_dispatch, ['uneven_loads', 'gradient_routes', 'shared_expert'])


def verify_production_exploration_examples(sparse_dispatch):
    _verify_cases(sparse_dispatch, ['empty_batch', 'unselected_expert', 'nonfinite_reply', 'malformed_reply', 'zero_width_reply'])


def verify_baseline_stable_group_and_known_output(sparse_dispatch):
    case = make_eval_data()['stable_order']
    x = case.inputs['x']
    out, perm = sparse_dispatch(**case.inputs)
    assert torch.equal(perm, torch.tensor([1, 3, 0, 2]))
    assert torch.allclose(out, torch.tensor([[1.5, 3.], [7.5, 10.]], dtype=x.dtype))


def verify_baseline_ties_keep_token_slot_order(sparse_dispatch):
    x = torch.ones((3, 2), dtype=torch.float64)
    ids = torch.tensor([[1, 1], [0, 1], [0, 1]])
    out, perm = sparse_dispatch(x, ids, torch.ones(ids.shape, dtype=x.dtype), _linear_experts())
    assert torch.equal(perm, torch.tensor([2, 4, 0, 1, 3, 5]))
    assert torch.allclose(out[:, 0], torch.tensor([4., 3., 3.], dtype=x.dtype))


def _pointwise_oracle(x, ids, weights, experts):
    # Direct framework affine operation, independent of dispatch/grouping.
    rows = []
    for token in range(len(x)):
        contributions = []
        for slot in range(ids.shape[1]):
            expert = experts[int(ids[token, slot])]
            value = torch.nn.functional.linear(x[token], expert.weight, expert.bias)
            contributions.append(weights[token, slot] * value)
        rows.append(torch.stack(contributions).sum(0))
    return torch.stack(rows)


def verify_followup_dense_forward_and_backward(sparse_dispatch):
    case = make_eval_data()['gradient_routes']
    x, ids, weights, experts = (case.inputs[key] for key in ('x', 'expert_ids', 'weights', 'experts'))
    got, _ = sparse_dispatch(**case.inputs)
    expected = _pointwise_oracle(x, ids, weights, experts)
    assert torch.allclose(got, expected, atol=1e-12)
    variables = (x, weights, *experts.parameters())
    # Compare vector-Jacobian products for every input/weight/selected parameter.
    probe = torch.arange(got.numel(), dtype=got.dtype).reshape_as(got) / 7
    actual_grads = torch.autograd.grad((got * probe).sum(), variables, allow_unused=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), variables, allow_unused=True)
    for actual, target in zip(actual_grads, expected_grads):
        assert actual is not None and target is not None, 'selected training variable lost its gradient'
        assert torch.allclose(actual, target, atol=1e-11)


def verify_followup_sparse_experts_receive_no_gradient(sparse_dispatch):
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    experts = _linear_experts()
    out, _ = sparse_dispatch(x, torch.tensor([[0], [0]]),
                             torch.ones(2, 1, dtype=x.dtype), experts)
    out.sum().backward()
    assert experts[0].weight.grad is not None
    assert all(expert.weight.grad is None for expert in experts[1:])


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid dispatch input accepted")


def verify_production_empty_and_immutability(sparse_dispatch):
    experts = _linear_experts()
    x = torch.empty(0, 2, dtype=torch.float64)
    ids = torch.empty(0, 2, dtype=torch.long)
    weights = torch.empty(0, 2, dtype=x.dtype)
    out, perm = sparse_dispatch(x, ids, weights, experts)
    assert out.shape == (0, 2) and perm.shape == (0,) and perm.dtype == torch.long
    x = torch.randn(3, 2, dtype=torch.float64)
    ids = torch.tensor([[2], [0], [2]])
    weights = torch.randn(3, 1, dtype=x.dtype)
    before = [a.clone() for a in (x, ids, weights)]
    sparse_dispatch(x, ids, weights, experts)
    assert all(torch.equal(a, b) for a, b in zip((x, ids, weights), before))


def verify_production_validation(sparse_dispatch):
    x = torch.ones(2, 2, dtype=torch.float64)
    ids = torch.tensor([[0], [1]])
    weights = torch.ones(2, 1, dtype=x.dtype)
    experts = _linear_experts()
    bad = [
        (x[0], ids, weights, experts),
        (x, ids[:, :0], weights[:, :0], experts),
        (x, ids[0], weights, experts),
        (x, ids.float(), weights, experts),
        (x, torch.tensor([[0], [3]]), weights, experts),
        (x, ids, weights.float(), experts),
        (x, ids, weights[:, :0], experts),
        (x, ids, weights, []),
        (torch.full_like(x, float("nan")), ids, weights, experts),
        (x, ids, torch.full_like(weights, float("inf")), experts),
    ]
    for args in bad:
        _reject(lambda args=args: sparse_dispatch(*args))


class _BadReply(nn.Module):
    def __init__(self, width, value):
        super().__init__()
        self.width = width
        self.value = value

    def forward(self, x):
        return x.new_full((len(x), self.width), self.value)


class _UnselectedMustStayIdle(nn.Module):
    def forward(self, x):
        raise AssertionError("unselected expert was invoked")


class _MalformedReply(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return self.value


def verify_production_selected_replies_finite_and_positive_width(sparse_dispatch):
    x = torch.ones(2, 2, dtype=torch.float64)
    ids = torch.zeros((2, 1), dtype=torch.long)
    weights = torch.ones((2, 1), dtype=x.dtype)
    for reply in (_BadReply(2, float("nan")), _BadReply(2, float("inf")),
                  _BadReply(0, 0.)):
        _reject(lambda reply=reply: sparse_dispatch(x, ids, weights, [reply]))
    _reject(lambda: sparse_dispatch(x[:0], ids[:0], weights[:0], [_BadReply(0, 0.)]))
    out, _ = sparse_dispatch(x, ids, weights,
                             [_BadReply(2, 1.), _UnselectedMustStayIdle()])
    assert torch.equal(out, torch.ones_like(x))


def verify_production_rejects_non_tensor_selected_reply(sparse_dispatch):
    x = torch.ones(2, 2, dtype=torch.float64)
    ids = torch.zeros((2, 1), dtype=torch.long)
    weights = torch.ones((2, 1), dtype=x.dtype)
    for invalid in (None, (x,), 2):
        _reject(lambda invalid=invalid: sparse_dispatch(
            x, ids, weights, [_MalformedReply(invalid)]))


def verify_production_empty_reply_contract(sparse_dispatch):
    x = torch.empty(0, 2, dtype=torch.float64)
    ids = torch.empty((0, 1), dtype=torch.long)
    weights = torch.empty((0, 1), dtype=x.dtype)
    invalid = (None, (x,), 2, torch.empty(0, 2, dtype=torch.float32),
               torch.empty(0, 2, dtype=x.dtype, device="meta"),
               torch.empty(0, 0, dtype=x.dtype))
    for reply in invalid:
        _reject(lambda reply=reply: sparse_dispatch(
            x, ids, weights, [_MalformedReply(reply)]))


# %%
# Opt in after correctness checks. Invocation counts expose useful work even
# when timings for tiny CPU workloads are noisy. No GPU/distributed claim.
def explore_dispatch_work(tokens=128, slots=2, expert_count=8, width=16,
                          device='cpu', repeats=5):
    if tokens < 1 or slots < 1 or expert_count < 1 or width < 1:
        raise ValueError('positive workload dimensions required')
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(101)
    x = torch.randn(tokens, width, generator=generator, device=device)
    ids = torch.randint(expert_count, (tokens, slots), generator=generator, device=device)
    weights = torch.full((tokens, slots), 1 / slots, device=device)
    class CountedLinear(nn.Linear):
        def __init__(self):
            super().__init__(width, width, bias=False, device=device)
            self.calls = 0
            with torch.no_grad(): self.weight.copy_(torch.eye(width, device=device))
        def forward(self, x):
            self.calls += 1
            return super().forward(x)
    # All weights are overwritten; preserve the learner's random state.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        experts = nn.ModuleList([CountedLinear() for _ in range(expert_count)])
    report = {}
    with torch.no_grad():
        for implementation in (dispatch_traceable, dispatch_batched, dispatch_grouped):
            for expert in experts: expert.calls = 0
            output, permutation = implementation(x, ids, weights, experts)
            assert torch.allclose(output, x, atol=1e-6, rtol=1e-5)
            assert torch.equal(permutation, torch.argsort(ids.flatten(), stable=True))
            calls = sum(expert.calls for expert in experts)
            report[implementation.__name__] = measure(
                lambda: implementation(x, ids, weights, experts), device=device,
                workload=dict(tokens=tokens, slots=slots, experts=expert_count,
                              width=width, dtype=str(x.dtype), seed=101), repeats=repeats)
            report[implementation.__name__].update(
                expert_calls_per_dispatch=calls,
                grouping_work=('one active-group scan after sort' if implementation is dispatch_grouped
                               else 'E scans over R sorted IDs' if implementation is dispatch_batched
                               else 'one route at a time after sort'))
    return report

# %%
from torch_judge import evaluate
if not evaluate("0101_sparse_dispatch_eval.py", sparse_dispatch):
    raise SystemExit(1)
