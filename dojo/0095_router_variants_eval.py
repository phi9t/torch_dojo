"""Reference and staged verifiers for MoE router weight contracts."""
# %%
import math
import torch
from torch.nn import functional as F

# %%
def route_variants(logits, expert_outputs, k, mode='selected_softmax', bias=None, scale=1.0):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or not isinstance(expert_outputs, torch.Tensor) or expert_outputs.ndim != 3):
        raise ValueError('need logits [T,E] and expert_outputs [T,E,D]')
    tokens, experts = logits.shape
    if (experts < 1 or expert_outputs.shape[:2] != (tokens, experts)
            or expert_outputs.shape[2] < 1 or not logits.is_floating_point()
            or not expert_outputs.is_floating_point()
            or logits.dtype != expert_outputs.dtype or logits.device != expert_outputs.device
            or not torch.isfinite(logits).all() or not torch.isfinite(expert_outputs).all()):
        raise ValueError('finite, compatible floating tensors required')
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= experts:
        raise ValueError('k must be an integer in [1,E]')
    if mode not in ('selected_softmax', 'full_truncate', 'full_renorm', 'sigmoid_bias'):
        raise ValueError('unknown router mode')
    try:
        route_scale = float(scale)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('scale must be finite') from None
    if not math.isfinite(route_scale):
        raise ValueError('scale must be finite')
    if mode == 'sigmoid_bias':
        if (bias is None or not isinstance(bias, torch.Tensor) or bias.shape != (experts,)
                or bias.dtype != logits.dtype or bias.device != logits.device
                or not torch.isfinite(bias).all() or route_scale <= 0):
            raise ValueError('sigmoid mode needs a finite bias [E] and positive scale')
        scores = logits.sigmoid()
        rank_scores = scores + bias
    else:
        if bias is not None or route_scale != 1:
            raise ValueError('bias and scale only apply to sigmoid mode')
        scores = logits if mode == 'selected_softmax' else logits.softmax(-1)
        rank_scores = scores
    ids = torch.argsort(rank_scores, dim=-1, descending=True, stable=True)[:, :k]
    if mode == 'selected_softmax':
        weights = logits.gather(1, ids).softmax(-1)
    elif mode == 'full_truncate':
        weights = scores.gather(1, ids)
    elif mode == 'full_renorm':
        selected = scores.gather(1, ids)
        weights = selected / selected.sum(-1, keepdim=True)
    else:
        # log-sigmoid keeps relative affinity when every sigmoid underflows.
        weights = F.logsigmoid(logits.gather(1, ids)).softmax(-1) * route_scale
    values = expert_outputs.gather(1, ids.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[2]))
    output = (values * weights.unsqueeze(-1)).sum(dim=1)
    return output, ids, weights, scores


# %% [verifiers]
def verify_baseline_selected_weights_and_output(route_variants):
    logits = torch.tensor([[2., 1., 0.], [0., 3., 2.]], dtype=torch.float64)
    values = torch.tensor([[[1.], [4.], [100.]], [[10.], [2.], [6.]]], dtype=torch.float64)
    output, ids, weights, scores = route_variants(logits, values, 2)
    assert ids.tolist() == [[0, 1], [1, 2]]
    expected_weights = torch.tensor([[math.e/(math.e+1), 1/(math.e+1)]] * 2,
                                    dtype=torch.float64)
    assert torch.allclose(weights, expected_weights, atol=1e-12)
    assert torch.allclose(output, (values.gather(1, ids[..., None]) * weights[..., None]).sum(1), atol=1e-12)
    assert torch.equal(scores, logits)


def verify_baseline_one_route_has_unit_weight(route_variants):
    logits = torch.tensor([[9., -9., 0.]], dtype=torch.float64)
    values = torch.tensor([[[2., 3.], [50., 60.], [70., 80.]]], dtype=torch.float64)
    output, ids, weights, _ = route_variants(logits, values, 1)
    assert ids.tolist() == [[0]] and torch.equal(weights, torch.ones_like(weights))
    assert torch.equal(output, values[:, 0, :])


def verify_followup_truncation_does_not_renormalize(route_variants):
    logits = torch.tensor([[2., 1., 0.]], dtype=torch.float64)
    values = torch.tensor([[[1.], [3.], [100.]]], dtype=torch.float64)
    full = logits.softmax(-1)
    truncated = route_variants(logits, values, 2, 'full_truncate')
    renorm = route_variants(logits, values, 2, 'full_renorm')
    assert truncated[1].tolist() == renorm[1].tolist() == [[0, 1]]
    assert torch.allclose(truncated[2], full[:, :2], atol=1e-12)
    assert torch.allclose(renorm[2], full[:, :2] / full[:, :2].sum(-1, keepdim=True), atol=1e-12)
    assert truncated[2].sum() < 1 and torch.allclose(renorm[2].sum(), torch.tensor(1., dtype=torch.float64))
    assert torch.equal(truncated[3], full) and torch.equal(renorm[3], full)
    assert truncated[0].item() < renorm[0].item()


def verify_followup_unselected_logit_changes_truncated_mass(route_variants):
    values = torch.tensor([[[2.], [4.], [9.]]], dtype=torch.float64)
    low = torch.tensor([[2., 1., -10.]], dtype=torch.float64)
    high = torch.tensor([[2., 1., 0.]], dtype=torch.float64)
    first = route_variants(low, values, 2, 'full_truncate')
    second = route_variants(high, values, 2, 'full_truncate')
    assert first[1].tolist() == second[1].tolist() == [[0, 1]]
    assert first[0].item() > second[0].item()
    for logits in (low, high):
        selected = route_variants(logits, values, 2, 'selected_softmax')
        renorm = route_variants(logits, values, 2, 'full_renorm')
        assert torch.allclose(selected[2], renorm[2], atol=1e-12)


def verify_production_bias_only_selects(route_variants):
    logits = torch.tensor([[0., 1., 2.]], dtype=torch.float64)
    values = torch.tensor([[[10.], [20.], [30.]]], dtype=torch.float64)
    bias = torch.tensor([4., 0., 0.], dtype=torch.float64)
    output, ids, weights, scores = route_variants(logits, values, 2, 'sigmoid_bias', bias, 2.5)
    assert ids.tolist() == [[0, 2]]
    affinity = logits.sigmoid().gather(1, ids)
    expected = affinity / affinity.sum(-1, keepdim=True) * 2.5
    assert torch.allclose(weights, expected, atol=1e-12)
    assert torch.allclose(output, (values.gather(1, ids[..., None]) * expected[..., None]).sum(1), atol=1e-12)
    assert torch.allclose(scores, logits.sigmoid(), atol=1e-12)
    changed_bias = torch.tensor([5., 0., 0.], dtype=torch.float64)
    assert torch.allclose(route_variants(logits, values, 2, 'sigmoid_bias', changed_bias, 2.5)[2], weights)


def verify_production_ties_extreme_and_empty(route_variants):
    logits = torch.zeros(2, 4, dtype=torch.float64)
    values = torch.arange(24., dtype=torch.float64).reshape(2, 4, 3)
    for mode in ('selected_softmax', 'full_truncate', 'full_renorm', 'sigmoid_bias'):
        bias = torch.zeros(4, dtype=torch.float64) if mode == 'sigmoid_bias' else None
        _, ids, _, _ = route_variants(logits, values, 2, mode, bias)
        assert ids.tolist() == [[0, 1], [0, 1]]
        empty = route_variants(logits[:0], values[:0], 2, mode, bias)
        assert [tensor.shape for tensor in empty] == [(0, 3), (0, 2), (0, 2), (0, 4)]
    extreme = torch.tensor([[1000., 999., -1000.]], dtype=torch.float64)
    vals = torch.ones(1, 3, 1, dtype=torch.float64)
    for mode in ('selected_softmax', 'full_truncate', 'full_renorm', 'sigmoid_bias'):
        bias = torch.zeros(3, dtype=torch.float64) if mode == 'sigmoid_bias' else None
        result = route_variants(extreme, vals, 2, mode, bias)
        assert all(torch.isfinite(tensor).all() for tensor in result if tensor.is_floating_point())


def verify_production_gradients_and_strided_inputs(route_variants):
    torch.manual_seed(95)
    logits = torch.tensor([[0.3, 1.5, -0.8]], dtype=torch.float64, requires_grad=True)
    values = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    for mode in ('selected_softmax', 'full_truncate', 'full_renorm', 'sigmoid_bias'):
        logits.grad = values.grad = None
        out, ids, weights, _ = route_variants(logits, values, 2, mode, bias if mode == 'sigmoid_bias' else None)
        out.sum().backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        assert values.grad is not None and torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad[:, ids[0, 0]]) > 0
        if mode == 'full_truncate':
            assert logits.grad[0, 2].abs() > 1e-8
        else:
            assert logits.grad[0, 2].abs() < 1e-12
        def objective(z):
            return route_variants(z, values.detach(), 2, mode, bias if mode == 'sigmoid_bias' else None)[0].sum()
        step = torch.zeros_like(logits)
        step[0, 0] = 1e-6
        finite_difference = ((objective(logits.detach()+step)-objective(logits.detach()-step)) / 2e-6).item()
        assert math.isclose(logits.grad[0, 0].item(), finite_difference, rel_tol=1e-5, abs_tol=1e-7)
    strided_logits = torch.randn(4, 6, dtype=torch.float64)[:, ::2]
    strided_values = torch.randn(4, 3, 4, dtype=torch.float64)[:, :, ::2]
    assert not strided_logits.is_contiguous() and not strided_values.is_contiguous()
    assert route_variants(strided_logits, strided_values, 2)[0].shape == (4, 2)


def verify_production_rejects_invalid_contract(route_variants):
    logits = torch.ones(2, 3)
    values = torch.ones(2, 3, 4)
    bad = [(logits[:, :0], values[:, :0], 1), (logits, values[:, :, :0], 1),
           (logits, values[:1], 1), (logits, values.double(), 1),
           (logits, values, 0), (logits, values, 4), (logits, values, True),
           (logits, values, 1, 'unknown'),
           (logits, values, 1, 'full_truncate', torch.zeros(3)),
           (logits, values, 1, 'selected_softmax', None, 2.),
           (logits, values, 1, 'sigmoid_bias', torch.zeros(2)),
           (logits, values, 1, 'sigmoid_bias', torch.zeros(3), 0.),
           (logits, values, 1, 'sigmoid_bias', torch.zeros(3), float('nan')),
           (logits * float('nan'), values, 1),
           (logits, values * float('inf'), 1)]
    for args in bad:
        try:
            route_variants(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid router case: {len(args)} arguments')
    originals = [logits.clone(), values.clone()]
    route_variants(logits, values, 2)
    assert torch.equal(logits, originals[0]) and torch.equal(values, originals[1])


# %%
from torch_judge import evaluate
if not evaluate('0095_router_variants_eval.py', route_variants):
    raise SystemExit(1)
