"""Reference and staged verifiers for absent, ungated, and gated shared experts."""
# %%
import torch
from torch import nn

# %%
class SharedBranch(nn.Module):
    def __init__(self, mode, shared_expert=None, gate=None):
        super().__init__()
        if mode not in ('absent', 'ungated', 'gated'):
            raise ValueError('unknown shared branch mode')
        if (mode == 'absent' and (shared_expert is not None or gate is not None)
                or mode == 'ungated' and (not isinstance(shared_expert, nn.Module) or gate is not None)
                or mode == 'gated' and (not isinstance(shared_expert, nn.Module)
                                        or not isinstance(gate, nn.Module))):
            raise ValueError('shared expert and gate modules do not match mode')
        self.mode = mode
        self.shared_expert = shared_expert
        self.gate = gate

    def forward(self, x, routed):
        if (not isinstance(x, torch.Tensor) or not isinstance(routed, torch.Tensor)
                or x.ndim < 2 or x.shape[-1] == 0 or x.shape != routed.shape
                or not x.is_floating_point() or not routed.is_floating_point()
                or x.dtype != routed.dtype or x.device != routed.device
                or not torch.isfinite(x).all() or not torch.isfinite(routed).all()):
            raise ValueError('finite, compatible floating tensors [...,D] required')
        if self.mode == 'absent':
            return routed
        try:
            shared = self.shared_expert(x)
            gate_logit = self.gate(x) if self.mode == 'gated' else None
        except (RuntimeError, TypeError) as exc:
            raise ValueError('shared expert or gate rejected input') from exc
        if (not isinstance(shared, torch.Tensor) or shared.shape != x.shape
                or shared.dtype != x.dtype or shared.device != x.device
                or not torch.isfinite(shared).all()):
            raise ValueError('shared expert must return finite [...,D] on input dtype/device')
        if self.mode == 'ungated':
            return routed + shared
        if (not isinstance(gate_logit, torch.Tensor)
                or gate_logit.shape != x.shape[:-1] + (1,)
                or gate_logit.dtype != x.dtype or gate_logit.device != x.device
                or not torch.isfinite(gate_logit).all()):
            raise ValueError('gate must return finite [...,1] on input dtype/device')
        return routed + gate_logit.sigmoid() * shared


# %% [verifiers]
def _identity_linear(width=2):
    layer = nn.Linear(width, width, bias=False).double()
    with torch.no_grad():
        layer.weight.copy_(torch.eye(width, dtype=torch.float64))
    return layer


def verify_baseline_absent_and_ungated(SharedBranch):
    x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    routed = torch.tensor([[10., 20.], [30., 40.]], dtype=torch.float64)
    assert torch.equal(SharedBranch('absent')(x, routed), routed)
    shared = _identity_linear()
    layer = SharedBranch('ungated', shared)
    assert torch.equal(layer(x, routed), routed + x)
    assert torch.equal(layer(x, torch.zeros_like(x)), x)


def verify_baseline_shared_module_registered(SharedBranch):
    shared = nn.Linear(3, 3, bias=True).double()
    layer = SharedBranch('ungated', shared)
    assert layer.shared_expert is shared
    assert set(layer.state_dict()) == {'shared_expert.weight', 'shared_expert.bias'}


def verify_followup_gate_uses_logits_per_token(SharedBranch):
    shared = _identity_linear()
    gate = nn.Linear(2, 1).double()
    with torch.no_grad():
        gate.weight.copy_(torch.tensor([[2., 0.]], dtype=torch.float64))
        gate.bias.zero_()
    layer = SharedBranch('gated', shared, gate)
    x = torch.tensor([[1., 2.], [-1., 2.]], dtype=torch.float64)
    routed = torch.tensor([[3., 4.], [3., 4.]], dtype=torch.float64)
    expected = routed + (x[:, :1] * 2).sigmoid() * x
    assert torch.allclose(layer(x, routed), expected, atol=1e-12)
    assert set(layer.state_dict()) == {'shared_expert.weight', 'gate.weight', 'gate.bias'}


def verify_followup_shared_not_normalized_with_routed(SharedBranch):
    x = torch.ones(3, 2, dtype=torch.float64)
    routed = torch.full_like(x, 2.)
    shared = _identity_linear()
    ungated = SharedBranch('ungated', shared)
    assert torch.equal(ungated(x, routed), torch.full_like(x, 3.))
    gate = nn.Linear(2, 1).double()
    with torch.no_grad():
        gate.weight.zero_(); gate.bias.zero_()
    gated = SharedBranch('gated', _identity_linear(), gate)
    assert torch.equal(gated(x, routed), torch.full_like(x, 2.5))


def verify_production_gradients_reach_every_branch(SharedBranch):
    torch.manual_seed(96)
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    routed = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    shared = nn.Linear(4, 4).double()
    gate = nn.Linear(4, 1).double()
    layer = SharedBranch('gated', shared, gate)
    output = layer(x, routed)
    assert output.shape == x.shape
    output.square().sum().backward()
    for gradient in (x.grad, routed.grad, shared.weight.grad, shared.bias.grad,
                     gate.weight.grad, gate.bias.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    assert torch.count_nonzero(x.grad.reshape(-1, 4), dim=1).min() > 0


def verify_production_strided_empty_and_no_mutation(SharedBranch):
    x = torch.arange(24., dtype=torch.float64).reshape(2, 3, 4).transpose(0, 1)
    routed = (x * 2).detach()
    assert not x.is_contiguous()
    before = x.clone(), routed.clone()
    layer = SharedBranch('ungated', _identity_linear(4))
    assert torch.allclose(layer(x, routed), x + routed)
    assert torch.equal(x, before[0]) and torch.equal(routed, before[1])
    gate = nn.Linear(4, 1).double()
    for mode, expert, gate_module in (('absent', None, None),
                                      ('ungated', _identity_linear(4), None),
                                      ('gated', _identity_linear(4), gate)):
        result = SharedBranch(mode, expert, gate_module)(x[:0], routed[:0])
        assert result.shape == (0, 2, 4)


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('invalid shared branch input accepted')


def verify_production_constructor_and_input_validation(SharedBranch):
    expert = _identity_linear()
    gate = nn.Linear(2, 1).double()
    for args in (('unknown',), ('absent', expert), ('absent', None, gate),
                 ('ungated',), ('ungated', expert, gate), ('gated', expert),
                 ('gated', None, gate), ('gated', 'not a module', gate),
                 ('gated', expert, 'not a module')):
        _reject(lambda args=args: SharedBranch(*args))
    layer = SharedBranch('ungated', expert)
    x = torch.ones(2, 2, dtype=torch.float64)
    routed = torch.zeros_like(x)
    bad = [(x[0], routed[0]), (x[:, :0], routed[:, :0]), (x, routed[:1]),
           (x, routed.float()), (x * float('nan'), routed),
           (x, routed * float('inf'))]
    for inputs in bad:
        _reject(lambda inputs=inputs: layer(*inputs))


def verify_production_module_output_validation(SharedBranch):
    x = torch.ones(2, 2, dtype=torch.float64)
    routed = torch.zeros_like(x)
    _reject(lambda: SharedBranch('ungated', nn.Linear(2, 3).double())(x, routed))
    _reject(lambda: SharedBranch('ungated', nn.Linear(2, 2).float())(x, routed))
    _reject(lambda: SharedBranch('gated', _identity_linear(), nn.Linear(2, 2).double())(x, routed))
    broken = _identity_linear()
    with torch.no_grad():
        broken.weight[0, 0] = float('nan')
    _reject(lambda: SharedBranch('ungated', broken)(x, routed))


# %%
from torch_judge import evaluate
if not evaluate('0096_shared_branch_eval.py', SharedBranch):
    raise SystemExit(1)
