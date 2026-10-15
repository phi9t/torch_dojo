"""Three-stage batch normalization reference and independent checks."""
# %%
import torch

# %%
def my_batch_norm(x, gamma, beta, running_mean, running_var,
                  eps=1e-5, momentum=0.1, training=True):
    if not all(isinstance(t, torch.Tensor) for t in (x, gamma, beta, running_mean, running_var)):
        raise ValueError("all inputs must be tensors")
    if x.ndim != 2 or min(x.shape) < 1 or any(t.shape != (x.shape[1],) for t in (gamma, beta, running_mean, running_var)):
        raise ValueError("x [N,D] and all feature tensors [D] required")
    if not x.is_floating_point() or not torch.isfinite(x).all() or any(not t.is_floating_point() or not torch.isfinite(t).all() for t in (gamma, beta, running_mean, running_var)):
        raise ValueError("finite floating inputs required")
    if any(t.dtype != x.dtype or t.device != x.device for t in (gamma, beta, running_mean, running_var)):
        raise ValueError("feature tensors must share x dtype/device")
    if not (0 < eps < float('inf')) or not (0 <= momentum <= 1):
        raise ValueError("eps>0 and momentum in [0,1] required")
    if training:
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        with torch.no_grad():
            running_mean.lerp_(mean.detach(), momentum)
            running_var.lerp_(var.detach(), momentum)
    else:
        mean, var = running_mean, running_var
    return (x - mean) * torch.rsqrt(var + eps) * gamma + beta

# %% [verifiers]
def verify_baseline_training_normalizes_features(my_batch_norm):
    x = torch.tensor([[1., 10.], [3., 20.]], requires_grad=True)
    g, b = torch.ones(2, requires_grad=True), torch.zeros(2, requires_grad=True)
    rm, rv = torch.zeros(2), torch.ones(2)
    out = my_batch_norm(x, g, b, rm, rv, eps=1e-5)
    expected = (x - x.mean(0)) / torch.sqrt(x.var(0, unbiased=False) + 1e-5)
    assert out.shape == x.shape and torch.allclose(out, expected, atol=1e-5)
    out.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x, g, b))


def verify_followup_running_buffers_and_eval(my_batch_norm):
    x = torch.tensor([[1., 3.], [5., 7.], [9., 11.]])
    g, b = torch.tensor([2., 3.]), torch.tensor([1., -1.])
    rm, rv = torch.tensor([10., -2.]), torch.tensor([4., 8.])
    my_batch_norm(x, g, b, rm, rv, momentum=.25, training=True)
    assert torch.allclose(rm, .75*torch.tensor([10., -2.]) + .25*x.mean(0))
    assert torch.allclose(rv, .75*torch.tensor([4., 8.]) + .25*x.var(0, unbiased=False))
    old_rm, old_rv = rm.clone(), rv.clone()
    out = my_batch_norm(x, g, b, rm, rv, training=False)
    assert torch.allclose(out, (x-old_rm)*torch.rsqrt(old_rv+1e-5)*g+b)
    assert torch.equal(rm, old_rm) and torch.equal(rv, old_rv)
    single = my_batch_norm(x[:1], g, b, rm, rv, training=True)
    assert torch.allclose(single, b[None], atol=1e-5)


def verify_production_validation_and_no_partial_mutation(my_batch_norm):
    x, g, b = torch.ones(2, 3), torch.ones(3), torch.zeros(3)
    rm, rv = torch.zeros(3), torch.ones(3)
    for bad_x, bad_eps, bad_mom in ((x[:, :2], 1e-5, .1), (x, 0, .1), (x, 1e-5, 1.1), (torch.full_like(x, float('nan')), 1e-5, .1)):
        try:
            my_batch_norm(bad_x, g, b, rm, rv, eps=bad_eps, momentum=bad_mom)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
        assert torch.equal(rm, torch.zeros(3)) and torch.equal(rv, torch.ones(3))


def verify_production_dtype_and_buffer_graph(my_batch_norm):
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    g, b = torch.ones(3, dtype=x.dtype, requires_grad=True), torch.zeros(3, dtype=x.dtype, requires_grad=True)
    rm, rv = torch.zeros(3, dtype=x.dtype), torch.ones(3, dtype=x.dtype)
    result = my_batch_norm(x, g, b, rm, rv)
    assert result.dtype == x.dtype and not rm.requires_grad and not rv.requires_grad
    result.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x, g, b))

# %%
from torch_judge import evaluate
if not evaluate('0007_batchnorm_eval.py', my_batch_norm):
    raise SystemExit(1)
