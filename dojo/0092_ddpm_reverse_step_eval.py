"""Reference and checks for one noise-prediction DDPM reverse step."""

# %%
import torch
from numbers import Integral

# %%
def ddpm_reverse_step(x_t, predicted_noise, betas, step, noise=None, generator=None):
    if (not isinstance(x_t, torch.Tensor) or not x_t.is_floating_point()
            or x_t.numel() == 0 or not torch.isfinite(x_t).all()):
        raise ValueError('x_t must be a nonempty finite floating tensor')
    if (not isinstance(predicted_noise, torch.Tensor)
            or predicted_noise.shape != x_t.shape or predicted_noise.dtype != x_t.dtype
            or predicted_noise.device != x_t.device or not torch.isfinite(predicted_noise).all()):
        raise ValueError('predicted noise must match x_t and be finite')
    if (not isinstance(betas, torch.Tensor) or betas.ndim != 1 or betas.numel() == 0
            or betas.dtype != x_t.dtype or betas.device != x_t.device
            or not torch.isfinite(betas).all() or not ((betas > 0) & (betas < 1)).all()):
        raise ValueError('betas must be a finite [T] schedule in (0, 1)')
    if not isinstance(step, Integral) or isinstance(step, bool) or not 1 <= step <= betas.numel():
        raise ValueError('step must be a one-based schedule index')
    if noise is not None and (
            not isinstance(noise, torch.Tensor) or noise.shape != x_t.shape
            or noise.dtype != x_t.dtype or noise.device != x_t.device
            or not torch.isfinite(noise).all()):
        raise ValueError('supplied noise must match x_t and be finite')
    if generator is not None and not isinstance(generator, torch.Generator):
        raise ValueError('generator must be a torch.Generator')
    if generator is not None and step > 1 and generator.device.type != x_t.device.type:
        raise ValueError('generator device must match x_t')
    # Keep small valid betas from disappearing in 1 - beta at input precision.
    schedule = betas.to(torch.float64)
    log_alpha = torch.log1p(-schedule)
    one_minus_alpha_bar = -torch.expm1(log_alpha[:step].sum())
    alpha_t = 1 - schedule[step - 1]
    mean = (x_t.to(torch.float64) - schedule[step - 1] * predicted_noise.to(torch.float64)
            / torch.sqrt(one_minus_alpha_bar)) / torch.sqrt(alpha_t)
    if step == 1:
        return mean.to(x_t.dtype)
    one_minus_alpha_bar_previous = -torch.expm1(log_alpha[:step - 1].sum())
    variance = schedule[step - 1] * one_minus_alpha_bar_previous / one_minus_alpha_bar
    if noise is None:
        noise = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device, generator=generator)
    return (mean + torch.sqrt(variance) * noise.to(torch.float64)).to(x_t.dtype)


# %% [verifiers]
def verify_baseline_hand_calculated_mean(ddpm_reverse_step):
    x = torch.tensor([2.], dtype=torch.float64)
    eps = torch.tensor([1.], dtype=torch.float64)
    result = ddpm_reverse_step(x, eps, torch.tensor([.1, .2], dtype=torch.float64), 1)
    assert torch.allclose(result, torch.tensor([1.7748517734455862], dtype=torch.float64), atol=1e-12)


def verify_followup_last_step_does_not_consume_generator(ddpm_reverse_step):
    betas = torch.tensor([.1, .2], dtype=torch.float64)
    x = torch.tensor([2.], dtype=torch.float64)
    eps = torch.tensor([1.], dtype=torch.float64)
    rng = torch.Generator().manual_seed(31)
    state = rng.get_state().clone()
    actual = ddpm_reverse_step(x, eps, betas, 1, generator=rng)
    assert torch.allclose(actual, torch.tensor([1.7748517734455862], dtype=torch.float64), atol=1e-12)
    assert torch.equal(rng.get_state(), state)


def verify_followup_posterior_variance_and_seeded_draw(ddpm_reverse_step):
    betas = torch.tensor([.2, .5], dtype=torch.float64)
    x = torch.ones((2, 2), dtype=torch.float64)
    eps = torch.zeros_like(x)
    # alpha_bar_1 = .8, alpha_bar_2 = .4, so variance = .5*.2/.6 = 1/6.
    supplied = torch.tensor([[2., -1.], [0., 3.]], dtype=torch.float64)
    expected_mean = 2**.5
    expected = expected_mean + supplied / 6**.5
    assert torch.allclose(ddpm_reverse_step(x, eps, betas, 2, noise=supplied), expected, atol=1e-12)
    rng = torch.Generator().manual_seed(42)
    expected_rng = torch.Generator().manual_seed(42)
    expected_draw = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=expected_rng)
    sampled = ddpm_reverse_step(x, eps, betas, 2, generator=rng)
    assert torch.allclose(sampled, expected_mean + expected_draw / 6**.5, atol=1e-12)
    assert torch.equal(rng.get_state(), expected_rng.get_state())


def verify_production_rejects_invalid_inputs(ddpm_reverse_step):
    x = torch.ones((2, 1), dtype=torch.float64)
    eps = torch.zeros_like(x)
    betas = torch.tensor([.1, .2], dtype=torch.float64)
    bad_calls = [
        (x, eps, betas, 0), (x, eps, betas, 3),
        (x, eps, betas, 1.0), (x, eps, betas, True),
        (x, eps, torch.empty(0, dtype=x.dtype), 1),
        (x, eps, torch.tensor([0., .2], dtype=x.dtype), 1),
        (x, eps, torch.tensor([.1, 1.], dtype=x.dtype), 1),
        (x, eps, torch.tensor([.1, float('nan')], dtype=x.dtype), 1),
        (x, eps, betas[:, None], 1),
        (x, eps.float(), betas, 1),
        (x, eps[:1], betas, 1),
        (x, torch.full_like(x, float('inf')), betas, 1),
        (torch.full_like(x, float('nan')), eps, betas, 1),
        (x, eps, betas.float(), 1),
        (torch.empty(0, dtype=x.dtype), torch.empty(0, dtype=x.dtype), betas, 1),
    ]
    for args in bad_calls:
        try:
            ddpm_reverse_step(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid DDPM input accepted: step={args[3]}')
    for bad_noise in (eps[:1], eps.float(), torch.full_like(x, float('nan'))):
        try:
            ddpm_reverse_step(x, eps, betas, 2, noise=bad_noise)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid supplied noise accepted')
    if torch.cuda.is_available():
        for args, kwargs in (
            ((x.cuda(), eps, betas, 1), {}),
            ((x, eps.cuda(), betas, 1), {}),
            ((x, eps, betas.cuda(), 1), {}),
            ((x, eps, betas, 2), {'noise': eps.cuda()}),
        ):
            try:
                ddpm_reverse_step(*args, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError('device mismatch accepted')


def verify_production_batch_gradient_and_immutability(ddpm_reverse_step):
    source = torch.tensor([[[1., 7.], [2., 8.]], [[3., 9.], [4., 10.]]],
                          dtype=torch.float64, requires_grad=True)
    x = source[..., 0]
    eps = torch.full_like(x, .25, requires_grad=True)
    betas = torch.tensor([.1, .2, .3], dtype=torch.float64)
    noise = torch.full_like(x, -.5)
    before = [v.detach().clone() for v in (source, eps, betas, noise)]
    output = ddpm_reverse_step(x, eps, betas, 3, noise=noise)
    assert output.shape == x.shape and output.dtype == x.dtype and output.device == x.device
    output.sum().backward()
    alpha_bar = (.9 * .8 * .7)
    expected_x_grad = 1 / .7**.5
    expected_eps_grad = -.3 / ((1 - alpha_bar)**.5 * .7**.5)
    assert torch.allclose(source.grad[..., 0], torch.full_like(x, expected_x_grad))
    assert torch.equal(source.grad[..., 1], torch.zeros_like(x))
    assert torch.allclose(eps.grad, torch.full_like(x, expected_eps_grad))
    for value, original in zip((source, eps, betas, noise), before):
        assert torch.equal(value.detach(), original)


def verify_production_small_beta_across_floating_dtypes(ddpm_reverse_step):
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        x = torch.tensor([1.], dtype=dtype, requires_grad=True)
        eps = torch.tensor([1.], dtype=dtype, requires_grad=True)
        betas = torch.tensor([1e-4 if dtype != torch.float32 else 1e-10], dtype=dtype)
        result = ddpm_reverse_step(x, eps, betas, 1)
        beta = betas.item()
        expected = torch.tensor([(1 - beta**.5) / (1 - beta)**.5], dtype=dtype)
        assert result.dtype == dtype and torch.isfinite(result).all()
        assert torch.allclose(result, expected, atol=1e-7, rtol=0)
        result.sum().backward()
        assert torch.isfinite(x.grad).all() and torch.isfinite(eps.grad).all()


# %%
from torch_judge import evaluate
if not evaluate('0092_ddpm_reverse_step_eval.py', ddpm_reverse_step):
    raise SystemExit(1)
