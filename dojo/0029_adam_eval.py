"""Adam reference and cumulative checks. Run this script for all stages."""
# %%
import math
import torch

# %%
class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        if not self.params or len({id(p) for p in self.params}) != len(self.params):
            raise ValueError("params must be nonempty and distinct")
        if any(not isinstance(p, torch.Tensor) or not p.is_leaf or not p.is_floating_point() for p in self.params):
            raise ValueError("params must be leaf floating tensors")
        if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if len(betas) != 2 or any(not isinstance(b, (int, float)) or not 0 <= b < 1 for b in betas):
            raise ValueError("betas must lie in [0,1)")
        if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.lr, self.betas, self.eps = lr, betas, eps
        self.state = {id(p): [torch.zeros_like(p), torch.zeros_like(p), 0] for p in self.params}

    def step(self):
        beta1, beta2 = self.betas
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse or grad.shape != p.shape:
                    raise ValueError("dense gradient must match parameter shape")
                m, v, t = self.state[id(p)]
                t += 1
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = (m / (1 - beta1 ** t)) / ((v / (1 - beta2 ** t)).sqrt() + self.eps)
                p.add_(update, alpha=-self.lr)
                self.state[id(p)][2] = t

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# %% [verifiers]
def verify_baseline_scalar_bias_correction(MyAdam):
    p = torch.tensor([1.0], requires_grad=True)
    opt = MyAdam([p], lr=0.1)
    p.grad = torch.tensor([2.0])
    opt.step()
    assert torch.allclose(p, torch.tensor([0.9]), atol=1e-6)
    assert p.grad.item() == 2.0


def verify_baseline_zero_grad(MyAdam):
    p = torch.tensor([1.0], requires_grad=True)
    opt = MyAdam([p])
    p.grad = torch.tensor([3.0])
    opt.zero_grad()
    assert p.grad.item() == 0.0 and p.item() == 1.0


def verify_followup_skipped_parameter_has_own_clock(MyAdam):
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    opt = MyAdam([a, b], lr=0.1, betas=(0.5, 0.9))
    a.grad = torch.tensor([2.0])
    opt.step()
    assert b.item() == 1.0
    a.grad = None
    b.grad = torch.tensor([2.0])
    opt.step()
    assert torch.allclose(b, torch.tensor([0.9]), atol=1e-6)


def verify_followup_repeated_step_and_none_grad(MyAdam):
    p = torch.tensor([1.0], requires_grad=True)
    opt = MyAdam([p], lr=0.01)
    opt.step()
    assert p.item() == 1.0
    for _ in range(2):
        p.grad = torch.tensor([p.item() * 2])
        opt.step()
        opt.zero_grad()
    assert p.item() < 1.0 and p.grad.item() == 0.0


def verify_production_torch_parity_with_delayed_grad(MyAdam):
    torch.manual_seed(19)
    p, q = [torch.randn(7, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    p_ref, q_ref = [x.detach().clone().requires_grad_() for x in (p, q)]
    opt = MyAdam([p, q], lr=0.005, betas=(0.7, 0.95), eps=1e-8)
    ref = torch.optim.Adam([p_ref, q_ref], lr=0.005, betas=(0.7, 0.95), eps=1e-8)
    for i in range(6):
        for x, y in ((p, p_ref), (q, q_ref)):
            gradient = torch.arange(1, 8, dtype=torch.float64) * (i + 1) / 7
            x.grad = None if x is q and i < 3 else gradient.clone()
            y.grad = None if y is q_ref and i < 3 else gradient.clone()
        opt.step()
        ref.step()
    assert torch.allclose(p, p_ref, atol=1e-12)
    assert torch.allclose(q, q_ref, atol=1e-12)


def verify_production_invalid_configuration(MyAdam):
    p = torch.tensor([1.0], requires_grad=True)
    for kwargs in ({"lr": -1}, {"eps": 0}, {"betas": (1, 0.9)}):
        try:
            MyAdam([p], **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject {kwargs}")

# %%
from torch_judge import evaluate
if not evaluate("0029_adam_eval.py", MyAdam):
    raise SystemExit(1)
