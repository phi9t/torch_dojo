"""Reference and staged checks for least-squares regression with three solvers."""
# %%
import math
import torch
from torch import nn

# %%
class LinearRegression:
    @staticmethod
    def _validate(X, y):
        if (X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]
                or X.shape[0] == 0 or X.shape[1] == 0
                or X.dtype not in (torch.float32, torch.float64) or y.dtype != X.dtype
                or y.device != X.device or not torch.isfinite(X).all()
                or not torch.isfinite(y).all()):
            raise ValueError("expect finite nonempty matching floating X[N,D], y[N]")

    @staticmethod
    def _training_args(lr, steps):
        if not isinstance(steps, int) or steps < 0 or not math.isfinite(lr) or lr <= 0:
            raise ValueError("expect positive finite lr and nonnegative integer steps")

    def closed_form(self, X, y):
        self._validate(X, y)
        with torch.no_grad():
            design = torch.cat((X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)), dim=1)
            theta = torch.linalg.lstsq(design, y).solution
            return theta[:-1], theta[-1]

    def gradient_descent(self, X, y, lr=0.01, steps=1000):
        self._validate(X, y)
        self._training_args(lr, steps)
        with torch.no_grad():
            w = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
            b = X.new_zeros(())
            for _ in range(steps):
                error = X @ w + b - y
                w = w - (2 * lr / X.shape[0]) * (X.T @ error)
                b = b - (2 * lr / X.shape[0]) * error.sum()
            return w, b

    def nn_linear(self, X, y, lr=0.01, steps=1000):
        self._validate(X, y)
        self._training_args(lr, steps)
        layer = nn.Linear(X.shape[1], 1, device=X.device, dtype=X.dtype)
        optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = (layer(X).squeeze(-1) - y).square().mean()
            loss.backward()
            optimizer.step()
        return layer.weight.detach().reshape(-1).clone(), layer.bias.detach().reshape(()).clone()

# %% [verifiers]
def _data(dtype=torch.float64):
    X = torch.tensor([[-2., 0.], [-1., 1.], [0., 2.], [1., -1.], [2., 1.], [3., -2.]], dtype=dtype)
    y = X @ torch.tensor([2., -1.], dtype=dtype) + 3
    return X, y


def verify_baseline_closed_form_and_shapes(LinearRegression):
    X, y = _data()
    model = LinearRegression()
    w, b = model.closed_form(X, y)
    assert w.shape == (2,) and b.shape == ()
    assert torch.allclose(w, torch.tensor([2., -1.], dtype=X.dtype), atol=1e-10)
    assert torch.allclose(b, torch.tensor(3., dtype=X.dtype), atol=1e-10)


def verify_baseline_manual_and_nn_converge(LinearRegression):
    X, y = _data(torch.float32)
    model = LinearRegression()
    for method in (model.gradient_descent, model.nn_linear):
        w, b = method(X, y, lr=.05, steps=400)
        assert w.shape == (2,) and b.shape == ()
        assert torch.mean((X @ w + b - y).square()) < 1e-5


def verify_followup_rank_deficient_minimum_norm(LinearRegression):
    X = torch.tensor([[0., 0.], [1., 1.], [2., 2.], [3., 3.]], dtype=torch.float64)
    y = 2 * X[:, 0] + 1
    w, b = LinearRegression().closed_form(X, y)
    assert torch.isfinite(w).all() and torch.isfinite(b)
    assert torch.allclose(X @ w + b, y, atol=1e-9)
    assert torch.allclose(w[0], w[1], atol=1e-9)


def verify_followup_dtype_and_input_immutability(LinearRegression):
    X, y = _data()
    X0, y0 = X.clone(), y.clone()
    model = LinearRegression()
    for method in (model.closed_form, model.gradient_descent, model.nn_linear):
        kwargs = {} if method == model.closed_form else {'lr': .05, 'steps': 60}
        w, b = method(X, y, **kwargs)
        assert w.dtype == X.dtype and b.dtype == X.dtype and w.device == X.device
    assert torch.equal(X, X0) and torch.equal(y, y0)


def verify_production_ill_conditioned_closed_form(LinearRegression):
    # Differently scaled columns make X.T @ X less well conditioned than X.
    x = torch.arange(1., 21., dtype=torch.float64)
    X = torch.stack((x, 1e-3 * x.square()), 1)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 7
    w, b = LinearRegression().closed_form(X, y)
    assert torch.max((X @ w + b - y).abs()) < 1e-9
    assert torch.allclose(w, torch.tensor([3., -2.], dtype=X.dtype), atol=1e-7)


def verify_production_validation_and_zero_steps(LinearRegression):
    model = LinearRegression()
    X, y = _data()
    w, b = model.gradient_descent(X, y, lr=.1, steps=0)
    assert torch.equal(w, torch.zeros_like(w)) and b.item() == 0
    for bad_X, bad_y in ((X[:0], y[:0]), (X, y[:1]), (X.long(), y)):
        try:
            model.closed_form(bad_X, bad_y)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid data must fail")
    try:
        model.gradient_descent(X, y, lr=0.)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid learning rate must fail")

# %%
from torch_judge import evaluate
if not evaluate('0040_linear_regression_eval.py', LinearRegression):
    raise SystemExit(1)
