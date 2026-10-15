"""Reference and staged checks for a tiny trainable MoE classifier."""
# %%
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

# %%
class TinyMoE(nn.Module):
    def __init__(self, d_model, d_hidden, num_experts, top_k, num_classes):
        super().__init__()
        dimensions = (d_model, d_hidden, num_experts, top_k, num_classes)
        if (any(isinstance(v, bool) or not isinstance(v, int) for v in dimensions)
                or any(v <= 0 for v in dimensions) or top_k > num_experts
                or num_classes < 2):
            raise ValueError("need positive widths, C>=2, and 1<=K<=E")
        self.top_k = top_k
        self.num_classes = num_classes
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_hidden), nn.Tanh(),
                          nn.Linear(d_hidden, num_classes))
            for _ in range(num_experts))

    def forward(self, x):
        parameter = self.router.weight
        if (not isinstance(x, torch.Tensor) or x.ndim != 2
                or x.shape[1] != self.router.in_features or not x.is_floating_point()
                or x.dtype != parameter.dtype or x.device != parameter.device
                or not torch.isfinite(x).all()):
            raise ValueError("need finite x[T,D] matching model dtype/device")
        if x.shape[0] == 0:
            return (x.new_empty((0, self.num_classes)),
                    torch.empty((0, self.top_k), dtype=torch.long, device=x.device),
                    torch.zeros(len(self.experts), dtype=torch.long, device=x.device))
        logits = self.router(x)
        ids = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :self.top_k]
        weights = logits.gather(1, ids).softmax(dim=1)
        counts = torch.bincount(ids.flatten(), minlength=len(self.experts))
        output = x.new_zeros((len(x), self.num_classes))
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(ids == expert_id)
            if rows.numel():
                output = output.index_add(0, rows,
                                          expert(x[rows]) * weights[rows, slots, None])
        return output, ids, counts

    def train_step(self, x, targets, optimizer, balance_weight=0.0):
        output, ids, counts = self(x)
        if (len(x) == 0 or not isinstance(targets, torch.Tensor)
                or targets.shape != (len(x),) or targets.dtype != torch.long
                or targets.device != x.device or torch.any(targets < 0)
                or torch.any(targets >= self.num_classes)
                or isinstance(balance_weight, bool)
                or not isinstance(balance_weight, (int, float))
                or not math.isfinite(balance_weight) or balance_weight < 0):
            raise ValueError("need nonempty batch, valid targets, and finite nonnegative balance")
        ce = F.cross_entropy(output, targets)
        full_prob = self.router(x).softmax(dim=1).mean(dim=0)
        aux = len(self.experts) * torch.sum(full_prob * counts.detach() / (len(x) * self.top_k))
        total = ce + balance_weight * aux
        diagnostics = {"loss": total.item(), "ce": ce.item(), "aux": aux.item(),
                       "counts": counts.detach().cpu().clone()}
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        return diagnostics

# %% [verifiers]
def verify_baseline_registered_structure_and_shapes(TinyMoE):
    model = TinyMoE(3, 5, 4, 2, 3).double()
    assert isinstance(model.router, nn.Linear)
    assert isinstance(model.experts, nn.ModuleList) and len(model.experts) == 4
    assert all(isinstance(expert, nn.Sequential) and len(expert) == 3
               for expert in model.experts)
    assert sum(p.numel() for p in model.parameters()) == 4*3+4 + 4*(5*3+5+3*5+3)
    out, ids, counts = model(torch.randn(7, 3, dtype=torch.float64))
    assert out.shape == (7, 3) and ids.shape == (7, 2) and counts.shape == (4,)
    assert counts.sum().item() == 14 and counts.dtype == torch.long


def verify_baseline_known_route_and_tie(TinyMoE):
    model = TinyMoE(2, 3, 3, 2, 2).double()
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.copy_(torch.tensor([0., 2., 1.]))
        for index, expert in enumerate(model.experts):
            expert[0].weight.zero_(); expert[0].bias.zero_()
            expert[2].weight.zero_(); expert[2].bias.fill_(index + 1.)
    out, ids, counts = model(torch.zeros(2, 2, dtype=torch.float64))
    expected = 2 * math.e / (math.e + 1) + 3 / (math.e + 1)
    assert torch.equal(ids, torch.tensor([[1, 2], [1, 2]]))
    assert torch.equal(counts, torch.tensor([0, 2, 2]))
    assert torch.allclose(out, torch.full((2, 2), expected, dtype=torch.float64))
    with torch.no_grad():
        model.router.bias.zero_()
    _, ids, _ = model(torch.zeros(1, 2, dtype=torch.float64))
    assert torch.equal(ids, torch.tensor([[0, 1]]))


def _manual_objective(model, x, y, balance_weight):
    logits = model.router(x)
    ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :model.top_k]
    weights = logits.gather(1, ids).softmax(dim=1)
    out = torch.zeros(x.shape[0], model.num_classes, dtype=x.dtype)
    for token in range(len(x)):
        for slot in range(model.top_k):
            out[token] = out[token] + weights[token, slot] * model.experts[int(ids[token, slot])](x[token:token+1])[0]
    counts = torch.bincount(ids.flatten(), minlength=len(model.experts)).detach()
    aux = len(model.experts) * torch.sum(logits.softmax(dim=1).mean(dim=0)
                                         * counts / (x.shape[0] * model.top_k))
    ce = F.cross_entropy(out, y)
    return ce + balance_weight * aux, ce, aux, counts


def verify_followup_step_matches_independent_objective(TinyMoE):
    torch.manual_seed(102)
    model = TinyMoE(3, 4, 3, 2, 2).double()
    expected_model = copy.deepcopy(model)
    x = torch.randn(5, 3, dtype=torch.float64)
    y = torch.tensor([0, 1, 0, 1, 1])
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    expected_opt = torch.optim.SGD(expected_model.parameters(), lr=.05)
    expected_opt.zero_grad()
    total, ce, aux, counts = _manual_objective(expected_model, x, y, .3)
    total.backward(); expected_opt.step()
    diagnostic = model.train_step(x, y, optimizer, .3)
    assert set(diagnostic) == {"loss", "ce", "aux", "counts"}
    for key, expected in (("loss", total), ("ce", ce), ("aux", aux)):
        assert isinstance(diagnostic[key], float)
        assert abs(diagnostic[key] - expected.item()) < 1e-11
    assert torch.equal(diagnostic["counts"], counts.cpu())
    assert diagnostic["counts"].dtype == torch.long and not diagnostic["counts"].requires_grad
    for actual, expected in zip(model.parameters(), expected_model.parameters()):
        assert torch.allclose(actual, expected, atol=1e-11)


def verify_followup_only_selected_experts_get_task_gradient(TinyMoE):
    model = TinyMoE(2, 3, 3, 1, 2).double()
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.copy_(torch.tensor([2., 0., -1.]))
    x = torch.randn(4, 2, dtype=torch.float64)
    y = torch.tensor([0, 1, 0, 1])
    model.train_step(x, y, torch.optim.SGD(model.parameters(), lr=.01))
    assert any(p.grad is not None for p in model.experts[0].parameters())
    assert all(p.grad is None for expert in model.experts[1:] for p in expert.parameters())


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid tiny MoE input accepted")


def verify_production_empty_and_validation(TinyMoE):
    for args in ((0, 2, 3, 1, 2), (2, 0, 3, 1, 2), (2, 2, 0, 1, 2),
                 (2, 2, 3, 0, 2), (2, 2, 3, 4, 2), (2, 2, 3, 1, 1)):
        _reject(lambda args=args: TinyMoE(*args))
    model = TinyMoE(2, 3, 3, 2, 2).double()
    empty = torch.empty(0, 2, dtype=torch.float64)
    out, ids, counts = model(empty)
    assert out.shape == (0, 2) and ids.shape == (0, 2)
    assert torch.equal(counts, torch.zeros(3, dtype=torch.long))
    _reject(lambda: model(torch.ones(2, 3, dtype=torch.float64)))
    _reject(lambda: model(torch.ones(2, 2, dtype=torch.int64)))
    _reject(lambda: model(torch.full((2, 2), float("nan"), dtype=torch.float64)))
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    _reject(lambda: model.train_step(empty, torch.empty(0, dtype=torch.long), opt))
    _reject(lambda: model.train_step(torch.ones(2, 2, dtype=torch.float64), torch.tensor([0, 2]), opt))
    _reject(lambda: model.train_step(torch.ones(2, 2, dtype=torch.float64), torch.tensor([0., 1.]), opt))
    _reject(lambda: model.train_step(torch.ones(2, 2, dtype=torch.float64), torch.tensor([0, 1]), opt, -1.))


def verify_production_repeated_training_reduces_loss_and_preserves_inputs(TinyMoE):
    torch.manual_seed(6)
    model = TinyMoE(2, 4, 2, 2, 2).double()
    optimizer = torch.optim.SGD(model.parameters(), lr=.15)
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=torch.float64)
    y = torch.tensor([0, 0, 1, 1])
    old_x, old_y = x.clone(), y.clone()
    first = model.train_step(x, y, optimizer)
    last = first
    for _ in range(25):
        last = model.train_step(x, y, optimizer)
    assert last["ce"] < first["ce"] - .05
    assert torch.equal(x, old_x) and torch.equal(y, old_y)
    assert first["counts"].sum().item() == len(x) * model.top_k


def verify_production_top1_balance_feedback_changes_next_routes(TinyMoE):
    torch.manual_seed(202)
    task_only = TinyMoE(2, 3, 3, 1, 2).double()
    with torch.no_grad():
        task_only.router.weight.zero_()
        task_only.router.bias.copy_(torch.tensor([.2, .1, -1.]))
    balanced = copy.deepcopy(task_only)
    x = torch.zeros(4, 2, dtype=torch.float64)
    y = torch.tensor([0, 1, 0, 1])
    before = task_only(x)[2].cpu()
    assert torch.equal(before, torch.tensor([4, 0, 0]))
    task_report = task_only.train_step(x, y,
        torch.optim.SGD(task_only.parameters(), lr=.5), balance_weight=0.)
    balanced_report = balanced.train_step(x, y,
        torch.optim.SGD(balanced.parameters(), lr=.5), balance_weight=1.)
    assert torch.equal(task_report["counts"], before)
    assert torch.equal(balanced_report["counts"], before)
    assert torch.allclose(task_only.router.bias.grad,
                          torch.zeros_like(task_only.router.bias.grad), atol=1e-12)
    assert balanced.router.bias.grad is not None
    assert balanced.router.bias.grad.abs().max() > .1
    assert torch.equal(task_only(x)[2].cpu(), before)
    assert not torch.equal(balanced(x)[2].cpu(), before)


# %%
from torch_judge import evaluate
if not evaluate("0102_tiny_moe_update_eval.py", TinyMoE):
    raise SystemExit(1)
