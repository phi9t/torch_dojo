"""Three-stage reference for multi-head attention."""

# %%
import math
import torch
from torch import nn

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if not isinstance(d_model, int) or not isinstance(num_heads, int) or d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("d_model must be a positive multiple of num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                allow: torch.Tensor | None = None) -> torch.Tensor:
        if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3 or Q.shape[0] != K.shape[0] or K.shape[:2] != V.shape[:2] or any(x.shape[-1] != self.d_model for x in (Q, K, V)) or K.shape[1] == 0:
            raise ValueError("need Q[B,Tq,D], K,V[B,Tk,D], Tk >= 1")
        if Q.dtype != K.dtype or Q.dtype != V.dtype or Q.device != K.device or Q.device != V.device:
            raise ValueError("Q, K, V need matching dtype and device")
        B, Tq, _ = Q.shape
        Tk = K.shape[1]

        def split(x: torch.Tensor, length: int) -> torch.Tensor:
            return x.reshape(B, length, self.num_heads, self.d_k).transpose(1, 2)

        q = split(self.W_q(Q), Tq)
        k = split(self.W_k(K), Tk)
        v = split(self.W_v(V), Tk)
        work_dtype = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
        scores = q.to(work_dtype) @ k.to(work_dtype).transpose(-2, -1) / math.sqrt(self.d_k)
        query_has_key = None
        if allow is not None:
            if allow.dtype != torch.bool or allow.device != Q.device:
                raise ValueError("allow must be bool on the input device")
            try:
                mask = torch.broadcast_to(allow, scores.shape)
            except RuntimeError as exc:
                raise ValueError("allow is not broadcastable to attention scores") from exc
            has_key = mask.any(dim=-1, keepdim=True)
            safe_scores = torch.where(has_key, scores.masked_fill(~mask, -torch.inf), torch.zeros_like(scores))
            weights = torch.softmax(safe_scores, dim=-1)
            weights = torch.where(mask, weights, torch.zeros_like(weights))
            query_has_key = has_key.any(dim=1).squeeze(-1)
        else:
            weights = torch.softmax(scores, dim=-1)
        attended = (weights @ v.to(work_dtype)).to(Q.dtype)
        joined = attended.transpose(1, 2).reshape(B, Tq, self.d_model)
        output = self.W_o(joined)
        if query_has_key is not None:
            output = output.masked_fill(~query_has_key[..., None], 0)
        return output


# %% [verifiers]
def verify_baseline_shapes(MultiHeadAttention):
    model = MultiHeadAttention(12, 3)
    Q, K, V = torch.randn(2, 3, 12), torch.randn(2, 5, 12), torch.randn(2, 5, 12)
    assert model.forward(Q, K, V).shape == (2, 3, 12)
    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert isinstance(getattr(model, name), nn.Linear)
        assert getattr(model, name).weight.shape == (12, 12)


def verify_baseline_numerical_parity(MultiHeadAttention):
    torch.manual_seed(11)
    model = MultiHeadAttention(8, 2)
    Q, K, V = torch.randn(1, 3, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    q = model.W_q(Q).reshape(1, 3, 2, 4).transpose(1, 2)
    k = model.W_k(K).reshape(1, 4, 2, 4).transpose(1, 2)
    v = model.W_v(V).reshape(1, 4, 2, 4).transpose(1, 2)
    expected = model.W_o((torch.softmax(q @ k.transpose(-2, -1) / 2, -1) @ v).transpose(1, 2).reshape(1, 3, 8))
    assert torch.allclose(model.forward(Q, K, V), expected, atol=1e-6)


def verify_followup_module_and_gradients(MultiHeadAttention):
    model = MultiHeadAttention(8, 2)
    assert isinstance(model, nn.Module) and len(list(model.parameters())) == 8
    base = torch.randn(3, 2, 8, dtype=torch.float64, requires_grad=True)
    x = base.transpose(0, 1)
    model = model.to(dtype=torch.float64)
    model(x, x, x).square().sum().backward()
    assert base.grad is not None and torch.isfinite(base.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def verify_followup_invalid_shapes(MultiHeadAttention):
    try:
        MultiHeadAttention(7, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("non-divisible width accepted")
    model = MultiHeadAttention(8, 2)
    try:
        model(torch.randn(1, 2, 8), torch.randn(1, 3, 8), torch.randn(1, 4, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("different K/V lengths accepted")


def verify_production_allowed_mask_and_all_masked_rows(MultiHeadAttention):
    torch.manual_seed(0)
    model = MultiHeadAttention(8, 2)
    Q = torch.randn(1, 2, 8, requires_grad=True)
    K = torch.randn(1, 3, 8, requires_grad=True)
    V = torch.randn(1, 3, 8, requires_grad=True)
    allow = torch.tensor([[[[True, False, False], [False, False, False]]]])
    result = model(Q, K, V, allow)
    assert torch.equal(result[:, 1], torch.zeros_like(result[:, 1]))
    changed = V.detach().clone(); changed[:, 1:] += 1000
    assert torch.allclose(result[:, 0], model(Q, K, changed, allow)[:, 0], atol=1e-5)
    result.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (Q, K, V))


def verify_production_batch_isolation_and_mask_contract(MultiHeadAttention):
    model = MultiHeadAttention(8, 2)
    Q, K, V = (torch.randn(2, 3, 8) for _ in range(3))
    allow = torch.ones((2, 1, 3, 3), dtype=torch.bool)
    expected = model(Q, K, V, allow)
    changed = K.clone(); changed[1] += 100
    actual = model(Q, changed, V, allow)
    assert torch.allclose(expected[0], actual[0], atol=1e-6)
    for bad in (torch.ones(2, 3, 3), torch.ones(1, 1, 3, 3, dtype=torch.float32)):
        try:
            model(Q, K, V, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mask accepted")


# %%
from torch_judge import evaluate
if not evaluate("0006_multihead_attention_eval.py", MultiHeadAttention):
    raise SystemExit(1)
