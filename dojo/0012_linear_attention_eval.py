"""Reference and staged checks for linear attention."""
# %%
import torch
import torch.nn.functional as F

# %%
def linear_attention(Q, K, V):
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("Q,K,V must be rank three")
    b, sq, dk = Q.shape
    if dk <= 0 or K.shape[0] != b or K.shape[2] != dk or V.shape[:2] != K.shape[:2]:
        raise ValueError("batch, key width and key/value lengths must agree")
    if any(not t.is_floating_point() for t in (Q, K, V)) or len({t.dtype for t in (Q, K, V)}) != 1 or len({t.device for t in (Q, K, V)}) != 1:
        raise ValueError("Q,K,V must share floating dtype and device")
    if K.shape[1] == 0 and sq:
        raise ValueError("a query needs at least one key")
    if sq == 0:
        return V.new_empty((b, 0, V.shape[-1]))
    q, k = F.elu(Q) + 1, F.elu(K) + 1
    summary = k.transpose(1, 2) @ V
    mass = k.sum(dim=1)
    numerator = q @ summary
    denominator = (q * mass[:, None, :]).sum(dim=-1, keepdim=True)
    return numerator / denominator.clamp_min(torch.finfo(Q.dtype).tiny)

# %% [verifiers]
def _dense_kernel(Q, K, V):
    q, k = F.elu(Q) + 1, F.elu(K) + 1
    weights = q @ k.transpose(-2, -1)
    return (weights @ V) / weights.sum(-1, keepdim=True)


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def verify_baseline_equal_keys(linear_attention):
    q = torch.tensor([[[1., -2.], [-4., 3.]]], dtype=torch.float64)
    k = torch.zeros(1, 3, 2, dtype=torch.float64)
    v = torch.tensor([[[1.], [3.], [8.]]], dtype=torch.float64)
    assert torch.allclose(linear_attention(q, k, v), torch.full((1, 2, 1), 4., dtype=q.dtype))


def verify_baseline_dense_kernel_parity(linear_attention):
    torch.manual_seed(12)
    q, k, v = [torch.randn(2, 5, d, dtype=torch.float64) for d in (4, 4, 3)]
    assert torch.allclose(linear_attention(q, k, v), _dense_kernel(q, k, v), atol=1e-10)


def verify_followup_unequal_lengths(linear_attention):
    torch.manual_seed(3)
    q, k, v = torch.randn(2, 1, 4), torch.randn(2, 7, 4), torch.randn(2, 7, 6)
    assert linear_attention(q, k, v).shape == (2, 1, 6)
    assert torch.allclose(linear_attention(q, k, v), _dense_kernel(q, k, v), atol=1e-5)
    assert linear_attention(q[:, :0], k, v).shape == (2, 0, 6)


def verify_followup_batch_isolation(linear_attention):
    q, k, v = torch.randn(2, 5, 4), torch.randn(2, 8, 4), torch.randn(2, 8, 3)
    first = linear_attention(q, k, v)[0]
    k[1] *= 100
    assert torch.allclose(first, linear_attention(q, k, v)[0])


def verify_production_contract(linear_attention):
    q, k, v = torch.randn(1, 2, 3), torch.randn(1, 4, 3), torch.randn(1, 4, 2)
    _reject(lambda: linear_attention(q, k[:, :0], v[:, :0]))
    _reject(lambda: linear_attention(q, k, v[:, :2]))
    _reject(lambda: linear_attention(q.double(), k, v))
    _reject(lambda: linear_attention(q.squeeze(0), k, v))
    _reject(lambda: linear_attention(q[:, :, :0], k[:, :, :0], v))


def verify_production_gradients_and_long_sequence(linear_attention):
    q = torch.randn(1, 9, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 11, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 11, 2, dtype=torch.float64, requires_grad=True)
    got = linear_attention(q[:, ::2], k[:, ::2], v[:, ::2])
    got.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    # Small feature width with long sequence makes an S-by-S implementation
    # disproportionately costly; reference uses only K^T V and sum(K).
    long = torch.randn(1, 4096, 4)
    assert linear_attention(long, long, long[:, :, :2]).shape == (1, 4096, 2)

# %%
from torch_judge import evaluate
if not evaluate("0012_linear_attention_eval.py", linear_attention):
    raise SystemExit(1)
