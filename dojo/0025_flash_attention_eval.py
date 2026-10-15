"""Reference and staged checks for tiled online-softmax attention."""
# %%
import math
import torch

# %%
def flash_attention(Q, K, V, block_size=32):
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be positive int")
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("Q,K,V must be rank three")
    b, sq, d = Q.shape
    if d <= 0 or K.shape[0] != b or K.shape[2] != d or V.shape[:2] != K.shape[:2]:
        raise ValueError("batch, key width and KV length must match")
    if (any(not t.is_floating_point() for t in (Q, K, V))
            or len({t.dtype for t in (Q, K, V)}) != 1
            or len({t.device for t in (Q, K, V)}) != 1):
        raise ValueError("Q,K,V must share floating dtype and device")
    sk = K.shape[1]
    if sk == 0 and sq:
        raise ValueError("queries require keys")
    if sq == 0:
        return V.new_empty((b, 0, V.shape[-1]))
    chunks = []
    for i in range(0, sq, block_size):
        q = Q[:, i:i+block_size]
        width = q.shape[1]
        m = Q.new_full((b, width, 1), -torch.inf)
        mass = Q.new_zeros((b, width, 1))
        weighted = Q.new_zeros((b, width, V.shape[-1]))
        for j in range(0, sk, block_size):
            k, v = K[:, j:j+block_size], V[:, j:j+block_size]
            scores = q @ k.transpose(-2, -1) / math.sqrt(d)
            next_m = torch.maximum(m, scores.max(-1, keepdim=True).values)
            correction = torch.exp(m - next_m)
            e = torch.exp(scores - next_m)
            weighted = weighted * correction + e @ v
            mass = mass * correction + e.sum(-1, keepdim=True)
            m = next_m
        chunks.append(weighted / mass)
    return torch.cat(chunks, dim=1)

# %% [verifiers]
def _dense(Q, K, V):
    return torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1]), -1) @ V


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def verify_baseline_dense_parity(flash_attention):
    torch.manual_seed(5)
    q, k, v = [torch.randn(2, 8, d, dtype=torch.float64) for d in (4, 4, 3)]
    assert torch.allclose(flash_attention(q, k, v, 3), _dense(q, k, v), atol=1e-11)


def verify_baseline_one_key(flash_attention):
    q = torch.tensor([[[100., -100.], [0., 0.]]])
    k = torch.tensor([[[1., 2.]]])
    v = torch.tensor([[[3., 5., 7.]]])
    assert torch.allclose(flash_attention(q, k, v, 1), v.expand(1, 2, 3))


def verify_followup_unequal_lengths_and_blocks(flash_attention):
    torch.manual_seed(6)
    q, k, v = torch.randn(2, 5, 4), torch.randn(2, 7, 4), torch.randn(2, 7, 3)
    expected = _dense(q, k, v)
    for block in (1, 2, 3, 9):
        assert torch.allclose(flash_attention(q, k, v, block), expected, atol=1e-5)
    assert flash_attention(q[:, :0], k, v).shape == (2, 0, 3)


def verify_followup_large_score_range(flash_attention):
    q = torch.tensor([[[200., 0.], [0., 200.]]], dtype=torch.float64)
    k = torch.tensor([[[0., 1.], [1., 0.], [-1., 0.]]], dtype=torch.float64)
    v = torch.tensor([[[1.], [3.], [9.]]], dtype=torch.float64)
    got = flash_attention(q, k, v, 1)
    assert torch.isfinite(got).all() and torch.allclose(got, _dense(q, k, v), atol=1e-10)


def verify_production_validation(flash_attention):
    q, k, v = torch.randn(1, 2, 3), torch.randn(1, 4, 3), torch.randn(1, 4, 2)
    for bad in (0, -1, 1.5, True):
        _reject(lambda: flash_attention(q, k, v, bad))
    _reject(lambda: flash_attention(q, k[:, :0], v[:, :0]))
    _reject(lambda: flash_attention(q.double(), k, v))
    _reject(lambda: flash_attention(q, k, v[:, :2]))


def verify_production_noncontiguous_gradients(flash_attention):
    q = torch.randn(2, 9, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 11, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 11, 3, dtype=torch.float64, requires_grad=True)
    got = flash_attention(q[:, ::2], k[:, ::2], v[:, ::2], 2)
    expected = _dense(q[:, ::2], k[:, ::2], v[:, ::2])
    assert torch.allclose(got, expected, atol=1e-11)
    got.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))

# %%
from torch_judge import evaluate
if not evaluate("0025_flash_attention_eval.py", flash_attention):
    raise SystemExit(1)
