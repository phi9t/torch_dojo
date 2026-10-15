"""Adjacent-pair RoPE reference and staged checks."""
# %%
import torch

# %%
def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 3 or k.shape != q.shape or q.shape[-1] == 0 or q.shape[-1] % 2:
        raise ValueError("q,k must share [B,S,even D] with D>0")
    if q.device != k.device or q.dtype != k.dtype or not q.is_floating_point():
        raise ValueError("q,k need matching floating dtype/device")
    _, length, width = q.shape
    work_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    pos = torch.arange(length, device=q.device, dtype=work_dtype)
    pair = torch.arange(0, width, 2, device=q.device, dtype=work_dtype)
    angle = pos[:, None] * torch.exp(-pair[None, :] * (torch.log(torch.tensor(10000., device=q.device, dtype=work_dtype)) / width))
    c, s = angle.cos().to(q.dtype), angle.sin().to(q.dtype)

    def rotate(x):
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)

    return rotate(q), rotate(k)

# %% [verifiers]
def verify_baseline_known_rotation(apply_rope):
    q = torch.tensor([[[3., 4.], [1., 0.]]])
    got, other = apply_rope(q, q)
    expected = torch.tensor([[[3., 4.], [torch.cos(torch.tensor(1.)), torch.sin(torch.tensor(1.))]]])
    assert got.shape == q.shape and torch.allclose(got, expected, atol=1e-6)
    assert torch.equal(got, other)


def verify_followup_relative_positions_and_gradients(apply_rope):
    torch.manual_seed(4)
    q = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    qr, kr = apply_rope(q, k)
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-12)
    shifted_q = torch.cat((torch.zeros(2, 3, 8, dtype=q.dtype), q), dim=1)
    shifted_k = torch.cat((torch.zeros(2, 3, 8, dtype=k.dtype), k), dim=1)
    sq, sk = apply_rope(shifted_q, shifted_k)
    assert torch.allclose((qr[:, 1] * kr[:, 3]).sum(-1), (sq[:, 4] * sk[:, 6]).sum(-1), atol=1e-12)
    (qr.square().sum() + kr.sum()).backward()
    assert q.grad is not None and k.grad is not None and torch.isfinite(q.grad).all()
    noncontiguous = torch.randn(2, 8, 4, dtype=torch.float64).transpose(1, 2)
    assert not noncontiguous.is_contiguous()
    assert apply_rope(noncontiguous, noncontiguous)[0].shape == noncontiguous.shape


def verify_production_dtype_empty_and_invalid(apply_rope):
    q = torch.randn(1, 0, 6, dtype=torch.float64)
    out, _ = apply_rope(q, q)
    assert out.shape == q.shape and out.dtype == q.dtype
    for a, b in ((torch.ones(1, 2, 3), torch.ones(1, 2, 3)),
                 (torch.ones(1, 2, 4), torch.ones(1, 3, 4)),
                 (torch.ones(1, 2, 4), torch.ones(1, 2, 4, dtype=torch.float64)),
                 (torch.ones(1, 2, 4, dtype=torch.int64), torch.ones(1, 2, 4, dtype=torch.int64))):
        try:
            apply_rope(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid RoPE input accepted")

# %%
from torch_judge import evaluate
if not evaluate("0024_rope_eval.py", apply_rope):
    raise SystemExit(1)
