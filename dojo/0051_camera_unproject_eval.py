"""Camera unprojection reference and cumulative geometry checks."""

# %%
import torch

# %%
def camera_unproject(pixels: torch.Tensor, depth: torch.Tensor,
                     K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or depth.shape != (len(pixels),)
            or K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,)
            or not pixels.is_floating_point() or pixels.dtype not in (torch.float32, torch.float64)
            or any(x.dtype != pixels.dtype or x.device != pixels.device for x in (depth, K, R, t))
            or any(not torch.isfinite(x).all() for x in (pixels, depth, K, R, t))
            or torch.any(depth <= 0)
            or not torch.allclose(K[2], K.new_tensor([0., 0., 1.]))):
        raise ValueError('finite matching pinhole tensors and positive depth required')
    try:
        pixel_h = torch.cat((pixels, torch.ones_like(depth[:, None])), dim=1)
        camera = torch.linalg.solve(K, pixel_h.T).T * depth[:, None]
    except torch.linalg.LinAlgError as error:
        raise ValueError('K must be invertible') from error
    return (camera - t) @ R

# %% [verifiers]
def verify_baseline_principal_point_and_focal_lengths(camera_unproject):
    K = torch.tensor([[100., 0., 20.], [0., 50., 10.], [0., 0., 1.]], dtype=torch.float64)
    uv = torch.tensor([[120., 60.], [-30., 35.]], dtype=K.dtype)
    out = camera_unproject(uv, torch.tensor([2., 4.], dtype=K.dtype), K,
                           torch.eye(3, dtype=K.dtype), torch.zeros(3, dtype=K.dtype))
    assert torch.allclose(out, torch.tensor([[2., 2., 2.], [-2., 2., 4.]], dtype=K.dtype), atol=1e-12)


def verify_followup_inverse_rotation_translation(camera_unproject):
    dtype = torch.float64
    R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=dtype)
    t = torch.tensor([1., 2., 3.], dtype=dtype)
    out = camera_unproject(torch.tensor([[-3., 1.]], dtype=dtype), torch.tensor([2.], dtype=dtype),
                           torch.eye(3, dtype=dtype), R, t)
    assert torch.allclose(out, torch.tensor([[0., 7., -1.]], dtype=dtype), atol=1e-12)


def verify_followup_gradients(camera_unproject):
    uv = torch.tensor([[0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    depth = torch.tensor([3.], dtype=uv.dtype, requires_grad=True)
    out = camera_unproject(uv, depth, torch.eye(3, dtype=uv.dtype),
                           torch.eye(3, dtype=uv.dtype), torch.zeros(3, dtype=uv.dtype))
    out.sum().backward()
    assert torch.allclose(uv.grad, torch.tensor([[3., 3.]], dtype=uv.dtype))
    assert torch.allclose(depth.grad, torch.tensor([0.8], dtype=uv.dtype))


def verify_production_empty_and_strided_input(camera_unproject):
    uv = torch.tensor([[2., 4., 99.], [3., 6., 99.]], dtype=torch.float32)[:, :2]
    assert not uv.is_contiguous()
    before = uv.clone()
    eye = torch.eye(3, dtype=uv.dtype)
    out = camera_unproject(uv, torch.tensor([2., 2.]), eye, eye, torch.zeros(3))
    assert out.dtype == uv.dtype and torch.allclose(out, torch.tensor([[4., 8., 2.], [6., 12., 2.]]))
    assert torch.equal(uv, before)
    empty = camera_unproject(uv[:0], torch.empty(0), eye, eye, torch.zeros(3))
    assert empty.shape == (0, 3) and empty.dtype == uv.dtype


def verify_production_reject_invalid_geometry(camera_unproject):
    uv = torch.zeros((1, 2))
    depth = torch.ones(1)
    eye = torch.eye(3)
    z = torch.zeros(3)
    invalid = [
        (uv, torch.zeros(1), eye, eye, z),
        (uv, depth, torch.zeros((3, 3)), eye, z),
        (uv, depth, eye.double(), eye, z),
        (uv, depth, eye, eye, torch.zeros(2)),
        (uv, depth, eye, eye, torch.tensor([float('nan'), 0., 0.])),
        (uv, depth[:0], eye, eye, z),
    ]
    for args in invalid:
        try:
            camera_unproject(*args)
        except ValueError:
            continue
        raise AssertionError('invalid unprojection input accepted')


# %%
from torch_judge import evaluate
if not evaluate('0051_camera_unproject_eval.py', camera_unproject):
    raise SystemExit(1)
