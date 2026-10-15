"""Sureal-inspired Pinhole Camera Projection reference and judge run.

After installing from the repository root, run inside dojo/: python 0044_camera_project_eval.py
"""

# %%
import torch

# %%
def camera_project(points: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (points.ndim != 2 or points.shape[1] != 3 or K.shape != (3, 3)
            or R.shape != (3, 3) or t.shape != (3,) or not points.is_floating_point()
            or any(x.dtype != points.dtype or x.device != points.device for x in (K, R, t))
            or any(not torch.isfinite(x).all() for x in (points, K, R, t))
            or not torch.allclose(K[2], K.new_tensor([0., 0., 1.]))):
        raise ValueError("finite matching tensors and pinhole K with last row [0,0,1] required")
    camera = points @ R.T + t
    if torch.any(camera[:, 2] <= 0):
        raise ValueError("points must be in front of the camera")
    homogeneous = camera @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:3], camera[:, 2]

# %%


# %% [verifiers]
def verify_baseline_identity_camera(camera_project):
    p = torch.tensor([[2.0, 4.0, 2.0], [-3.0, 3.0, 3.0]])
    uv, depth = camera_project(p, torch.eye(3), torch.eye(3), torch.zeros(3))
    assert torch.allclose(uv, torch.tensor([[1.0, 2.0], [-1.0, 1.0]]))
    assert torch.allclose(depth, torch.tensor([2.0, 3.0]))


def verify_followup_intrinsics_and_translation(camera_project):
    K = torch.tensor([[100.0, 0.0, 20.0], [0.0, 50.0, 10.0], [0.0, 0.0, 1.0]])
    uv, depth = camera_project(torch.tensor([[0.0, 0.0, 2.0]]), K, torch.eye(3), torch.tensor([2.0, -2.0, 1.0]))
    assert torch.allclose(depth, torch.tensor([3.0]))
    assert torch.allclose(uv, torch.tensor([[20.0 + 200.0 / 3.0, 10.0 - 100.0 / 3.0]]), atol=1e-05)


def verify_followup_rotation_and_gradient(camera_project):
    p = torch.tensor([[1.0, 2.0, 4.0]], requires_grad=True)
    R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    uv, _ = camera_project(p, torch.eye(3), R, torch.zeros(3))
    assert torch.allclose(uv, torch.tensor([[-0.5, 0.25]]))
    uv.sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def verify_production_reject_points_behind_camera(camera_project):
    try:
        camera_project(torch.tensor([[0.0, 0.0, 0.0]]), torch.eye(3), torch.eye(3), torch.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError('nonpositive camera depth must fail')


def verify_production_empty_batch_and_shape(camera_project):
    p = torch.zeros((0, 3), dtype=torch.float64)
    uv, depth = camera_project(p, torch.eye(3, dtype=p.dtype), torch.eye(3, dtype=p.dtype), torch.zeros(3, dtype=p.dtype))
    assert uv.shape == (0, 2) and depth.shape == (0,)
    try:
        camera_project(torch.zeros(1, 2), torch.eye(3), torch.eye(3), torch.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid points shape accepted")
    p = torch.tensor([[0., 0., 1.]])
    for K in (torch.eye(3, dtype=torch.float64), torch.full((3, 3), float('nan')),
              torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 0., 1.]])):
        try:
            camera_project(p, K, torch.eye(3), torch.zeros(3))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid intrinsics accepted")


def verify_production_noncontiguous_input_and_no_mutation(camera_project):
    p = torch.tensor([[1., 2., 4., 9.], [3., -1., 2., 8.]], dtype=torch.float64)[:, :3]
    assert not p.is_contiguous()
    before = p.clone()
    K = torch.tensor([[2., 0., 10.], [0., 3., 20.], [0., 0., 1.]], dtype=p.dtype)
    uv, depth = camera_project(p, K, torch.eye(3, dtype=p.dtype), torch.zeros(3, dtype=p.dtype))
    assert torch.allclose(uv, torch.tensor([[10.5, 21.5], [13., 18.5]], dtype=p.dtype))
    assert torch.equal(depth, torch.tensor([4., 2.], dtype=p.dtype)) and torch.equal(p, before)


# %%
print("Pinhole Camera Projection: challenge the judge!")
from torch_judge import evaluate
if not evaluate('0044_camera_project_eval.py', camera_project):
    raise SystemExit(1)
