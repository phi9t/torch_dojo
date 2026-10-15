"""Two-view triangulation reference and cumulative checks."""

# %%
import torch

# %%
def triangulate_point(pixel1: torch.Tensor, pixel2: torch.Tensor,
                      K1: torch.Tensor, R1: torch.Tensor, t1: torch.Tensor,
                      K2: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    args = (pixel1, pixel2, K1, R1, t1, K2, R2, t2)
    if (pixel1.shape != (2,) or pixel2.shape != (2,) or K1.shape != (3, 3)
            or K2.shape != (3, 3) or R1.shape != (3, 3) or R2.shape != (3, 3)
            or t1.shape != (3,) or t2.shape != (3,)
            or pixel1.dtype not in (torch.float32, torch.float64)
            or any(x.dtype != pixel1.dtype or x.device != pixel1.device for x in args)
            or any(not torch.isfinite(x).all() for x in args)):
        raise ValueError('finite, matching camera tensors required')
    tol = 1e-5 if pixel1.dtype == torch.float32 else 1e-10
    last = pixel1.new_tensor([0., 0., 1.])
    if not torch.allclose(K1[2], last, atol=tol, rtol=0) or not torch.allclose(K2[2], last, atol=tol, rtol=0):
        raise ValueError('K must be pinhole intrinsics')
    try:
        ray1 = R1.T @ torch.linalg.solve(K1, torch.cat((pixel1, pixel1.new_ones(1))))
        ray2 = R2.T @ torch.linalg.solve(K2, torch.cat((pixel2, pixel2.new_ones(1))))
    except torch.linalg.LinAlgError as error:
        raise ValueError('K must be invertible') from error
    ray1 = ray1 / torch.linalg.vector_norm(ray1)
    ray2 = ray2 / torch.linalg.vector_norm(ray2)
    if torch.linalg.vector_norm(torch.linalg.cross(ray1, ray2)) <= (1e-4 if pixel1.dtype == torch.float32 else 1e-7):
        raise ValueError('near-parallel viewing rays')
    P1 = K1 @ torch.cat((R1, t1[:, None]), dim=1)
    P2 = K2 @ torch.cat((R2, t2[:, None]), dim=1)
    A = torch.stack((pixel1[0] * P1[2] - P1[0], pixel1[1] * P1[2] - P1[1],
                     pixel2[0] * P2[2] - P2[0], pixel2[1] * P2[2] - P2[1]))
    _, s, vh = torch.linalg.svd(A)
    if s[-2] <= tol * s[0] or vh[-1, 3].abs() <= tol:
        raise ValueError('degenerate triangulation')
    X = vh[-1, :3] / vh[-1, 3]
    if (R1 @ X + t1)[2] <= 0 or (R2 @ X + t2)[2] <= 0:
        raise ValueError('point must be in front of both cameras')
    return X

# %% [verifiers]
def verify_baseline_stereo_depth(triangulate_point):
    K = torch.diag(torch.tensor([100., 100., 1.], dtype=torch.float64))
    I = torch.eye(3, dtype=K.dtype)
    z = torch.zeros(3, dtype=K.dtype)
    p = triangulate_point(torch.tensor([10., -5.], dtype=K.dtype), torch.tensor([-10., -5.], dtype=K.dtype),
                          K, I, z, K, I, torch.tensor([-.4, 0., 0.], dtype=K.dtype))
    assert torch.allclose(p, torch.tensor([.2, -.1, 2.], dtype=K.dtype), atol=1e-10)


def verify_followup_rotated_camera_and_principal_point(triangulate_point):
    K = torch.tensor([[200., 0., 30.], [0., 150., 20.], [0., 0., 1.]], dtype=torch.float64)
    I = torch.eye(3, dtype=K.dtype)
    z = torch.zeros(3, dtype=K.dtype)
    R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=K.dtype)
    t = torch.tensor([-.5, .2, 0.], dtype=K.dtype)
    X = torch.tensor([.3, .4, 2.], dtype=K.dtype)
    def project(rotation, translation):
        h = K @ (rotation @ X + translation)
        return h[:2] / h[2]
    out = triangulate_point(project(I, z), project(R, t), K, I, z, K, R, t)
    assert torch.allclose(out, X, atol=1e-9)


def verify_production_near_parallel_and_behind(triangulate_point):
    K = torch.eye(3, dtype=torch.float64)
    I = K
    z = torch.zeros(3, dtype=K.dtype)
    cases = [
        (torch.tensor([0., 0.], dtype=K.dtype), torch.tensor([0., 0.], dtype=K.dtype),
         K, I, z, K, I, torch.tensor([-1., 0., 0.], dtype=K.dtype)),
        (torch.tensor([0., 0.], dtype=K.dtype), torch.tensor([1e-9, 0.], dtype=K.dtype),
         K, I, z, K, I, torch.tensor([-1., 0., 0.], dtype=K.dtype)),
        (torch.tensor([0., 0.], dtype=K.dtype), torch.tensor([1., 0.], dtype=K.dtype),
         K, I, z, K, I, torch.tensor([-1., 0., 0.], dtype=K.dtype)),
    ]
    for args in cases:
        try:
            triangulate_point(*args)
        except ValueError:
            continue
        raise AssertionError('degenerate or behind-camera point accepted')


def verify_production_validation_and_dtype(triangulate_point):
    K = torch.eye(3, dtype=torch.float32)
    I = K
    z = torch.zeros(3)
    p = torch.tensor([.2, 0.])
    q = torch.tensor([-.2, 0.])
    out = triangulate_point(p, q, K, I, z, K, I, torch.tensor([-.4, 0., 0.]))
    assert out.dtype == torch.float32 and torch.allclose(out, torch.tensor([.2, 0., 1.]), atol=1e-5)
    invalid = [
        (p[:1], q, K, I, z, K, I, z),
        (p, q.double(), K, I, z, K, I, z),
        (p, q, torch.zeros((3, 3)), I, z, K, I, z),
        (p, q, K, I, z, K, I, torch.tensor([float('nan'), 0., 0.])),
    ]
    for args in invalid:
        try:
            triangulate_point(*args)
        except ValueError:
            continue
        raise AssertionError('invalid camera input accepted')


# %%
from torch_judge import evaluate
if not evaluate('0052_triangulate_point_eval.py', triangulate_point):
    raise SystemExit(1)
