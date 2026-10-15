"""Temporal ego transform reference and cumulative checks."""

# %%
import numpy as np

# %%
# Drill 1: The named order prints `[3. 1. 0.]`; the reversed product prints
# `[ 5. -1.  0.]`. A matrix product applies right to left. In
# `current_from_world @ world_from_t @ p` the point is first read in the ego
# frame at time $t$ and sent to the world, $(5, 1, 0)$, then the current ego's
# offset of 2 m is removed. Reversed, the current-ego inverse acts on a point
# that is not yet in the world frame, and the old ego's yaw then turns that
# wrong offset: the frame names no longer meet in the middle.
#
# Drill 2: It prints `[1. 0. 0.] [-1. -2.  0.]`. A rigid pose maps
# $w = R\,e + d$, so solving for the ego point gives $e = R^T (w - d)$: undo
# the translation first, in world axes, then undo the rotation. The world point
# is 1 m along world $+y$ from the ego. The ego's $+x$ axis points forward, and
# a counter-clockwise yaw of $+\pi/2$ turns it onto world $+y$, so the point is
# straight ahead: $(1, 0, 0)$. $R^T w - d$ subtracts a world-axis offset from a
# vector already in ego axes, so it is only right when $R = I$; the inverse
# translation is $-R^T d$, not $-d$.
#
# Drill 3: Multiplying by the mask prints `[[ 1.  2.  3.] [nan nan nan]]`, while
# `np.where` prints `[[1. 2. 3.] [0. 0. 0.]]`. IEEE arithmetic defines
# $0 \cdot \text{NaN} = \text{NaN}$, so a 0/1 mask cannot erase a NaN, and the
# rotation and translation keep it NaN. `np.where` SELECTS a value instead of
# computing with the padded one, so selecting zeros keeps padding out of the
# output.

# %%
def to_current_ego(trajectories: np.ndarray, world_from_t: np.ndarray,
                   world_from_current: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
        raise ValueError('trajectories must be [B,A,T,3]')
    B, A, T, _ = trajectories.shape
    if (world_from_t.shape != (B, T, 4, 4) or world_from_current.shape != (B, 4, 4)
            or valid.shape != (B, A, T) or valid.dtype != np.bool_
            or trajectories.dtype not in (np.float32, np.float64)
            or world_from_t.dtype != trajectories.dtype
            or world_from_current.dtype != trajectories.dtype):
        raise ValueError('pose/mask shapes or dtypes mismatch')
    if (not np.isfinite(world_from_t).all() or not np.isfinite(world_from_current).all()
            or not np.isfinite(trajectories[valid]).all()):
        raise ValueError('valid points and poses must be finite')
    eye = np.eye(3, dtype=trajectories.dtype)
    last = np.array([0., 0., 0., 1.], dtype=trajectories.dtype)
    tol = 1e-5 if trajectories.dtype == np.float32 else 1e-8
    for poses in (world_from_t, world_from_current):
        R = poses[..., :3, :3]
        if (not np.allclose(np.swapaxes(R, -1, -2) @ R, eye, atol=tol, rtol=0)
                or not np.allclose(np.linalg.det(R), 1., atol=tol, rtol=0)
                or not np.allclose(poses[..., 3, :], last, atol=tol, rtol=0)):
            raise ValueError('poses must be proper rigid transforms')
    points = np.where(valid[..., None], trajectories, np.zeros((), dtype=trajectories.dtype))
    R_t = world_from_t[..., :3, :3]
    t_t = world_from_t[..., :3, 3]
    world = np.einsum('btij,batj->bati', R_t, points) + t_t[:, None]
    centered = world - world_from_current[:, None, None, :3, 3]
    R_current = world_from_current[:, :3, :3]
    out = np.einsum('bji,batj->bati', R_current, centered)
    return np.where(valid[..., None], out, np.zeros((), dtype=trajectories.dtype))

# %% [verifiers]
def verify_baseline_identity_and_mask(to_current_ego):
    p = np.array([[[[1., 2., 3.], [99., 99., 99.]]]])
    poses = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
    current = np.eye(4)[None]
    out = to_current_ego(p, poses, current, np.array([[[True, False]]]))
    assert np.array_equal(out, np.array([[[[1., 2., 3.], [0., 0., 0.]]]]))


def verify_followup_rotation_translation_order(to_current_ego):
    p = np.array([[[[1., 0., 0.]]]])
    poses = np.eye(4)[None, None].copy()
    poses[0, 0, :2, :2] = [[0., -1.], [1., 0.]]
    poses[0, 0, 0, 3] = 5.
    current = np.eye(4)[None].copy()
    current[0, 0, 3] = 2.
    out = to_current_ego(p, poses, current, np.ones((1, 1, 1), bool))
    assert np.allclose(out[0, 0, 0], [3., 1., 0.])


def verify_followup_multiple_actors_and_time_steps(to_current_ego):
    p = np.array([[[[1., 0., 0.], [2., 0., 0.]],
                   [[3., 0., 0.], [4., 0., 0.]]]])
    poses = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
    poses[0, 1, 0, 3] = 10.
    current = np.eye(4)[None]
    out = to_current_ego(p, poses, current, np.ones((1, 2, 2), bool))
    assert np.array_equal(out[..., 0], np.array([[[1., 12.], [3., 14.]]]))


def verify_production_masked_nan_float32_and_strides(to_current_ego):
    base = np.array([[[[1., 2., 3., 99.], [np.nan, np.nan, np.nan, 99.]]]], dtype=np.float32)
    p = base[..., :3]
    before = p.copy()
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (1, 2, 4, 4)).copy()
    out = to_current_ego(p, poses, np.eye(4, dtype=np.float32)[None], np.array([[[True, False]]]))
    assert out.dtype == np.float32 and np.array_equal(out[0, 0, 0], [1., 2., 3.])
    assert np.array_equal(out[0, 0, 1], [0., 0., 0.])
    assert np.array_equal(p, before, equal_nan=True)
    empty = to_current_ego(p[:, :0, :0], poses[:, :0], np.eye(4, dtype=np.float32)[None],
                           np.empty((1, 0, 0), bool))
    assert empty.shape == (1, 0, 0, 3) and empty.dtype == np.float32


def verify_production_invalid_inputs(to_current_ego):
    p = np.zeros((1, 1, 1, 3))
    poses = np.eye(4)[None, None]
    current = np.eye(4)[None]
    valid = np.ones((1, 1, 1), bool)
    reflected = poses.copy()
    reflected[0, 0, 0, 0] = -1.
    cases = [
        (p[..., :2], poses, current, valid),
        (p, poses, current, valid.astype(int)),
        (p, reflected, current, valid),
        (p, poses.astype(np.float32), current, valid),
        (np.full_like(p, np.nan), poses, current, valid),
    ]
    for args in cases:
        try:
            to_current_ego(*args)
        except ValueError:
            continue
        raise AssertionError('invalid temporal geometry accepted')


# %%
from torch_judge import evaluate
if not evaluate('0065_temporal_transforms_eval.py', to_current_ego):
    raise SystemExit(1)
