"""Reference and staged checks for historical points in the current ego frame."""
# Drill answers: current_from_history = inv(world_from_current) @ world_from_history,
# and a rigid inverse is [R^T | -R^T t] -- no general matrix inverse needed.

# %%
import numpy as np


# %%
def transform_history(points, world_from_history, world_from_current):
    points = np.asarray(points, dtype=np.float64)
    wh = np.asarray(world_from_history, dtype=np.float64)
    wc = np.asarray(world_from_current, dtype=np.float64)
    if (points.ndim != 4 or points.shape[-1] != 3 or wh.shape != (*points.shape[:1], points.shape[2], 4, 4)
            or wc.shape != (points.shape[0], 4, 4)):
        raise ValueError("need points[B,A,T,3], world_from_history[B,T,4,4], world_from_current[B,4,4]")
    Rh, th = wh[..., :3, :3], wh[..., :3, 3]  # [B,T,3,3], [B,T,3]
    Rc, tc = wc[..., :3, :3], wc[..., :3, 3]  # [B,3,3], [B,3]
    # current_from_history: R = Rc^T Rh, t = Rc^T (th - tc)   (rigid inverse, no np.linalg.inv)
    R = np.einsum("bji,btjk->btik", Rc, Rh)  # [B,T,3,3]
    t = np.einsum("bji,btj->bti", Rc, th - tc[:, None])  # [B,T,3]
    return np.einsum("btij,batj->bati", R, points) + t[:, None]  # [B,A,T,3]


# %% [verifiers]
def pose(theta=0.0, t=(0.0, 0.0, 0.0)):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


def random_pose(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    T = np.eye(4); T[:3, :3] = q; T[:3, 3] = rng.normal(size=3)
    return T


def verify_baseline_identity_and_worked_translation(transform_history):
    pts = np.random.default_rng(0).normal(size=(2, 3, 4, 3))
    eye_h, eye_c = np.tile(np.eye(4), (2, 4, 1, 1)), np.tile(np.eye(4), (2, 1, 1))
    assert np.allclose(transform_history(pts, eye_h, eye_c), pts)
    out = transform_history(np.array([[[[2., 0., 0.]]]]), pose(t=(10, 0, 0))[None, None], pose(t=(7, 0, 0))[None])
    assert out.shape == (1, 1, 1, 3) and np.allclose(out, [[[[5., 0., 0.]]]])


def verify_baseline_ninety_degree_rotation(transform_history):
    out = transform_history(np.array([[[[1., 0., 0.]]]]), pose(np.pi / 2)[None, None], np.eye(4)[None])
    assert np.allclose(out, [[[[0., 1., 0.]]]], atol=1e-12)
    back = transform_history(np.array([[[[0., 1., 0.]]]]), np.eye(4)[None, None], pose(np.pi / 2)[None])
    assert np.allclose(back, [[[[1., 0., 0.]]]], atol=1e-12)


def verify_followup_matches_explicit_matrix_inverse(transform_history):
    rng = np.random.default_rng(1)
    B, A, T = 2, 3, 4
    pts = rng.normal(size=(B, A, T, 3))
    wh = np.stack([[random_pose(rng) for _ in range(T)] for _ in range(B)])
    wc = np.stack([random_pose(rng) for _ in range(B)])
    out = transform_history(pts, wh, wc)
    homog = np.concatenate([pts, np.ones((B, A, T, 1))], axis=-1)
    for b in range(B):
        for t in range(T):
            expected = (np.linalg.inv(wc[b]) @ wh[b, t] @ homog[b, :, t].T).T[:, :3]
            assert np.allclose(out[b, :, t], expected, atol=1e-10)


def verify_followup_round_trip_recovers_points(transform_history):
    rng = np.random.default_rng(2)
    B, A, T = 2, 3, 3
    pts = rng.normal(size=(B, A, T, 3))
    wh = np.stack([[random_pose(rng) for _ in range(T)] for _ in range(B)])
    wc = np.stack([random_pose(rng) for _ in range(B)])
    out = transform_history(pts, wh, wc)
    for t in range(T):  # treat the current frame as the "history" and go back
        back = transform_history(out[:, :, t:t + 1], wc[:, None], wh[:, t])
        assert np.allclose(back[:, :, 0], pts[:, :, t], atol=1e-10)


def verify_production_pure_vectorized_and_batch_independent(transform_history):
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(3, 5, 4, 3))
    wh = np.stack([[random_pose(rng) for _ in range(4)] for _ in range(3)])
    wc = np.stack([random_pose(rng) for _ in range(3)])
    snapshot = [a.copy() for a in (pts, wh, wc)]
    out = transform_history(pts, wh, wc)
    for before, after in zip(snapshot, (pts, wh, wc)):
        assert np.array_equal(before, after)
    one = transform_history(pts[1:2], wh[1:2], wc[1:2])
    assert np.allclose(out[1:2], one) and out.dtype == np.float64
    import time
    big = rng.normal(size=(8, 512, 16, 3))
    bwh = np.tile(wh[:1, :1], (8, 16, 1, 1)); bwc = np.tile(wc[:1], (8, 1, 1))
    start = time.perf_counter()
    transform_history(big, bwh, bwc)
    assert time.perf_counter() - start < 2.0


def verify_production_rejects_mismatched_shapes(transform_history):
    pts, wh, wc = np.zeros((1, 2, 3, 3)), np.tile(np.eye(4), (1, 3, 1, 1)), np.eye(4)[None]
    for args in ((pts, wh[:, :2], wc), (pts, wh, np.eye(4)), (pts[..., :2], wh, wc),
                 (pts, np.tile(np.eye(3), (1, 3, 1, 1)), wc), (pts, wh, np.tile(np.eye(4), (2, 1, 1)))):
        try:
            transform_history(*args)
        except ValueError:
            continue
        raise AssertionError("invalid shapes accepted")


# %%
from torch_judge import evaluate
if not evaluate("0131_history_to_ego_frame_eval.py", transform_history):
    raise SystemExit(1)
