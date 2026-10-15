"""Sureal-inspired Resize Pixel Centers reference and judge run.

After installing from the repository root, run inside dojo/: python 0042_pixel_centers_eval.py
"""

# %%
from hypothesis import given, settings, strategies as st
import torch

# %%
def pixel_centers(points: torch.Tensor, original_hw: tuple[int, int], resized_hw: tuple[int, int], pad_xy: tuple[int, int], inverse: bool = False) -> torch.Tensor:
    if (points.ndim < 1 or points.shape[-1] != 2 or not points.is_floating_point()
            or not torch.isfinite(points).all()):
        raise ValueError("points must be finite floating coordinates [...,2]")
    if (len(original_hw) != 2 or len(resized_hw) != 2 or len(pad_xy) != 2
            or any(not isinstance(v, int) or v <= 0 for v in (*original_hw, *resized_hw))
            or any(not isinstance(v, int) for v in pad_xy)):
        raise ValueError("image sizes must be positive integer (height,width); padding integer (x,y)")
    scale = points.new_tensor((resized_hw[1] / original_hw[1], resized_hw[0] / original_hw[0]))
    pad = points.new_tensor(pad_xy)
    return (points - pad + 0.5) / scale - 0.5 if inverse else (points + 0.5) * scale - 0.5 + pad

# %%


# %% [verifiers]
def verify_baseline_half_pixel_and_padding(pixel_centers):
    p = torch.tensor([[0.0, 0.0], [3.0, 1.0]])
    out = pixel_centers(p, (3, 4), (6, 8), (1, 2))
    assert torch.allclose(out, torch.tensor([[1.5, 2.5], [7.5, 4.5]]))


def verify_followup_anisotropic_resize_and_inverse(pixel_centers):
    p = torch.tensor([[0.2, 1.3], [5.0, 2.0]], dtype=torch.float64)
    q = pixel_centers(p, (4, 8), (12, 4), (3, 1))
    assert torch.allclose(pixel_centers(q, (4, 8), (12, 4), (3, 1), inverse=True), p, atol=1e-12)


def verify_followup_empty_and_batched_coordinates(pixel_centers):
    points = torch.tensor([[[0., 0.], [1., 2.]], [[3., 4.], [5., 6.]]], dtype=torch.float64)
    resized = pixel_centers(points, (8, 10), (4, 20), (-2, 3))
    assert resized.shape == points.shape
    assert torch.allclose(pixel_centers(resized, (8, 10), (4, 20), (-2, 3), inverse=True), points)
    assert pixel_centers(points[:0], (8, 10), (4, 20), (0, 0)).shape == (0, 2, 2)


def verify_production_gradient_through_coordinates(pixel_centers):
    p = torch.tensor([[1.0, 2.0]], requires_grad=True)
    pixel_centers(p, (5, 5), (10, 15), (0, 0)).sum().backward()
    assert torch.allclose(p.grad, torch.tensor([[3.0, 2.0]]))


def verify_production_contract_and_immutability(pixel_centers):
    points = torch.tensor([[.25, 1.25]], dtype=torch.float64)
    original = points.clone()
    output = pixel_centers(points, (2, 3), (4, 9), (1, 2))
    assert output.dtype == points.dtype and output.device == points.device
    assert torch.equal(points, original)
    for args in ((points, (0, 3), (4, 9), (1, 2)), (points, (2, 3), (4, 0), (1, 2)),
                 (torch.zeros(2, 3), (2, 3), (4, 9), (1, 2))):
        try:
            pixel_centers(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid image geometry accepted")


def verify_production_generated_round_trip(pixel_centers):
    """Exercise anisotropic scales, offsets, and fractional coordinates."""
    @settings(max_examples=30, deadline=None, derandomize=True)
    @given(st.tuples(st.integers(1, 20), st.integers(1, 20)),
           st.tuples(st.integers(1, 20), st.integers(1, 20)),
           st.tuples(st.integers(-5, 5), st.integers(-5, 5)),
           st.lists(st.floats(-10, 10, allow_nan=False), min_size=2, max_size=2))
    def property_check(original_hw, resized_hw, pad_xy, xy):
        point = torch.tensor([xy], dtype=torch.float64)
        resized = pixel_centers(point, original_hw, resized_hw, pad_xy)
        recovered = pixel_centers(resized, original_hw, resized_hw, pad_xy, inverse=True)
        assert resized.shape == point.shape
        assert torch.allclose(recovered, point, atol=1e-10, rtol=1e-10)

    property_check()


# %%
print("Resize Pixel Centers: challenge the judge!")
from torch_judge import evaluate
if not evaluate('0042_pixel_centers_eval.py', pixel_centers):
    raise SystemExit(1)
