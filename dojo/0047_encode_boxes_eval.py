"""Sureal-inspired Encode 3D Detection Boxes reference and judge run.

After installing from the repository root, run inside dojo/: python 0047_encode_boxes_eval.py
"""

# %%
import torch

# %%
def encode_boxes(boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    if (boxes.ndim != 2 or boxes.shape[1] != 7 or boxes.shape != anchors.shape
            or not boxes.is_floating_point() or boxes.dtype != anchors.dtype or boxes.device != anchors.device
            or not torch.isfinite(boxes).all() or not torch.isfinite(anchors).all()
            or torch.any(boxes[:, 3:6] <= 0) or torch.any(anchors[:, 3:6] <= 0)):
        raise ValueError("paired positive-size (N, 7) boxes required")
    diagonal = torch.hypot(anchors[:, 3], anchors[:, 4])
    xy = (boxes[:, :2] - anchors[:, :2]) / diagonal[:, None]
    z = (boxes[:, 2:3] - anchors[:, 2:3]) / anchors[:, 5:6]
    sizes = torch.log(boxes[:, 3:6]) - torch.log(anchors[:, 3:6])
    yaw = boxes[:, 6:7] - anchors[:, 6:7]
    return torch.cat((xy, z, sizes, yaw), dim=1)

# %%


# %% [verifiers]
def verify_baseline_anchor_encodes_to_zero(encode_boxes):
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.2]])
    assert torch.allclose(encode_boxes(a, a), torch.zeros_like(a))


def verify_baseline_location_size_and_yaw_residuals(encode_boxes):
    a = torch.tensor([[0.0, 0.0, 0.0, 3.0, 4.0, 2.0, 3.0]])
    b = torch.tensor([[5.0, -5.0, 2.0, 6.0, 2.0, 4.0, -3.0]])
    out = encode_boxes(b, a)
    expected = torch.tensor([[1.0, -1.0, 1.0, 0.69314718056, -0.69314718056, 0.69314718056, -6.0]])
    assert torch.allclose(out, expected, atol=1e-06)


def verify_followup_batch_and_gradients(encode_boxes):
    a = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 1.0]])
    b = a.clone().requires_grad_()
    out = encode_boxes(b, a)
    assert out.shape == (2, 7)
    out.sum().backward()
    assert b.grad is not None and torch.isfinite(b.grad).all()


def verify_followup_empty_batch(encode_boxes):
    empty = torch.empty((0, 7), dtype=torch.float64)
    assert encode_boxes(empty, empty).shape == (0, 7)


def verify_production_reject_nonphysical_dimensions(encode_boxes):
    a = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    b = a.clone()
    b[0, 4] = 0.0
    try:
        encode_boxes(b, a)
    except ValueError:
        pass
    else:
        raise AssertionError('zero box width must fail')


def verify_production_extreme_ratios_and_immutability(encode_boxes):
    anchor = torch.tensor([[0., 0., 0., 1e-300, 2., 3., 3.]], dtype=torch.float64)
    box = torch.tensor([[0., 0., 0., 1e300, 2., 3., -3.]], dtype=torch.float64)
    original = box.clone(), anchor.clone()
    encoded = encode_boxes(box, anchor)
    assert torch.isfinite(encoded).all()
    assert torch.allclose(encoded[:, 3], torch.tensor([600. * 2.302585092994046], dtype=box.dtype))
    assert torch.equal(encoded[:, 6], torch.tensor([-6.], dtype=box.dtype))
    assert torch.equal(box, original[0]) and torch.equal(anchor, original[1])


def verify_production_invalid_inputs(encode_boxes):
    good = torch.tensor([[0., 0., 0., 1., 1., 1., 0.]])
    for bad in (torch.zeros(1, 6), torch.full((1, 7), float('nan')),
                torch.tensor([[0., 0., 0., 1., -1., 1., 0.]])):
        try:
            encode_boxes(bad, good)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid box accepted")


# %%
print("Encode 3D Detection Boxes: challenge the judge!")
from torch_judge import evaluate
if not evaluate('0047_encode_boxes_eval.py', encode_boxes):
    raise SystemExit(1)
