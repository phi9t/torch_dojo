"""Sureal-inspired Masked Semantic IoU reference and judge run.

After installing from the repository root, run inside dojo/: python 0043_semantic_iou_eval.py
"""

# %%
import torch

# %%
def semantic_iou(truth: torch.Tensor, prediction: torch.Tensor, support: torch.Tensor, num_classes: int) -> torch.Tensor:
    if (not isinstance(num_classes, int) or num_classes < 2 or truth.shape != prediction.shape
            or truth.shape != support.shape or truth.device != prediction.device
            or truth.device != support.device or truth.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or prediction.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or support.dtype != torch.bool or torch.any((truth < 0) | (truth >= num_classes))
            or torch.any((prediction < 0) | (prediction >= num_classes))):
        raise ValueError("matching integer labels and boolean support required; labels in [0,C)")
    eligible = support.bool() & (truth != 0)
    scores = []
    for cls in range(1, num_classes):
        t = (truth == cls) & eligible
        p = (prediction == cls) & eligible
        union = (t | p).sum()
        if union.item():
            scores.append((t & p).sum().to(torch.float32) / union)
    return torch.stack(scores).mean() if scores else torch.zeros((), device=truth.device)

# %%


# %% [verifiers]
def verify_baseline_missing_prediction_counts_as_false_negative(semantic_iou):
    truth = torch.tensor([1, 1, 2, 0])
    pred = torch.tensor([1, 2, 0, 2])
    out = semantic_iou(truth, pred, torch.ones(4, dtype=torch.bool), 4)
    assert torch.allclose(out, torch.tensor((0.5 + 0.0) / 2))


def verify_baseline_support_excludes_pixels(semantic_iou):
    truth = torch.tensor([1, 2, 2])
    pred = torch.tensor([1, 1, 0])
    out = semantic_iou(truth, pred, torch.tensor([True, False, False]), 3)
    assert torch.allclose(out, torch.tensor(1.0))


def verify_followup_no_eligible_classes(semantic_iou):
    out = semantic_iou(torch.tensor([0, 0]), torch.tensor([1, 2]), torch.ones(2, dtype=torch.bool), 3)
    assert out.ndim == 0 and out.item() == 0.0


def verify_followup_predicted_only_class_enters_mean(semantic_iou):
    out = semantic_iou(torch.tensor([1, 1]), torch.tensor([1, 2]), torch.ones(2, dtype=torch.bool), 4)
    assert torch.allclose(out, torch.tensor(0.25))


def verify_production_empty_and_invalid_labels(semantic_iou):
    empty = torch.zeros((0, 2), dtype=torch.int64)
    out = semantic_iou(empty, empty, torch.zeros_like(empty, dtype=torch.bool), 3)
    assert out.ndim == 0 and out.item() == 0
    for pred in (torch.tensor([-1]), torch.tensor([3])):
        try:
            semantic_iou(torch.tensor([1]), pred, torch.tensor([True]), 3)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range class accepted")
    for bad_support, bad_classes in ((torch.tensor([1]), 3), (torch.tensor([True]), 1)):
        try:
            semantic_iou(torch.tensor([1]), torch.tensor([1]), bad_support, bad_classes)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid support or class count accepted")


def verify_production_shape_and_no_mutation(semantic_iou):
    truth = torch.tensor([[1, 2], [0, 1]])
    pred = torch.tensor([[2, 2], [1, 0]])
    support = torch.tensor([[True, True], [False, True]])
    before = truth.clone(), pred.clone(), support.clone()
    score = semantic_iou(truth, pred, support, 3)
    assert torch.allclose(score, torch.tensor((0. + .5) / 2))
    assert all(torch.equal(x, y) for x, y in zip((truth, pred, support), before))
    try:
        semantic_iou(truth, pred.reshape(-1), support, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched shapes accepted")


# %%
print("Masked Semantic IoU: challenge the judge!")
from torch_judge import evaluate
if not evaluate('0043_semantic_iou_eval.py', semantic_iou):
    raise SystemExit(1)
