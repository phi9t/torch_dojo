"""Three-stage embedding lookup reference and checks."""
# %%
import torch
import torch.nn as nn

# %%
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        if not isinstance(num_embeddings, int) or not isinstance(embedding_dim, int) or num_embeddings < 1 or embedding_dim < 1:
            raise ValueError("positive integer vocabulary size and width required")
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices):
        if not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) or indices.device != self.weight.device:
            raise ValueError("integral indices on weight device required")
        if torch.any((indices < 0) | (indices >= self.weight.shape[0])):
            raise ValueError("index out of range")
        return self.weight[indices.long()]

# %% [verifiers]
def verify_baseline_registered_lookup(MyEmbedding):
    layer = MyEmbedding(5, 3)
    assert isinstance(layer, nn.Module) and isinstance(layer.weight, nn.Parameter)
    assert layer.weight.shape == (5, 3) and 'weight' in layer.state_dict()
    indices = torch.tensor([2, 0, 2])
    out = layer(indices)
    assert out.shape == (3, 3) and torch.equal(out, layer.weight[indices])


def verify_followup_repeated_gradient_and_rank(MyEmbedding):
    layer = MyEmbedding(6, 2)
    indices = torch.tensor([[1, 4], [1, 1]])
    out = layer(indices)
    assert out.shape == (2, 2, 2)
    out.sum().backward()
    grad = layer.weight.grad
    assert torch.equal(grad[1], torch.tensor([3., 3.]))
    assert torch.equal(grad[4], torch.tensor([1., 1.]))
    assert torch.equal(grad[0], torch.zeros(2))
    assert list(layer.parameters()) == [layer.weight]


def verify_production_empty_and_optimizer_step(MyEmbedding):
    layer = MyEmbedding(5, 2)
    empty = layer(torch.empty(0, 3, dtype=torch.long))
    assert empty.shape == (0, 3, 2)
    before = layer.weight.detach().clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=.1)
    layer(torch.tensor([2, 2])).sum().backward()
    optimizer.step()
    assert not torch.equal(layer.weight[2], before[2])
    assert torch.equal(layer.weight[0], before[0])


def verify_production_reject_invalid_dimensions_and_indices(MyEmbedding):
    for v, d in ((0, 3), (2, 0), (-1, 3)):
        try:
            MyEmbedding(v, d)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid embedding dimensions accepted")
    layer = MyEmbedding(3, 2)
    for indices in (torch.tensor([-1]), torch.tensor([3]), torch.tensor([1.0])):
        try:
            layer(indices)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid embedding index accepted")

# %%
from torch_judge import evaluate
if not evaluate('0018_embedding_eval.py', MyEmbedding):
    raise SystemExit(1)
