# %%
# 0018 · Embedding lookup and sparse row updates

# %%
import torch
import torch.nn as nn

# %%
# Worked probe: each integer index appends the embedding width.
weight_probe = torch.arange(12).reshape(4, 3)
index_probe = torch.tensor([[2, 0], [1, 2]])
print("lookup shape:", weight_probe[index_probe].shape)

# Predict which weight rows receive gradients for indices [[2,0],[1,2]].

# %% [answer]
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # Replace this body; self.weight must be an nn.Parameter [V,D].

    def forward(self, indices):
        pass

# %%
# Baseline: trainable weight [V,D]; integer lookup returns
# indices.shape + (D,). No nn.Embedding.
from torch_judge import check
check("0018_embedding_eval.py", stage="baseline")

# %%
# Follow-up: multidimensional and repeated indices. The gradient for
# a repeated row adds contributions; unused rows get zero gradient.
# The parameter must appear in module.parameters() and state_dict().
check("0018_embedding_eval.py", stage="followup")

# %%
# Production: reject nonpositive V/D, nonintegral indices, negative
# and out-of-range indices (Python indexing otherwise wraps negatives).
# Support empty index tensors and same-device execution. An optimizer
# step should change only rows addressed by the loss.
check("0018_embedding_eval.py", stage="production")
