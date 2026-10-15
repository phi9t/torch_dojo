# %%
# 0024 · Rotary position embedding (adjacent feature pairs)
# Keep one function and revise it through three cumulative stages.

# %%
import torch

# %%
# Baseline — Pair [3,4] has norm 5. At position zero, angle is zero, so
# that pair is unchanged. At position p, pair i rotates by p/10000^(2i/D).
probe = torch.tensor([3., 4.])
print("norm:", probe.norm(), "position-zero pair:", probe)
# q,k [B,S,D], even D. Rotate (0,1), (2,3), ... and preserve shape.

# Derive why a rotation leaves the norm of [3,4] unchanged.

# %% [answer]
def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass

# %%
from torch_judge import check
check("0024_rope_eval.py", stage="baseline")

# %%
# Follow-up — Preserve each vector norm and pairwise relative-position dot
# products under a common sequence shift. Support non-contiguous q/k,
# arbitrary batches, and gradients through both outputs.
check("0024_rope_eval.py", stage="followup")

# %%
# Production — Reject incompatible shapes, odd D, mixed dtype/device, and
# non-floating inputs. Keep dtype/device, work with empty S, and form angle
# tables in float64 for float64 inputs. Avoid per-position Python loops.
check("0024_rope_eval.py", stage="production")
