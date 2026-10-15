# %%
# 0076 · Where did the winning pixel come from?
# Goal: Max-pool an NCHW batch and return, for every pooled value, the
# coordinates of the winning pixel in the original input.
#
# CV02. The report mentions max pooling and selected coordinates. NCHW,
# valid windows, and row-major ties are authored practice rules.

# %%
import numpy as np
from torch_judge import check

# %%
# Predict: with kernel (2,2) and stride (1,1), the center 9 wins
# FOUR overlapping windows. Its coordinates are original input coordinates.
x_probe = np.array([[[[1., 2., 3.], [4., 9., 6.], [7., 8., 5.]]]])
print('Expected pooled values:', np.full((2, 2), 9.))
print('Expected winning coordinates:', np.full((2, 2, 2), (1, 1)))
# A backward pass must ADD four gradients at the same input coordinate.

# %% [drill]
# Window-local or original?
# The centre 9 wins all four overlapping $2 \times 2$ windows. Predict each window's flat `argmax` and the original coordinate it maps to.
import numpy as np
x = np.array([[1., 2., 3.], [4., 9., 6.], [7., 8., 5.]])
for row in range(2):
    for col in range(2):
        local = int(x[row:row + 2, col:col + 2].argmax())
        print((row, col), "local:", local, "original:", (row + local // 2, col + local % 2))

# %% [drill]
# Divide by which side?
# In a $2 \times 3$ window the maximum sits at flat index 4. Predict its (row, col) when you divide by the kernel width and when you divide by the kernel height.
import numpy as np
kh, kw = 2, 3
patch = np.array([[0., 1., 2.], [3., 9., 5.]])
index = int(patch.argmax())
print("index:", index, " divmod by kw:", divmod(index, kw), " divmod by kh:", divmod(index, kh))

# %% [drill]
# Which tie wins?
# Two cells tie for the maximum. Predict the winner when the window is scanned in row-major and in column-major order.
import numpy as np
patch = np.array([[1., 5.], [5., 0.]])
c = int(patch.ravel().argmax())
f = int(patch.ravel(order="F").argmax())
print("row-major:", divmod(c, 2), " column-major:", divmod(f, 2)[::-1])

# %%
# Contract: finite numeric x [B,C,H,W], positive integer kernel/stride pairs.
# Valid windows only; reject a kernel larger than the image. Return values
# [B,C,OH,OW] and integer (row,col) coordinates [B,C,OH,OW,2].
# OH = (H-kh)//sh+1, OW similarly. First winner in row-major order.

# %% [answer]
def max_pool2d_with_coordinates(
    x: np.ndarray, kernel: tuple[int, int] = (2, 2), stride: tuple[int, int] = (2, 2)
) -> tuple[np.ndarray, np.ndarray]:
    pass

# %%
# Baseline — $2 \times 2$ nonoverlapping windows and original coordinates.
check('0076_max_pool_coordinates_eval.py', stage='baseline')

# %%
# Follow-up — Overlapping windows, odd edges, negative values, and ties.
# Predict: why would returning a window-local index fail the probe above?
check('0076_max_pool_coordinates_eval.py', stage='followup')

# %%
# Production — Reject invalid size/input contracts; preserve batch/channel
# independence. A loop over output windows is reasonable; avoid a copy of
# every input window when the image becomes large.
check('0076_max_pool_coordinates_eval.py', stage='production')
