# NumPy and PyTorch practice cheatsheet

For workouts **0125–0131** and the [interactive fundamentals page](https://phi9t.github.io/torch_dojo/fundamentals/).
Practice the first four in order; use the last three as extensions.
Read a pattern, close it, then reproduce it with a tiny example.

The kernels below assume the documented shapes and valid scalar arguments.
Start with float64; keep masks boolean and indices integer. The question scripts
also specify input validation. `np` means `numpy`; `torch` means PyTorch.

## The seven patterns

| Workout | Shape flow | Remember |
| --- | --- | --- |
| [0125 Attention](../../dojo/0125_masked_self_attention.py) · 25 min | `[B,T,D] → [B,H,T,dh] → [B,H,T,T] → [B,T,D]` | Project, split, scale, mask, safe softmax, mix values, merge, project |
| [0126 Contrastive](../../dojo/0126_contrastive_loss.py) · 20 min | `[N,D] → [n,D] → [n,n] → scalar` | Filter both sets, normalize, diagonal positives, average both directions |
| [0127 Conv2D](../../dojo/0127_conv2d_scratch.py) · 20 min | `[B,Cin,H,W] → [B,Cout,Ho,Wo]` | Cross-correlation; sum over input channels and kernel taps |
| [0128 IoU](../../dojo/0128_box_iou_grid.py) · 15 min | `[N,1,4]` with `[1,M,4] → [N,M]` | Clamp each overlap side before multiplying; guard zero union |
| [0129 Huber](../../dojo/0129_masked_huber_grad.py) · 10 min | `[N,D] → scalar + [N,D]` | Residual = prediction − target; divide by valid rows × D |
| [0130 Voxels](../../dojo/0130_voxel_feature_mean.py) · 15 min | `[N,3] → keys → K groups → [K,C]` | Floor, bounds, unique/inverse, scatter sums, divide by counts |
| [0131 Ego frames](../../dojo/0131_history_to_ego_frame.py) · 10 min | `[B,A,T,3] → [B,A,T,3]` | History → world → inverse current; `R⁻¹ = Rᵀ` |

`H` means heads in attention and image height in Conv2D. `dh = D // H`.
`n` is the number of valid contrastive pairs; `K` is the number of occupied voxels.

## API translation card

| Operation | NumPy | PyTorch |
| --- | --- | --- |
| Float64 | `np.asarray(x, dtype=np.float64)` | `x.to(dtype=torch.float64)` for a tensor |
| Allocate beside x | `np.zeros_like(x)` | `torch.zeros_like(x)` |
| New shape, same dtype/device | `np.zeros(shape, dtype=x.dtype)` | `x.new_zeros(shape)` |
| Copy | `x.copy()` | `x.clone()` |
| Reshape | `x.reshape(B,T,H,dh)` | `x.reshape(B,T,H,dh)` |
| Reorder all axes | `x.transpose(0,2,1,3)` | `x.permute(0,2,1,3)` |
| Swap last two axes | `x.swapaxes(-1,-2)` | `x.transpose(-1,-2)` |
| Add singleton axes | `x[:,None,:,None]` | Same indexing syntax |
| Reduce, retain axis | `x.sum(axis=-1, keepdims=True)` | `x.sum(dim=-1, keepdim=True)` |
| Maximum values along axis | `x.max(axis=-1, keepdims=True)` | `x.amax(dim=-1, keepdim=True)` |
| Any / all along axis | `x.any(axis=-1)` / `x.all(axis=-1)` | `x.any(dim=-1)` / `x.all(dim=-1)` |
| Elementwise max / min | `np.maximum(x,y)` / `np.minimum(x,y)` | `torch.maximum(x,y)` / `torch.minimum(x,y)` |
| Clamp lower bound | `np.maximum(x,0)` | `x.clamp_min(0)` |
| Clamp interval | `np.clip(x,-d,d)` | `x.clamp(-d,d)` |
| Select values | `np.where(mask,x,y)` | `torch.where(mask,x,y)` |
| Select valid rows | `x[valid]` | `x[valid]` |
| Last-axis L2 norm | `np.linalg.norm(x,axis=-1,keepdims=True)` | `torch.linalg.vector_norm(x,dim=-1,keepdim=True)` |
| Diagonal of 2D matrix | `np.diag(x)` | `x.diagonal()` |
| Explicit contraction | `np.einsum("btij,batj->bati",R,p)` | `torch.einsum("btij,batj->bati",R,p)` |
| Group integer keys | `np.unique(k,return_inverse=True,return_counts=True)` | `torch.unique(k,sorted=True,return_inverse=True,return_counts=True)` |
| Add rows into repeated destinations | `np.add.at(out,idx,src)` | `out.index_add_(0,idx,src)` |
| Compare numerically | `np.testing.assert_allclose(a,b)` | `torch.testing.assert_close(a,b)` |

**Axis traps:** NumPy `.T` reverses every axis, so reserve it for 2D matrices here.
PyTorch `transpose` swaps two axes; `permute` specifies the whole order.
`torch.max(x, dim=...)` returns values **and indices**; `amax` returns values.
Use `reshape` after an axis permutation: `view` can fail on a non-contiguous tensor.
See [NumPy transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
and [PyTorch tensor operations](https://docs.pytorch.org/docs/stable/tensors.html).

**Broadcasting:** dimensions align from the right; each pair must be equal or
contain a 1. `@` contracts the last two matrix axes and broadcasts the leading axes.
`reshape` regroups elements; it does not perform an axis permutation.

**Masking:** both `where` APIs receive already-computed branches. Thus
`where(mask, numerator / denominator, 0)` can still compute `0/0`.
Make the operands safe **before** dividing. In PyTorch, an invalid operation in
an unselected branch can also corrupt gradients.

## 0125 Masked multi-head self-attention

Inputs: `X[B,T,D]`, four weights `[D,D]`, `valid_mask[B,T]`, positive
`num_heads` dividing D, and a causal flag. Return output `[B,T,D]` and
weights `[B,H,T,T]`. The projection convention is **`Q = X @ Wq`**, with no biases.
For `nn.Linear`, the equivalent weight is `Wq.T` because the layer computes `X @ weight.T`.

```python
def self_attention(X, Wq, Wk, Wv, Wo, num_heads, valid_mask, causal=False):
    B, T, D = X.shape
    H, dh = num_heads, D // num_heads
    def split(W):
        return (X @ W).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    q, k, v = split(Wq), split(Wk), split(Wv)       # [B,H,T,dh]
    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(dh)

    allowed = valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
    # [B,1,T,T]: valid query ROWS AND valid key COLUMNS
    if causal:
        allowed = allowed & np.tril(np.ones((T, T), dtype=bool))
    weights = safe_softmax(scores, allowed)
    context = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return context @ Wo, weights

def safe_softmax(scores, allowed):
    has_key = allowed.any(axis=-1, keepdims=True)
    masked = np.where(allowed, scores, -np.inf)
    safe = np.where(has_key, masked, 0.0)         # prevent -inf - (-inf)
    e = np.where(allowed, np.exp(safe - safe.max(axis=-1, keepdims=True)), 0.0)
    denom = np.where(has_key, e.sum(axis=-1, keepdims=True), 1.0)
    return e / denom                             # empty row: 0/1 = 0
```

PyTorch: use `.permute(0,2,1,3)` for split/merge, `k.transpose(-1,-2)` for
scores, and `dh ** 0.5` for scaling. Create the causal mask on `X.device`.
The softmax pattern translates directly:

```python
def torch_safe_softmax(scores, allowed):
    has_key = allowed.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~allowed, -torch.inf)
    safe = torch.where(has_key, masked, 0.0)
    e = torch.where(allowed, (safe - safe.amax(dim=-1, keepdim=True)).exp(), 0.0)
    denom = torch.where(has_key, e.sum(dim=-1, keepdim=True), 1.0)
    return e / denom
```

Check: with `Wq=Wk=0`, `Wv=Wo=I`, one head, and `X=[[[1,0],[0,2]]]`,
outputs are `[[[0.5,1],[0.5,1]]]`; causal outputs are `[[[1,0],[0.5,1]]]`.
An all-invalid batch element has zero outputs and weights.

Remember: scale by **sqrt(dh)**; mask both queries and keys; merge heads before
`Wo`. Zero weights times finite values produce zero context. If adding an output
bias in another problem, explicitly zero invalid outputs after that projection.

## 0126 Symmetric contrastive loss

Inputs: `X,Y[N,D]`, positive temperature, boolean valid pairs `[N]`.
Filter rows from **both** sets before normalization. A positive pair is diagonal
entry `(i,i)` in the compacted score matrix.

```python
def contrastive_loss(X, Y, temperature, valid_mask, eps=1e-12):
    x, y = X[valid_mask], Y[valid_mask]            # [n,D]
    if len(x) == 0:
        return 0.0
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
    y = y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), eps)
    S = x @ y.T / temperature                    # [n,n]
    def ce(L):
        m = L.max(axis=-1, keepdims=True)
        shifted = L - m
        return (np.log(np.exp(shifted).sum(axis=-1)) - np.diag(shifted)).mean()
    return 0.5 * (ce(S) + ce(S.T))
```

PyTorch normalization uses
`torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)`.
The same explicit CE computation is:

```python
def torch_diagonal_ce(L):
    shifted = L - L.amax(dim=-1, keepdim=True)
    return (shifted.exp().sum(dim=-1).log() - shifted.diagonal()).mean()
```

`torch.logsumexp(L, dim=-1) - L.diagonal()` is the built-in equivalent when
allowed; rehearse the explicit version above. For no valid pairs, a PyTorch
loss connected to the inputs can be `(X[valid_mask].sum() + Y[valid_mask].sum()) * 0`.

Check: `X=Y=eye(2)`, temperature 1 → **0.3132616875**. Identical embeddings
→ `log(n)`; one valid pair → 0. **`log(n)` is a uniform-guess baseline, not an upper bound.**

Gradient extension: for `P=softmax(S, axis=1)` and
`Q=softmax(S.T, axis=1).T`, the symmetric loss has
`dL/dS = (P + Q - 2I) / (2n)`. These are gradients of temperature-scaled
logits; gradients of unscaled similarities gain another `1/temperature`.
For norm above eps, normalization removes the gradient component parallel to
the unit embedding and divides by the original norm.

## 0127 Conv2D from scratch

Use the [focused Conv2D cheatsheet](conv2d-cheatsheet.md) for the direct-window
implementation, slice indexing, im2col layout, API translations and memory estimates.

Inputs: `X[batch_size,input_channels,input_height,input_width]`, kernel `W[output_channels,input_channels,kernel_height,kernel_width]`, bias `[output_channels]`.
Symmetric padding `(padding_height,padding_width)`, strides `(stride_height,stride_width)`, dilation/groups = 1.

`output_height = (input_height + 2 * padding_height - kernel_height) // stride_height + 1`; `output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1`.
Cross-correlation multiplies the window by the kernel **without flipping it**.

```python
def conv2d(X, W, bias, stride=(1, 1), padding=(0, 0)):
    batch_size, input_channels, input_height, input_width = X.shape
    output_channels, _, kernel_height, kernel_width = W.shape
    stride_height, stride_width = stride; padding_height, padding_width = padding
    output_height, output_width = (input_height + 2*padding_height - kernel_height)//stride_height + 1, (input_width + 2*padding_width - kernel_width)//stride_width + 1
    Xp = np.pad(X, ((0,0), (0,0), (padding_height,padding_height), (padding_width,padding_width)))
    out = np.zeros((batch_size, output_channels, output_height, output_width), dtype=X.dtype) + bias[None,:,None,None]
    for i in range(kernel_height):
        for j in range(kernel_width):
            patch = Xp[:, :, i:i+stride_height*output_height:stride_height, j:j+stride_width*output_width:stride_width]  # [batch_size,input_channels,output_height,output_width]
            out += np.einsum('bchw,oc->bohw', patch, W[:, :, i, j])
    return out
```

PyTorch uses `torch.nn.functional.pad(X, (padding_width,padding_width,padding_height,padding_height))`: the padding tuple
starts with the **last spatial axis**. This is a padding helper, not a convolution
helper. Allocate with `X.new_zeros((batch_size,output_channels,output_height,output_width))`; keep the same `torch.einsum`.

Patch extraction alternative:

```python
# NumPy; Xp is already padded
patches = np.lib.stride_tricks.sliding_window_view(Xp, (kernel_height, kernel_width), axis=(-2, -1))
patches = patches[:, :, ::stride_height, ::stride_width]              # [batch_size,input_channels,output_height,output_width,kernel_height,kernel_width]
out = np.einsum('bchwij,ocij->bohw', patches, W) + bias[None,:,None,None]
```

PyTorch's corresponding view is `Xp.unfold(2,kernel_height,stride_height).unfold(3,kernel_width,stride_width)`;
contract with the same einsum string. Both extraction patterns expose overlapping
views. Treat them as read-only. Flattening them into an im2col matrix may allocate
`batch_size*output_height*output_width*input_channels*kernel_height*kernel_width` elements. See the
[NumPy window API](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html)
and [PyTorch unfold API](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unfold.html).

Check: input `1..9` in a 3×3 image and kernel `[[1,0],[0,-1]]` → a 2×2
output of **−4**. A 1×1 kernel is a per-pixel channel projection.

## 0128 Pairwise box IoU

Inputs: `a[N,4]`, `b[M,4]`, ordered `[xmin,ymin,xmax,ymax]` in continuous
coordinates. There is **no +1** in width or height.

```python
def pairwise_iou(a, b):
    lo = np.maximum(a[:,None,:2], b[None,:,:2])    # [N,M,2]
    hi = np.minimum(a[:,None,2:], b[None,:,2:])
    inter = np.maximum(hi - lo, 0).prod(axis=-1)
    aa = np.maximum(a[:,2:] - a[:,:2], 0).prod(axis=-1)
    ab = np.maximum(b[:,2:] - b[:,:2], 0).prod(axis=-1)
    union = aa[:,None] + ab[None,:] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
```

PyTorch: `torch.maximum/minimum`, `.clamp_min(0)`, and `.prod(dim=-1)`.
For the final division, protect the denominator first:

```python
ok = union > 0
denom = torch.where(ok, union, torch.ones_like(union))
iou = torch.where(ok, inter / denom, torch.zeros_like(inter))
```

Check: `[0,0,2,2]` against `[1,1,3,3]` → **1/7**. Touching edges and zero
union → 0. Empty inputs retain shape `[N,M]`. Swapping inputs transposes the result.
For strictly disjoint boxes, `1−IoU` is locally flat in position; it gives no
direction toward the target.

## 0129 Masked Huber loss and manual gradient

Inputs: prediction/target `[N,D]`, valid rows `[N]`, positive delta.
`r = prediction - target`; average over **valid rows × D**, not all N rows.

`loss(r) = 0.5*r²` if `|r| ≤ delta`, else `delta*(|r| - 0.5*delta)`.
`d loss / dr = clip(r, -delta, delta)` before mean reduction.

```python
def huber_loss_and_grad(prediction, target, valid_mask, delta):
    count = int(valid_mask.sum()) * prediction.shape[1]
    if count == 0:
        return 0.0, np.zeros_like(prediction)
    p = np.where(valid_mask[:,None], prediction, 0.0)
    t = np.where(valid_mask[:,None], target, 0.0)
    r = p - t                                   # discard invalid garbage before math
    per = np.where(np.abs(r) <= delta, 0.5*r*r, delta*(np.abs(r)-0.5*delta))
    return per.sum()/count, np.clip(r, -delta, delta)/count
```

PyTorch uses `torch.where`, `r.abs()`, and `r.clamp(-delta,delta)`.
Sanitizing the inputs first also protects autograd when invalid rows contain NaNs.
For a differentiable zero loss on all-invalid input, use
`prediction[valid_mask].sum() * 0`; return `torch.zeros_like(prediction)` as
the explicit manual gradient. For a differentiable loss with D=0, the same empty
sum works. Keep the scalar denominator positive in the nonempty branch.

Check: residuals `[[-2,0.5]]`, delta 1 → loss **0.8125**, gradient
**[[-0.5,0.25]]**. Check the gradient using float64 central differences:
`(L(p+h*e_i) - L(p-h*e_i)) / (2*h)`, with `h=1e-6`.

## 0130 Point-to-voxel feature averaging

Inputs: finite points `[N,3]`, finite features `[N,C]`, origin `[3]`,
positive voxel size, positive integer grid `(gx,gy,gz)`.
Return lexicographically sorted coordinates `[K,3]`, means `[K,C]`, counts `[K]`.

```python
def voxel_mean(points, features, origin, voxel_size, grid_shape):
    cell = np.floor((points - origin) / voxel_size)
    inside = ((cell >= 0) & (cell < np.asarray(grid_shape))).all(axis=1)
    idx = cell[inside].astype(np.int64)           # bounds BEFORE integer cast
    C = features.shape[1]
    if len(idx) == 0:
        return np.empty((0,3), np.int64), np.empty((0,C)), np.empty(0, np.int64)
    key = np.ravel_multi_index(idx.T, grid_shape)
    keys, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.zeros((len(keys), C), dtype=features.dtype)
    np.add.at(sums, inverse, features[inside])     # repeated indices accumulate
    coords = np.stack(np.unravel_index(keys, grid_shape), axis=1)
    return coords, sums / counts[:,None], counts
```

PyTorch grouping pattern, after the same float bounds check:

```python
idx = cell[inside].to(torch.int64)
gx, gy, gz = grid_shape
key = (idx[:,0] * gy + idx[:,1]) * gz + idx[:,2]
keys, inverse, counts = torch.unique(key, sorted=True, return_inverse=True, return_counts=True)
sums = features.new_zeros((keys.numel(), features.shape[1]))
sums.index_add_(0, inverse, features[inside])
means = sums / counts[:,None]
coords = torch.stack((keys // (gy*gz), (keys // gz) % gy, keys % gz), dim=1)
```

Keep the shape-bound tensor on `points.device`. The flat-key method assumes the
grid's address range fits int64. Sorted flat keys give x, then y, then z order.
`inverse[i]` is the compact group id for kept point i; `counts[k]` is that group's size.

**Collision trap:** `sums[inverse] += features[inside]` can lose contributions
when indices repeat. Use [NumPy add.at](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)
or [PyTorch index_add_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html).
For general scatter axes, the equivalent is
`sums.scatter_add_(0, inverse[:,None].expand_as(features[inside]), features[inside])`.

Check: x positions `[0.1,0.9,1.0,2.0]`, features `[2,4,8,99]`, origin 0,
size 1, grid `(2,1,1)` → coords `[[0,0,0],[1,0,0]]`, means `[[3],[8]]`, counts `[2,1]`.
The upper boundary is excluded. `floor(-0.1)=-1`; truncating to int gives 0.

## 0131 Historical points into the current ego frame

Inputs: points `[B,A,T,3]`, historical world poses `[B,T,4,4]`, current world
pose `[B,4,4]`. Poses use column vectors: `p_world = R @ p_local + t`.

`current_from_history = inverse(world_from_current) @ world_from_history`.
For a rigid pose, the inverse is `[Rᵀ | -Rᵀt]`.

```python
def transform_history(points, world_from_history, world_from_current):
    Rh = world_from_history[..., :3, :3]          # [B,T,3,3]
    th = world_from_history[..., :3, 3]           # [B,T,3]
    Rc = world_from_current[..., :3, :3]          # [B,3,3]
    tc = world_from_current[..., :3, 3]           # [B,3]
    R = np.einsum('bji,btjk->btik', Rc, Rh)       # Rc.T @ Rh
    t = np.einsum('bji,btj->bti', Rc, th - tc[:,None])
    return np.einsum('btij,batj->bati', R, points) + t[:,None]
```

The PyTorch version uses the exact same indexing and `torch.einsum` equations.
Read `btij,batj->bati` as: batch/time select a rotation; sum over input coordinate
j; preserve agent a and emit output coordinate i.

Check: history x=10, current x=7, historical point `[2,0,0]` → **[5,0,0]**.
History +90° about z, current identity, point `[1,0,0]` → **[0,1,0]**.
Transform out and back to catch a reversed composition or translation sign.

## Complexity to say aloud

Memory includes outputs and extra arrays, excludes inputs, and assumes float64
with fixed-size 3D transforms. Copies from dtype conversion are excluded.

| Pattern | Time | Memory |
| --- | --- | --- |
| Attention, including four projections | `O(B*T*D² + B*T²*D)` | `O(B*T*D + B*H*T²)` |
| Contrastive, n valid pairs | `O(N*D + n²*D)` including input filtering | `O(n*D + n² + N)` |
| Conv2D, tap-loop version | `O(B*Cout*Cin*Ho*Wo*Kh*Kw)` | Padded input `O(B*Cin*(H+2 * padding_height)*(W+2 * padding_width))` + output-sized arrays `O(B*Cout*Ho*Wo)` |
| Materialized im2col | Same arithmetic order | Adds `O(B*Ho*Wo*Cin*Kh*Kw)` |
| Pairwise IoU | `O(N*M + N + M)` | `O(N*M + N + M)` |
| Huber + full manual gradient | `O(N*D)` | `O(N*D)` |
| Voxel mean with sorting-based grouping | `O(N*log N + N*C)` | `O(N + N*C + K*C)` including selected features |
| Ego transforms | `O(B*T + B*A*T)` | `O(B*T + B*A*T)` |

## Final practice checks

1. State the axes and the reduction: keys, candidates, channels, coordinates, or points?
2. Check a tiny hand example before randomized inputs.
3. Check all-invalid/empty behavior, zero denominators, and boundaries.
4. Keep masks boolean; validate shapes and scalar constraints with `ValueError`.
5. Preserve caller inputs; write accumulated results into fresh arrays.
6. Explain both time and peak memory, including pair grids and materialized patches.

For PyTorch, keep newly created tensors on the input's device. Build constants
with `x.new_zeros(...)` or `device=x.device`. Use `clone()` to preserve autograd;
`detach()` cuts the graph. Floating-point scatter sums on a GPU can vary with
addition order. Use tolerance-based comparisons for numerical checks.
