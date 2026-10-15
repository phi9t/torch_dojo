# Submanifold Conv2D and Conv3D cheatsheet

Companion to [workout 0121](../../dojo/0121_submanifold_conv.py).
**Update features at the existing active coordinates; preserve the active set
and its row order.** Missing neighbors contribute zero. An active coordinate
with a zero feature vector remains active.

The workout uses stride 1, dilation 1, an odd scalar kernel size and no bias.
The same rulebook/apply functions handle 2D and 3D.

## Shapes and coordinate order

| Quantity | 2D | 3D |
| --- | --- | --- |
| Coordinate row | `[batch, height, width]` | `[batch, depth, height, width]` |
| `coords` | `[num_active_sites, 3]` | `[num_active_sites, 4]` |
| `spatial_shape` | `(input_height, input_width)` | `(input_depth, input_height, input_width)` |
| `num_spatial_dims` | 2 | 3 |
| `num_offsets` | `kernel_size ** 2` | `kernel_size ** 3` |
| Size-3 kernel | 9 offsets | 27 offsets |
| `features` | `[num_active_sites, input_channels]` | Same |
| `weight` | `[num_offsets, input_channels, output_channels]` | Same |
| `rulebook` | `[num_offsets, num_active_sites]`, int64 | Same |
| Output features | `[num_active_sites, output_channels]` | Same |

Spatial axes must use the same order in coordinates, `spatial_shape`, kernel
offsets and weights. Batch is part of the coordinate key; it is never shifted
by a spatial kernel offset.

## What makes it submanifold

| Operation | Where output coordinates exist |
| --- | --- |
| Dense convolution | Every output-grid location |
| Regular sparse convolution | Locations reached by the kernel from active inputs; the active set can expand |
| Submanifold convolution | Exactly the input active coordinates |

For a width-axis slice through either a 2D or 3D grid:

```text
width                 0       1       2       3
input support         ●       ●       ·       ●
submanifold support   ●       ●       ·       ●
```

With a size-3 kernel, sites 0 and 1 exchange information. Site 3 sees itself;
it cannot directly reach site 1. Repeated layers propagate through existing
neighbor connections. They do not create a bridge through the hole.
Larger kernels or a separate downsampling stage can change connectivity.

## Rulebook meaning and offset order

For each active output site, look for active input neighbors at:

```text
source_spatial_coordinate = target_spatial_coordinate + offset
```

Record the input row number, rather than its feature values:

```python
rulebook[offset_index, target_row] = source_row
# -1 means the neighbor is missing or out of bounds.
```

This convention matches cross-correlation: add the offset to the target
coordinate; keep the associated weight in its original orientation.

```python
from itertools import product

radius = kernel_size // 2
offsets = list(product(range(-radius, radius + 1), repeat=num_spatial_dims))
```

The last spatial axis varies fastest. For kernel size 3:

| Selected offset | 2D index | 3D index |
| --- | ---: | ---: |
| Width −1, other axes 0 | 3: `(0,-1)` | 12: `(0,0,-1)` |
| Center | 4: `(0,0)` | 13: `(0,0,0)` |
| Width +1, other axes 0 | 5: `(0,1)` | 14: `(0,0,1)` |

The center rulebook row equals `arange(num_active_sites)`.

## Readable shared rulebook builder

This learning kernel assumes validated metadata; use the production checklist
below before applying it to arbitrary inputs. It keeps coordinates as tuples,
so row-boundary wrapping cannot occur.

```python
def build_rulebook(coords, spatial_shape, batch_size, kernel_size=3):
    num_spatial_dims = len(spatial_shape)
    num_active_sites = coords.shape[0]
    radius = kernel_size // 2
    offsets = list(product(range(-radius, radius + 1), repeat=num_spatial_dims))
    coordinate_rows = coords.tolist()
    lookup = {tuple(coord): row for row, coord in enumerate(coordinate_rows)}
    rulebook = torch.full(
        (len(offsets), num_active_sites), -1,
        dtype=torch.int64, device=coords.device,
    )
    for target_row, coord in enumerate(coordinate_rows):
        batch, *target_spatial = coord
        for offset_index, offset in enumerate(offsets):
            source_spatial = tuple(
                value + delta for value, delta in zip(target_spatial, offset)
            )
            inside = all(
                0 <= value < size
                for value, size in zip(source_spatial, spatial_shape)
            )
            if inside:
                rulebook[offset_index, target_row] = lookup.get(
                    (batch, *source_spatial), -1
                )
    return rulebook
```

The dictionary is a clear CPU reference strategy. On CUDA, `.tolist()` transfers
coordinates to the host, and scalar writes into a CUDA rulebook add overhead.
Use device-side lookup or a sparse-convolution library for a measured GPU workload.

The workout answer instead sorts packed coordinate keys once, retains each key's
original feature row, and uses `torch.searchsorted` for batched neighbor lookups.
Before adding an offset, it filters targets whose neighbors would leave the
grid. This also prevents coordinate addition from overflowing near int64's limit.
An insertion position is a match only if it is in range and its key equals
the neighbor key; missing neighbors remain −1.

## Shared feature application

For each offset: **gather source rows → multiply by the offset's channel matrix
→ accumulate into target rows**.

```python
def apply_rulebook(features, weight, rulebook):
    num_offsets, num_active_sites = rulebook.shape
    output_channels = weight.shape[2]
    output = features.new_zeros((num_active_sites, output_channels))
    for offset_index in range(num_offsets):
        source_rows = rulebook[offset_index]
        valid = source_rows >= 0
        target_rows = valid.nonzero(as_tuple=True)[0]
        contribution = features[source_rows[valid]] @ weight[offset_index]
        output = output.index_add(0, target_rows, contribution)
    return output
```

For one offset, if `num_connections` neighbor links exist:

```text
gathered features   [num_connections, input_channels]
offset weights     [input_channels, output_channels]
contributions      [num_connections, output_channels]
```

Mask −1 **before** indexing: `features[-1]` means the last feature row.
Selecting only valid rows also avoids multiplying a dummy feature by zero,
which would not protect against NaN values in that dummy row.

Within one offset, target rows are unique. Several offsets contribute to the
same output, so their contributions must be added rather than overwritten.

## Hand-checkable 2D and 3D versions

Use one input/output channel and three active sites, ordered by width 0, 1, 3.
Give them features 2, 4, 8. Set center weight to 2, width −1 weight to 1,
and width +1 weight to 3. Set all other weights to zero.

```python
features = torch.tensor([[2.0], [4.0], [8.0]], dtype=torch.float64)

# 2D: [batch, height, width]
coords_2d = torch.tensor([[0, 1, 0], [0, 1, 1], [0, 1, 3]], dtype=torch.int64)
weight_2d = torch.zeros((9, 1, 1), dtype=torch.float64)
weight_2d[3, 0, 0], weight_2d[4, 0, 0], weight_2d[5, 0, 0] = 1, 2, 3
rulebook_2d = build_rulebook(coords_2d, (3, 4), batch_size=1)
output_2d = apply_rulebook(features, weight_2d, rulebook_2d)

# 3D: [batch, depth, height, width]
coords_3d = torch.tensor(
    [[0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 3]], dtype=torch.int64
)
weight_3d = torch.zeros((27, 1, 1), dtype=torch.float64)
weight_3d[12, 0, 0], weight_3d[13, 0, 0], weight_3d[14, 0, 0] = 1, 2, 3
rulebook_3d = build_rulebook(coords_3d, (3, 3, 4), batch_size=1)
output_3d = apply_rulebook(features, weight_3d, rulebook_3d)
# Both outputs: [[16.0], [10.0], [16.0]]
```

The arithmetic is identical in both dimensions:

```text
width 0: 2*center_weight + 4*positive_width_weight = 2*2 + 4*3 = 16
width 1: 2*negative_width_weight + 4*center_weight = 2*1 + 4*2 = 10
width 3: 8*center_weight = 16
```

Showing only the width-neighbor rows, the rulebook is:

```text
                       target row 0   target row 1   target row 2
width offset -1             -1              0             -1
center                       0              1              2
width offset +1              1             -1             -1
```

There is no output row for width 2. If the feature at width 0 becomes zero,
its coordinate still exists and it can receive information from width 1.

## NumPy and PyTorch patterns

| Task | PyTorch | NumPy |
| --- | --- | --- |
| Missing-neighbor table | `torch.full(shape,-1,dtype=torch.int64,device=coords.device)` | `np.full(shape,-1,dtype=np.int64)` |
| Gather valid sources | `features[source_rows[valid]]` | Same indexing syntax |
| Find target rows | `valid.nonzero(as_tuple=True)[0]` | `np.flatnonzero(valid)` |
| Channel multiplication | `gathered @ weight[offset_index]` | Same syntax |
| Accumulate rows | `output.index_add(0,target_rows,contribution)` | `np.add.at(output,target_rows,contribution)` |
| Zero output beside features | `features.new_zeros(shape)` | `np.zeros(shape,dtype=features.dtype)` |
| Detect duplicate coordinates | `torch.unique(coords,dim=0)` | `np.unique(coords,axis=0)` |

NumPy's `np.add.at` handles repeated destination indices. PyTorch's `index_add`
keeps the feature computation in autograd. See [NumPy indexed accumulation](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)
and [PyTorch index_add](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add.html).

## Gradients and dense-oracle checks

Coordinates, lookup indices and activity are discrete metadata. Gradients flow
through gathered features, channel matrices and accumulated outputs.
For a connection with incoming output gradient `gradient_output`:

```text
gradient_features[source_row] += gradient_output[target_row] @ weight[offset].T
gradient_weight[offset]       += outer(features[source_row], gradient_output[target_row])
```

One source can feed multiple targets, so its backward contributions accumulate.
An all-zero active feature may receive a nonzero gradient.

For empty inputs, the learning kernel returns the correct zero-row output.
If an optimized implementation skips all empty contributions, retain zero-valued
dependencies on the inputs so backward still works, for example:
`output + features.sum()*0 + weight.sum()*0`, for finite features and weights.

For a small-grid oracle, scatter features into an otherwise-zero dense grid,
run stride-1 dense Conv2D/Conv3D with padding `kernel_size//2`, then gather
outputs at the original coordinates. Dense convolution is for verification;
the sparse implementation must not allocate that full grid.

Convert the sparse weight layout to the dense layout with:

```python
output_channels = weight.shape[2]
input_channels = weight.shape[1]
dense_weight = weight.permute(2, 1, 0).reshape(
    output_channels, input_channels, *([kernel_size] * num_spatial_dims)
)
```

This relies on the exact offset ordering above. Check feature and weight
gradients against the oracle using the same scalar loss in both computations.

## Production checks

- `coords` is int64, rank 2, with exactly 2 or 3 spatial axes.
- `spatial_shape` matches those axes; its dimensions and `batch_size` are positive integers.
- `kernel_size` is a positive odd integer; this workout uses a centered kernel.
- Coordinates are unique and each batch/spatial component is in bounds.
- The batch/spatial key space fits int64; check products using Python integers before packing keys.
- Features, weights and rulebook have compatible shapes and devices; features and weights share a supported floating dtype.
- Every rulebook entry is −1 or a valid input row. Preserve coordinate row order and batch isolation.
- Empty inputs, all-zero active rows, float32/float64 and strided feature views work without input mutation.

If packing keys, use the appropriate row-major address:

```text
2D: (batch*input_height + height)*input_width + width
3D: ((batch*input_depth + depth)*input_height + height)*input_width + width
```

Validate the **neighbor's individual coordinates before packing**.
At the last width column, adding 1 to a flat key can alias the next row's
first column. A flat-key range check alone does not catch this error.

## Compute, storage and reuse

Let `num_connections` count all valid source/target links across all offsets.
Then `num_connections ≤ num_offsets*num_active_sites`.

| Part | Cost |
| --- | --- |
| Dictionary rulebook construction | Expected `O(num_offsets*num_active_sites*num_spatial_dims)` time, excluding validation |
| Sorted-key lookup in the workout | Sorting plus one batched binary search and per-axis bounds checks per offset |
| Feature application | `O(num_connections*input_channels*output_channels)` arithmetic |
| Dense rulebook | `num_offsets*num_active_sites*bytes_per_index` |
| Feature storage | `num_active_sites*input_channels*bytes_per_value` |
| Output storage | `num_active_sites*output_channels*bytes_per_value` |

The shown apply loop also scans `num_offsets*num_active_sites` rulebook entries.
For the sorted-key builder, sorting costs `O(num_active_sites*log(num_active_sites))`;
each offset costs at most `O(num_active_sites*(num_spatial_dims + log(num_active_sites)))`.
Per-offset gathers and contributions require workspace; autograd may retain
intermediates from multiple offsets for backward.
The out-of-place `index_add` also creates a new output tensor for each offset;
this readable reference is not a fused sparse GPU kernel.

For 400,000 active sites with a size-3 kernel:

| Rulebook | int32 | int64 used by this workout |
| --- | ---: | ---: |
| 2D, 9 offsets | 14.4 MB | 28.8 MB |
| 3D, 27 offsets | 43.2 MB | 86.4 MB |

These are decimal MB and exclude features, outputs and training intermediates.
Compare them with the feature payload before deciding which storage dominates.

Cache the rulebook when coordinates, coordinate row order, spatial shape,
batch metadata and kernel geometry remain unchanged. Channel weights can change
without rebuilding geometry. A change to activity or row order invalidates the cache.

Run the workout's stages from the repository root:

```bash
PYTHONPATH=. python dojo/0121_submanifold_conv_eval.py --candidate --stage production
```

Recall phrase: **keep active sites fixed; look up neighbors, gather features,
apply an offset-specific channel matrix, and accumulate.**
