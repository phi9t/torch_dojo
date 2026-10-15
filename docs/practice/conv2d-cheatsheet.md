# Conv2D cheatsheet

Companion to [workout 0127](../../dojo/0127_conv2d_scratch.py) and the
[interactive Conv2D widget](https://phi9t.github.io/torch_dojo/fundamentals/#conv).
The contract is **cross-correlation**, zero padding, positive integer strides,
dilation 1 and groups 1. Start with float64.

## Shape card

| Name | Shape | Meaning |
| --- | --- | --- |
| `X` | `[batch_size,input_channels,input_height,input_width]` | Batch of channel-first images |
| `W` | `[output_channels,input_channels,kernel_height,kernel_width]` | One full filter per output channel |
| `bias` | `[output_channels]` | One bias per output channel |
| `Xp` | `[batch_size,input_channels,input_height+2 * padding_height,input_width+2 * padding_width]` | Input with spatial zero padding |
| `out` | `[batch_size,output_channels,output_height,output_width]` | Output maps |
| `window` | `[input_channels,kernel_height,kernel_width]` | All input channels at one spatial location |
| `W[o]` | `[input_channels,kernel_height,kernel_width]` | Filter for output channel o |

Use descriptive dimension names; `W` names the filter tensor.
**Every output channel consumes every input channel.** A filter with input_channels=3
and a 3×3 kernel sums 27 products per output value, then adds one bias.

## Formula and output dimensions

For a padded input, one output value is:

```text
out[b,o,y,x] = bias[o]
            + Σ over c,i,j of Xp[b,c,y*stride_height+i,x*stride_width+j] * W[o,c,i,j]
```

`b,o,y,x` survive in the output; `c,i,j` are summed away. Do not flip the filter.
The last valid window must satisfy `y*stride_height + kernel_height ≤ input_height + 2 * padding_height`, so:

```python
output_height = (input_height + 2 * padding_height - kernel_height) // stride_height + 1
output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1
```

The floor discards a trailing partial window. Reject nonpositive output dimensions.
For the globals in the workout:

```text
X: [2,3,7,6]    W: [4,3,3,3]    stride: (2,1)    padding: (1,1)
output_height = (7 + 2 - 3) // 2 + 1 = 4
output_width = (6 + 2 - 3) // 1 + 1 = 6
out: [2,4,4,6]
```

## Direct window implementation

Start with this pattern to make the scalar formula visible in code.
The compact examples below assume validated float64 arrays and
positive kernel/output dimensions; the workout answer includes validation.

```python
def conv2d_windows(X, W, bias, stride=(1, 1), padding=(0, 0)):
    batch_size, input_channels, input_height, input_width = X.shape
    output_channels, _, kernel_height, kernel_width = W.shape
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    output_height = (input_height + 2 * padding_height - kernel_height) // stride_height + 1
    output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1
    Xp = np.pad(
        X,
        ((0, 0), (0, 0), (padding_height, padding_height), (padding_width, padding_width)),
    )
    out = np.empty((batch_size, output_channels, output_height, output_width), dtype=np.float64)
    for b in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    window = Xp[b, :, y*stride_height:y*stride_height+kernel_height, x*stride_width:x*stride_width+kernel_width]
                    out[b, o, y, x] = (window * W[o]).sum() + bias[o]
    return out
```

Notice the slice stop: **start + kernel size**. The default slice step is 1:
stride moves the window's origin; dilation would change spacing inside a window.
`Xp` and `out` are new arrays, so the input tensors stay unchanged.

## Vectorize positions and loop over kernel taps

The workout's production answer uses this version. It validates shapes and
spatial pairs, converts inputs to float64, and handles non-contiguous arrays
without modifying caller inputs. It rejects boolean or fractional stride/padding
values and empty kernels with `ValueError`.

Fix kernel offset `(i,j)`. Select the corresponding input value for every batch,
channel and output position at once:

```python
patch = Xp[:, :, i:i + stride_height*output_height:stride_height, j:j + stride_width*output_width:stride_width]  # [batch_size,input_channels,output_height,output_width]
```

Its sampled row indices are `i, i+stride_height, ..., i+(output_height-1)*stride_height`. The slice stop is
exclusive. Here the slice step is stride because it traverses **output positions**.

```python
out = np.zeros((batch_size, output_channels, output_height, output_width), dtype=np.float64) + bias[None, :, None, None]
for i in range(kernel_height):
    for j in range(kernel_width):
        patch = Xp[:, :, i:i + stride_height*output_height:stride_height, j:j + stride_width*output_width:stride_width]
        out += np.einsum("bchw,oc->bohw", patch, W[:, :, i, j])
```

Read the einsum as: retain `b,h,w`, introduce output channel `o`, and sum
over input channel `c`. The outer loops sum over kernel offsets `i,j`.
Add bias once, outside these loops.

## Patch views and im2col

Extract all windows, then contract input channels and kernel axes:

```python
patches = np.lib.stride_tricks.sliding_window_view(Xp, (kernel_height, kernel_width), axis=(2, 3))
patches = patches[:, :, ::stride_height, ::stride_width]             # [batch_size,input_channels,output_height,output_width,kernel_height,kernel_width]
out = np.einsum("bchwij,ocij->bohw", patches, W) + bias[None, :, None, None]
```

For an explicit matrix multiplication, align the flattened axes carefully:

```python
cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size*output_height*output_width, input_channels*kernel_height*kernel_width)
filters = W.reshape(output_channels, input_channels*kernel_height*kernel_width)
out = (cols @ filters.T + bias).reshape(batch_size, output_height, output_width, output_channels).transpose(0, 3, 1, 2)
```

Each row of `cols` is one window flattened in `(c,i,j)` order. Each row of
`filters` must use that same order. A reshape alone cannot reorder patch axes.
The extracted windows share storage; treat them as read-only. Making `cols`
contiguous can materialize all repeated window values.
See [NumPy sliding_window_view](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html).

## NumPy and PyTorch API card

| Task | NumPy | PyTorch |
| --- | --- | --- |
| Float64 | `np.asarray(X,dtype=np.float64)` | `X.to(dtype=torch.float64)` |
| Spatial zero padding | `np.pad(X,((0,0),(0,0),(padding_height,padding_height),(padding_width,padding_width)))` | `torch.nn.functional.pad(X,(padding_width,padding_width,padding_height,padding_height))` |
| Allocate output | `np.empty(shape,dtype=np.float64)` | `X.new_empty(shape)` after float64 conversion |
| Bias broadcasting | `bias[None,:,None,None]` | Same syntax |
| Channel contraction | `np.einsum("bchw,oc->bohw",patch,weights)` | `torch.einsum("bchw,oc->bohw",patch,weights)` |
| Patch extraction | `sliding_window_view` followed by stride slicing | `Xp.unfold(2,kernel_height,stride_height).unfold(3,kernel_width,stride_width)` |
| Reorder patch axes | `patches.transpose(0,2,3,1,4,5)` | `patches.permute(0,2,3,1,4,5)` |
| Check numbers | `np.testing.assert_allclose(actual,expected)` | `torch.testing.assert_close(actual,expected)` |

NumPy specifies padding for every axis. PyTorch starts with the last axis:
**left, right, top, bottom**. Keep PyTorch weights, bias and input on the same
device and dtype. After a permutation, use `reshape`; `view` can fail on
non-contiguous storage.

```python
# PyTorch patch contraction; Xp is already padded.
patches = Xp.unfold(2, kernel_height, stride_height).unfold(3, kernel_width, stride_width)  # [batch_size,input_channels,output_height,output_width,kernel_height,kernel_width]
out = torch.einsum("bchwij,ocij->bohw", patches, W) + bias[None, :, None, None]
```

The chained [Tensor.unfold](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unfold.html)
calls expose views. [torch.nn.functional.unfold](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html) instead presents flattened
patches shaped `[batch_size,input_channels*kernel_height*kernel_width,output_height*output_width]`; avoid mixing the two layouts.

## Time and memory

These approaches perform `O(batch_size*output_channels*output_height*output_width*input_channels*kernel_height*kernel_width)` arithmetic. The direct
version has more Python-loop overhead. The tap-loop version delegates batch,
channel and spatial work to NumPy; im2col delegates a large contraction to matmul.

Memory below excludes caller inputs and dtype-conversion copies:

| Approach | Main allocations |
| --- | --- |
| Direct windows | Padded input + output + a temporary `[input_channels,kernel_height,kernel_width]` product |
| Kernel-tap loops | Padded input + output + an output-sized contraction temporary |
| Patch view + einsum | Padded input + output; contraction may need additional workspace |
| Materialized im2col | Padded input + output + `batch_size*output_height*output_width*input_channels*kernel_height*kernel_width` patch elements |

Exact patch/input element ratio:
`output_height*output_width*kernel_height*kernel_width / (input_height*input_width)`. It is `kernel_height*kernel_width` when output and input spatial sizes
match; stride and padding change the ratio.

For batch_size=8, input_channels=64, 56×56 images, 3×3 kernel, stride 1, padding 1, float64:
input = **12.25 MiB**; materialized patches = **110.25 MiB**, exactly **9×**.
Those figures exclude the padded input, weights, output and matmul workspace.

## Decisive checks and common mistakes

| Check | Expected | Catches |
| --- | --- | --- |
| Input `[[1,2,3],[4,5,6],[7,8,9]]`, filter `[[1,0],[0,-1]]`, zero bias | 2×2 of −4 | Window arithmetic |
| Input `[[0,1],[2,3]]`, filter `[[1,2],[3,4]]`, zero bias | 20; flipped gives 10 | Filter orientation |
| 1×1 filter | Per-pixel channel projection | Channel reduction |
| Zero input, nonzero bias | Each output channel equals its bias everywhere | Missing or repeatedly added bias |
| Stride `(2,1)`, padding `(1,1)`, non-square image | Matches derived output_height/output_width | Axis swaps and window offsets |
| Two or more input/output channels | Sum input_channels, preserve output_channels | Mixing channel axes |
| Snapshot inputs before calling | Inputs unchanged | Accidental writes through views |

Validate ranks, matching input_channels, bias shape `[output_channels]`, positive kernel/stride sizes,
nonnegative padding, and positive output_height/output_width. Use `ValueError` for invalid contracts.
`np.empty` is appropriate only when every output entry is assigned; accumulation
requires an initialized output.

Run from the repository root:

```bash
PYTHONPATH=. python dojo/0127_conv2d_scratch_eval.py --candidate --stage production
```

Explain the operation aloud: “Choose an output location, take a strided-origin
window across every input channel, multiply by one unflipped filter, sum, and
add its bias. Repeat for every output channel.”
