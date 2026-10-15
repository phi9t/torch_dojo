"""Reference and staged checks for Conv2D from scratch (NumPy)."""
# Drill answers: out = floor((input_height + 2p - kernel_height) / s) + 1. Cross-correlation does not
# flip the filter: [[1,0],[0,-1]] over [[1,2],[4,5]] is 1 - 5 = -4, not 2 - 4.

# %%
import numpy as np


# %%
def conv2d(X: np.ndarray, W: np.ndarray, bias: np.ndarray,
           stride: tuple[int, int] = (1, 1), padding: tuple[int, int] = (0, 0)) -> np.ndarray:
    # X [batch_size,input_channels,input_height,W]; W [output_channels,input_channels,kernel_height,kernel_width]; bias [output_channels] -> out [batch_size,output_channels,output_height,output_width] float64
    X, W, bias = (np.asarray(a, dtype=np.float64) for a in (X, W, bias))
    if X.ndim != 4 or W.ndim != 4:
        raise ValueError("X and W must be four-dimensional")
    batch_size, input_channels, input_height, input_width = X.shape
    output_channels, filter_channels, kernel_height, kernel_width = W.shape
    if filter_channels != input_channels or bias.shape != (output_channels,):
        raise ValueError("filters must match input channels; bias must be [output_channels]")
    if kernel_height < 1 or kernel_width < 1:
        raise ValueError("kernel dimensions must be positive")

    def spatial_pair(value, name, minimum):
        try:
            pair = tuple(value)
        except TypeError:
            raise ValueError(f"{name} must contain two integers") from None
        if len(pair) != 2 or any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer))
            or v < minimum
            for v in pair
        ):
            raise ValueError(f"{name} must contain two integers >= {minimum}")
        return tuple(int(v) for v in pair)

    stride_height, stride_width = spatial_pair(stride, "stride", 1)
    padding_height, padding_width = spatial_pair(padding, "padding", 0)
    output_height = (input_height + 2 * padding_height - kernel_height) // stride_height + 1
    output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1
    if output_height < 1 or output_width < 1:
        raise ValueError("kernel larger than padded input")

    # Pad height and width with zeros; leave batch and channels alone.
    Xp = np.pad(
        X,
        ((0, 0), (0, 0), (padding_height, padding_height), (padding_width, padding_width)),
    )
    out = np.zeros((batch_size, output_channels, output_height, output_width), dtype=np.float64)
    out += bias[None, :, None, None]
    # Loop only over kernel taps. Each slice covers every output position;
    # einsum sums input channels while retaining batch and output channels.
    # This avoids materializing the full im2col matrix.
    for i in range(kernel_height):
        for j in range(kernel_width):
            patch = Xp[
                :, :,
                i:i + stride_height * output_height:stride_height,
                j:j + stride_width * output_width:stride_width,
            ]  # [batch_size,input_channels,output_height,output_width]
            out += np.einsum("bchw,oc->bohw", patch, W[:, :, i, j])
    return out


# %% [verifiers]
def naive_conv(X, W, bias, stride, padding):
    batch_size, input_channels, input_height, input_width = X.shape
    output_channels, _, kernel_height, kernel_width = W.shape
    stride_height, stride_width = stride; padding_height, padding_width = padding
    Xp = np.pad(
        X,
        ((0, 0), (0, 0), (padding_height, padding_height), (padding_width, padding_width)),
    )
    output_height, output_width = (input_height + 2 * padding_height - kernel_height) // stride_height + 1, (input_width + 2 * padding_width - kernel_width) // stride_width + 1
    out = np.zeros((batch_size, output_channels, output_height, output_width))
    for b in range(batch_size):
        for o in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    out[b, o, y, x] = bias[o] + sum(
                        Xp[b, c, y * stride_height + i, x * stride_width + j] * W[o, c, i, j]
                        for c in range(input_channels) for i in range(kernel_height) for j in range(kernel_width))
    return out


def verify_baseline_hand_checked_example(conv2d):
    X = np.arange(1., 10.).reshape(1, 1, 3, 3)
    W = np.array([[[[1., 0.], [0., -1.]]]])
    out = conv2d(X, W, np.zeros(1))
    assert out.shape == (1, 1, 2, 2) and np.allclose(out, -4.0)


def verify_baseline_one_by_one_is_channel_projection(conv2d):
    rng = np.random.default_rng(0)
    X, W, b = rng.normal(size=(2, 3, 4, 5)), rng.normal(size=(4, 3, 1, 1)), rng.normal(size=4)
    expected = np.einsum("bchw,oc->bohw", X, W[:, :, 0, 0]) + b[None, :, None, None]
    assert np.allclose(conv2d(X, W, b), expected)


def verify_followup_channels_stride_and_padding_match_naive(conv2d):
    rng = np.random.default_rng(1)
    for stride, padding, kernel in (((1, 1), (0, 0), (3, 3)), ((2, 2), (1, 1), (3, 3)),
                                    ((2, 1), (2, 0), (2, 3)), ((3, 2), (1, 2), (3, 2))):
        X, b = rng.normal(size=(2, 3, 7, 6)), rng.normal(size=4)
        W = rng.normal(size=(4, 3, *kernel))
        out = conv2d(X, W, b, stride, padding)
        assert out.shape == naive_conv(X, W, b, stride, padding).shape
        assert np.allclose(out, naive_conv(X, W, b, stride, padding)), (stride, padding)


def verify_followup_output_shape_formula_and_no_filter_flip(conv2d):
    X = np.zeros((1, 1, 8, 9))
    out = conv2d(X, np.zeros((2, 1, 3, 4)), np.zeros(2), (2, 3), (1, 2))
    assert out.shape == (1, 2, (8 + 2 - 3) // 2 + 1, (9 + 4 - 4) // 3 + 1)
    X = np.arange(4.).reshape(1, 1, 2, 2)
    W = np.array([[[[1., 2.], [3., 4.]]]])
    assert conv2d(X, W, np.zeros(1))[0, 0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3 + 3 * 4  # 20, not 10


def verify_production_pure_bias_and_zero_padding(conv2d):
    rng = np.random.default_rng(2)
    X, W, b = rng.normal(size=(1, 2, 5, 5)), rng.normal(size=(3, 2, 3, 3)), np.array([1., 2., 3.])
    snapshot = [a.copy() for a in (X, W, b)]
    out = conv2d(X, W, b, (1, 1), (1, 1))
    assert out.shape == (1, 3, 5, 5) and out.dtype == np.float64
    for before, after in zip(snapshot, (X, W, b)):
        assert np.array_equal(before, after)
    zero = conv2d(np.zeros_like(X), W, b, (1, 1), (1, 1))
    assert np.allclose(zero, b[None, :, None, None])
    # a corner output with padding sees only the in-bounds 2x2 part of the filter
    assert np.isclose(out[0, 0, 0, 0] - b[0], (X[0, :, :2, :2] * W[0, :, 1:, 1:]).sum())


def verify_production_rejects_invalid_contracts(conv2d):
    X, W, b = np.zeros((1, 2, 4, 4)), np.zeros((3, 2, 3, 3)), np.zeros(3)
    for args in ((X, np.zeros((3, 1, 3, 3)), b), (X, W, np.zeros(2)), (X, W, b, (0, 1)),
                 (X, W, b, (1, 1), (-1, 0)), (X, np.zeros((3, 2, 5, 5)), b), (np.zeros((2, 4, 4)), W, b)):
        try:
            conv2d(*args)
        except ValueError:
            continue
        raise AssertionError("invalid input accepted")


def verify_production_noncontiguous_inputs_and_float64_conversion(conv2d):
    rng = np.random.default_rng(127)
    X = rng.normal(size=(2, 3, 7, 6)).astype(np.float32).swapaxes(2, 3)
    W = rng.normal(size=(4, 3, 2, 3)).astype(np.float32)[..., ::-1]
    b = np.arange(4, dtype=np.int32)
    before = [a.copy() for a in (X, W, b)]
    assert not X.flags.c_contiguous and not W.flags.c_contiguous
    actual = conv2d(X, W, b, (np.int64(2), np.int64(1)), (1, 2))
    expected = naive_conv(X.astype(np.float64), W.astype(np.float64), b, (2, 1), (1, 2))
    assert actual.dtype == np.float64 and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    for old, current in zip(before, (X, W, b)):
        np.testing.assert_array_equal(old, current)


def verify_production_rejects_malformed_spatial_pairs_and_empty_kernels(conv2d):
    X, W, b = np.zeros((1, 2, 4, 4)), np.zeros((3, 2, 3, 3)), np.zeros(3)
    bad_pairs = (None, 1, (), (1,), (1, 1, 1), (1.0, 1), (True, 1), (1, np.bool_(True)), "11")
    for bad in bad_pairs:
        for stride, padding in ((bad, (0, 0)), ((1, 1), bad)):
            try:
                conv2d(X, W, b, stride, padding)
            except ValueError:
                continue
            raise AssertionError("malformed stride/padding pair accepted")
    for shape in ((3, 2, 0, 3), (3, 2, 3, 0)):
        try:
            conv2d(X, np.zeros(shape), b)
        except ValueError:
            continue
        raise AssertionError("empty kernel accepted")


# %%
from torch_judge import evaluate
if not evaluate("0127_conv2d_scratch_eval.py", conv2d):
    raise SystemExit(1)
