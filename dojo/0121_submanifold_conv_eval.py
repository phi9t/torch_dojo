"""Submanifold sparse convolution reference and staged checks."""
# %%
from functools import partial
from itertools import product
import types
import torch
from torch.nn import functional as F
from torch_judge import EvalCase, compare_values

def _offsets(dims: int, kernel_size: int = 3) -> list[tuple[int, ...]]:
    radius = kernel_size // 2
    return list(product(range(-radius, radius + 1), repeat=dims))


def build_rulebook(
    coords: torch.Tensor,
    spatial_shape: tuple[int, ...],
    batch_size: int,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Reference: D-dimensional active-neighbour lookup."""
    dims, spatial_shape = _validate_sparse_metadata(
        coords, spatial_shape, batch_size, kernel_size
    )
    rows = {tuple(int(value) for value in coord): row
            for row, coord in enumerate(coords.tolist())}
    rulebook = torch.full(
        (kernel_size ** dims, coords.shape[0]),
        -1,
        dtype=torch.int64,
        device=coords.device,
    )
    for column, coord in enumerate(coords.tolist()):
        batch = int(coord[0])
        spatial = [int(value) for value in coord[1:]]
        for offset, delta in enumerate(_offsets(dims, kernel_size)):
            neighbour_spatial = [value + step for value, step in zip(spatial, delta)]
            if (0 <= batch < batch_size
                    and all(0 <= value < size
                            for value, size in zip(neighbour_spatial, spatial_shape))):
                neighbour = (batch, *neighbour_spatial)
                rulebook[offset, column] = rows.get(neighbour, -1)
    return rulebook


def apply_rulebook(
    features: torch.Tensor,
    weight: torch.Tensor,
    rulebook: torch.Tensor,
) -> torch.Tensor:
    """Reference: gather active neighbours, matmul, and add."""
    output = features.new_zeros((rulebook.shape[1], weight.shape[2]))
    for offset in range(rulebook.shape[0]):
        source_rows = rulebook[offset]
        active = source_rows >= 0
        target_rows = active.nonzero(as_tuple=True)[0]
        contribution = features[source_rows[active]] @ weight[offset]
        output = output.index_add(0, target_rows, contribution)
    return output


# %% [verifiers]
_INT64_MAX = torch.iinfo(torch.int64).max
_FORBIDDEN_DENSE_CONV_NAMES = frozenset({
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
})


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive Python integer")
    return value


def _spatial_shape_tuple(spatial_shape, dims: int) -> tuple[int, ...]:
    try:
        values = tuple(spatial_shape)
    except TypeError as error:
        raise ValueError("spatial_shape must be a sequence of positive integers") from error
    if len(values) != dims:
        raise ValueError("spatial_shape rank must match coords")
    shape = tuple(_positive_int(value, f"spatial_shape[{axis}]")
                  for axis, value in enumerate(values))
    cells = 1
    for size in shape:
        cells *= size
        if cells > _INT64_MAX:
            raise ValueError("spatial key space overflows int64")
    return shape


def _validate_sparse_metadata(
    coords: torch.Tensor,
    spatial_shape: tuple[int, ...],
    batch_size: int,
    kernel_size: int = 3,
) -> tuple[int, tuple[int, ...]]:
    if not isinstance(coords, torch.Tensor) or coords.dtype != torch.int64 or coords.ndim != 2:
        raise ValueError("coords must be an int64 [K, 1+D] tensor")
    dims = coords.shape[1] - 1
    if dims not in (2, 3):
        raise ValueError("coords must describe 2D or 3D sparse sites")
    batch_size = _positive_int(batch_size, "batch_size")
    spatial_shape = _spatial_shape_tuple(spatial_shape, dims)
    kernel_size = _positive_int(kernel_size, "kernel_size")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    total_cells = batch_size
    for size in spatial_shape:
        total_cells *= size
        if total_cells > _INT64_MAX:
            raise ValueError("batch/spatial key space overflows int64")
    if coords.shape[0] == 0:
        return dims, spatial_shape

    bounds = coords.new_tensor((batch_size, *spatial_shape))
    if ((coords < 0) | (coords >= bounds)).any():
        raise ValueError("coords are out of range")
    if torch.unique(coords, dim=0).shape[0] != coords.shape[0]:
        raise ValueError("coords must be unique")
    return dims, spatial_shape


def _dense_active_oracle(coords, features, weight, spatial_shape, batch_size):
    dims, spatial_shape = _validate_sparse_metadata(
        coords, spatial_shape, batch_size
    )
    dense = features.new_zeros((batch_size, features.shape[1], *spatial_shape))
    for row, coord in enumerate(coords.tolist()):
        dense[(int(coord[0]), slice(None), *[int(axis) for axis in coord[1:]])] = features[row]
    kernel_size = round(weight.shape[0] ** (1 / dims))
    conv_weight = weight.permute(2, 1, 0).reshape(
        weight.shape[2], weight.shape[1], *([kernel_size] * dims)
    )
    if dims == 2:
        dense_output = F.conv2d(dense, conv_weight, padding=kernel_size // 2)
    elif dims == 3:
        dense_output = F.conv3d(dense, conv_weight, padding=kernel_size // 2)
    else:
        raise ValueError("oracle supports 2D and 3D")
    if coords.shape[0] == 0:
        return features.new_empty((0, weight.shape[2]))
    return torch.stack([
        dense_output[(int(coord[0]), slice(None), *[int(axis) for axis in coord[1:]])]
        for coord in coords.tolist()
    ])


def _asymmetric_weight(offsets: int, channels_in: int, channels_out: int):
    weight = torch.arange(
        offsets * channels_in * channels_out,
        dtype=torch.float64,
    ).reshape(offsets, channels_in, channels_out)
    signs = torch.where(
        torch.arange(offsets, dtype=torch.int64) % 2 == 0,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(-1.0, dtype=torch.float64),
    )
    return (weight + 0.25) * signs[:, None, None] / 11.0


def _candidate_forward(build_rulebook, apply_rulebook, case):
    inputs = case.inputs
    rulebook = build_rulebook(
        inputs["coords"], inputs["spatial_shape"], inputs["batch_size"]
    )
    return apply_rulebook(inputs["features"], inputs["weight"], rulebook)


def _run_with_grads(case, forward):
    inputs = case.inputs
    features = inputs["features"].clone().detach().requires_grad_(True)
    weight = inputs["weight"].clone().detach().requires_grad_(True)
    output = forward(inputs, features, weight)
    cotangent = torch.linspace(
        -0.7, 0.9, output.numel(), dtype=output.dtype, device=output.device
    ).reshape_as(output)
    loss = (output * cotangent).sum()
    loss.backward()
    return {
        "output": output.detach(),
        "features_grad": features.grad.detach(),
        "weight_grad": weight.grad.detach(),
    }


def _value_and_grads(build_rulebook, apply_rulebook, case):
    def forward(inputs, features, weight):
        rulebook = build_rulebook(
            inputs["coords"], inputs["spatial_shape"], inputs["batch_size"]
        )
        return apply_rulebook(features, weight, rulebook)

    return _run_with_grads(case, forward)


def _dense_value_and_grads(case):
    def forward(inputs, features, weight):
        return _dense_active_oracle(
            inputs["coords"], features, weight,
            inputs["spatial_shape"], inputs["batch_size"],
        )

    return _run_with_grads(case, forward)


def _code_uses_dense_convolution(code: types.CodeType) -> bool:
    if any(name in _FORBIDDEN_DENSE_CONV_NAMES for name in code.co_names):
        return True
    return any(
        isinstance(value, types.CodeType) and _code_uses_dense_convolution(value)
        for value in code.co_consts
    )


def _reject_dense_convolution_calls(*functions):
    offenders = [
        function.__name__
        for function in functions
        if _code_uses_dense_convolution(function.__code__)
    ]
    if offenders:
        names = ", ".join(offenders)
        forbidden = ", ".join(sorted(_FORBIDDEN_DENSE_CONV_NAMES))
        raise AssertionError(
            f"answer functions may not call dense convolution ({names}); "
            f"forbidden names: {forbidden}"
        )


def _answer_functions(build_rulebook):
    apply_rulebook = build_rulebook.__globals__.get("apply_rulebook")
    if not callable(apply_rulebook):
        raise AssertionError("answer section must define apply_rulebook")
    _reject_dense_convolution_calls(build_rulebook, apply_rulebook)
    return build_rulebook, apply_rulebook


def make_eval_data():
    cases = {}
    fixture_coords = torch.tensor([
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 3],
        [0, 0, 1],
        [0, 2, 1],
    ], dtype=torch.int64)
    expected_rulebook = torch.tensor([
        [-1, -1, -1, -1,  0],
        [-1,  3, -1, -1,  1],
        [ 3, -1, -1, -1, -1],
        [-1,  0, -1, -1, -1],
        [ 0,  1,  2,  3,  4],
        [ 1, -1, -1, -1, -1],
        [-1, -1, -1,  0, -1],
        [-1,  4, -1,  1, -1],
        [ 4, -1, -1, -1, -1],
    ], dtype=torch.int64)
    cases["hand_rulebook"] = EvalCase(
        name="hand_rulebook",
        stage="baseline",
        purpose="A literal 3x3 rulebook keeps the x=2 hole inactive and maps the center row to itself.",
        inputs=dict(coords=fixture_coords, spatial_shape=(3, 4), batch_size=1),
        expected=expected_rulebook,
    )

    oracle_coords = torch.tensor([
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 3],
        [0, 0, 1],
        [0, 2, 1],
    ], dtype=torch.int64)
    features = torch.tensor([
        [1.0, -2.0],
        [0.5,  3.0],
        [0.0,  0.0],
        [4.0, -1.0],
        [2.0,  1.5],
    ], dtype=torch.float64)
    weight = torch.arange(54, dtype=torch.float64).reshape(9, 2, 3) / 7.0
    weight[1::2] = -weight[1::2]
    expected = _dense_active_oracle(oracle_coords, features, weight, (3, 4), 1)
    cases["dense_oracle"] = EvalCase(
        name="dense_oracle",
        stage="baseline",
        purpose="Asymmetric cross-correlation weights match dense conv sampled only at active input sites.",
        inputs=dict(coords=oracle_coords, spatial_shape=(3, 4), batch_size=1,
                    features=features, weight=weight),
        expected=expected,
        comparison=partial(compare_values, atol=1e-11, rtol=1e-11),
    )

    coords_3d = torch.tensor([
        [1, 1, 2, 0],
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 0, 1, 0],
        [1, 2, 1, 0],
        [0, 1, 1, 0],
    ], dtype=torch.int64)
    features_3d = torch.tensor([
        [ 0.0,  0.0],
        [ 1.5, -0.5],
        [ 2.0,  1.0],
        [-1.0,  0.25],
        [ 0.75, -1.25],
        [ 3.0,  2.0],
    ], dtype=torch.float64)
    weight_3d = _asymmetric_weight(27, 2, 4)
    expected_3d = _dense_active_oracle(coords_3d, features_3d, weight_3d, (3, 3, 2), 2)
    cases["followup_3d_batches_order"] = EvalCase(
        name="followup_3d_batches_order",
        stage="followup",
        purpose=("3D unsorted coordinates keep input row order, separate batches, "
                 "and keep an all-zero feature row active."),
        inputs=dict(coords=coords_3d, spatial_shape=(3, 3, 2), batch_size=2,
                    features=features_3d, weight=weight_3d),
        expected=expected_3d,
        comparison=partial(compare_values, atol=1e-11, rtol=1e-11),
    )

    empty_coords = torch.empty((0, 4), dtype=torch.int64)
    empty_features = torch.empty((0, 2), dtype=torch.float64)
    empty_weight = _asymmetric_weight(27, 2, 3)
    cases["followup_empty_3d"] = EvalCase(
        name="followup_empty_3d",
        stage="followup",
        purpose="An empty active set returns an empty active-row output with the requested output channel count.",
        inputs=dict(coords=empty_coords, spatial_shape=(2, 3, 4), batch_size=2,
                    features=empty_features, weight=empty_weight),
        expected=torch.empty((0, 3), dtype=torch.float64),
        comparison=partial(compare_values, atol=1e-11, rtol=1e-11),
    )

    grad_coords = torch.tensor([
        [0, 1, 1, 1],
        [0, 1, 2, 1],
        [0, 2, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 2],
    ], dtype=torch.int64)
    grad_features = torch.tensor([
        [ 0.2, -1.0,  1.5],
        [ 2.0,  0.5, -0.5],
        [-1.2,  1.0,  0.25],
        [ 3.0, -2.0,  1.0],
        [ 0.5,  1.25, -1.5],
    ], dtype=torch.float64)
    grad_weight = _asymmetric_weight(27, 3, 2)
    grad_case = EvalCase(
        name="followup_gradients",
        stage="followup",
        purpose="Feature and weight gradients match a dense 3D cross-correlation oracle.",
        inputs=dict(coords=grad_coords, spatial_shape=(3, 4, 3), batch_size=2,
                    features=grad_features, weight=grad_weight),
        expected=None,
        comparison=partial(compare_values, atol=1e-10, rtol=1e-10),
    )
    grad_case.expected = _dense_value_and_grads(grad_case)
    grad_case.__post_init__()
    cases["followup_gradients"] = grad_case

    alias_coords = torch.tensor([
        [0, 0, 2],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
    ], dtype=torch.int64)
    alias_features = torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float64)
    alias_weight = _asymmetric_weight(9, 1, 2)
    alias_expected = _dense_active_oracle(alias_coords, alias_features, alias_weight, (2, 3), 1)
    cases["production_flat_key_boundaries"] = EvalCase(
        name="production_flat_key_boundaries",
        stage="production",
        purpose=("A site at x=X-1 is not adjacent to the next row's x=0 site, "
                 "and x=-1 does not wrap to the previous row."),
        inputs=dict(coords=alias_coords, spatial_shape=(2, 3), batch_size=1,
                    features=alias_features, weight=alias_weight),
        expected=alias_expected,
        comparison=partial(compare_values, atol=1e-11, rtol=1e-11),
    )

    strided_base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    strided_features = strided_base[:, ::2].requires_grad_()
    strided_coords = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ], dtype=torch.int64)
    strided_weight = _asymmetric_weight(27, 3, 2).to(torch.float32)
    strided_expected = _dense_active_oracle(
        strided_coords, strided_features, strided_weight, (2, 2, 2), 1
    )
    cases["production_strided_float32"] = EvalCase(
        name="production_strided_float32",
        stage="production",
        purpose="A non-contiguous float32 feature view keeps its channel order.",
        inputs=dict(coords=strided_coords, spatial_shape=(2, 2, 2), batch_size=1,
                    features=strided_features, weight=strided_weight),
        expected=strided_expected,
        comparison=partial(compare_values, atol=2e-5, rtol=2e-5),
    )

    zero_coords = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int64)
    zero_features = torch.tensor([[0.0, 0.0], [2.0, -3.0]], dtype=torch.float64)
    zero_weight = _asymmetric_weight(9, 2, 1)
    zero_expected = _dense_active_oracle(zero_coords, zero_features, zero_weight, (1, 3), 1)
    cases["production_zero_feature_row"] = EvalCase(
        name="production_zero_feature_row",
        stage="production",
        purpose="An all-zero feature vector remains an active output row.",
        inputs=dict(coords=zero_coords, spatial_shape=(1, 3), batch_size=1,
                    features=zero_features, weight=zero_weight),
        expected=zero_expected,
        comparison=partial(compare_values, atol=1e-11, rtol=1e-11),
    )

    empty_2d = EvalCase(
        name="production_empty_2d",
        stage="production",
        purpose="An empty 2D active set is valid and returns empty float64 rows.",
        inputs=dict(coords=torch.empty((0, 3), dtype=torch.int64),
                    spatial_shape=(3, 5), batch_size=2,
                    features=torch.empty((0, 2), dtype=torch.float64),
                    weight=_asymmetric_weight(9, 2, 4)),
        expected=torch.empty((0, 4), dtype=torch.float64),
        comparison=partial(compare_values, atol=1e-11, rtol=1e-11),
    )
    cases["production_empty_2d"] = empty_2d
    return cases


def _expect_value_error(label, fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError for {label}")


def verify_baseline_hand_sized_rulebook(build_rulebook):
    build_rulebook, _ = _answer_functions(build_rulebook)
    case = make_eval_data()["hand_rulebook"]
    actual = build_rulebook(**case.inputs)
    errors = case.comparison(actual, case.expected)
    assert not errors, "; ".join(errors)


def verify_baseline_dense_conv_oracle(build_rulebook):
    build_rulebook, apply_rulebook = _answer_functions(build_rulebook)
    case = make_eval_data()["dense_oracle"]
    actual = _candidate_forward(build_rulebook, apply_rulebook, case)
    errors = case.comparison(actual, case.expected)
    assert not errors, "; ".join(errors)


def verify_followup_3d_batches_order_and_zero_feature_rows(build_rulebook):
    build_rulebook, apply_rulebook = _answer_functions(build_rulebook)
    case = make_eval_data()["followup_3d_batches_order"]
    actual = _candidate_forward(build_rulebook, apply_rulebook, case)
    errors = case.comparison(actual, case.expected)
    assert not errors, "; ".join(errors)


def verify_followup_empty_active_set(build_rulebook):
    build_rulebook, apply_rulebook = _answer_functions(build_rulebook)
    case = make_eval_data()["followup_empty_3d"]
    actual = _candidate_forward(build_rulebook, apply_rulebook, case)
    errors = case.comparison(actual, case.expected)
    assert not errors, "; ".join(errors)


def verify_followup_feature_and_weight_gradients(build_rulebook):
    build_rulebook, apply_rulebook = _answer_functions(build_rulebook)
    case = make_eval_data()["followup_gradients"]
    actual = _value_and_grads(build_rulebook, apply_rulebook, case)
    errors = case.comparison(actual, case.expected)
    assert not errors, "; ".join(errors)


def verify_production_rejects_invalid_sparse_metadata(build_rulebook):
    build_rulebook, _ = _answer_functions(build_rulebook)
    valid = dict(
        coords=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int64),
        spatial_shape=(2, 2, 2),
        batch_size=1,
    )
    changes = {
        "duplicate coords": dict(coords=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int64)),
        "negative x": dict(coords=torch.tensor([[0, 0, 0, -1]], dtype=torch.int64)),
        "wrapped x": dict(coords=torch.tensor([[0, 0, 0, 2]], dtype=torch.int64)),
        "out-of-range y": dict(coords=torch.tensor([[0, 0, 2, 0]], dtype=torch.int64)),
        "out-of-range z": dict(coords=torch.tensor([[0, 2, 0, 0]], dtype=torch.int64)),
        "negative batch": dict(coords=torch.tensor([[-1, 0, 0, 0]], dtype=torch.int64)),
        "out-of-range batch": dict(coords=torch.tensor([[1, 0, 0, 0]], dtype=torch.int64)),
        "float coords": dict(coords=torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)),
        "bad batch size": dict(batch_size=0),
        "bool batch size": dict(batch_size=True),
        "zero spatial extent": dict(spatial_shape=(2, 0, 2)),
        "bool spatial extent": dict(spatial_shape=(2, True, 2)),
        "rank mismatch": dict(coords=torch.tensor([[0, 0, 0]], dtype=torch.int64)),
        "int64 key-space overflow": dict(spatial_shape=(2 ** 62, 4, 2)),
    }
    for label, change in changes.items():
        _expect_value_error(label, lambda change=change: build_rulebook(**(valid | change)))


def verify_production_valid_sparse_limits(build_rulebook):
    build_rulebook, apply_rulebook = _answer_functions(build_rulebook)
    data = make_eval_data()
    for name in (
        "production_flat_key_boundaries",
        "production_strided_float32",
        "production_zero_feature_row",
        "production_empty_2d",
    ):
        case = data[name]
        actual = _candidate_forward(build_rulebook, apply_rulebook, case)
        errors = case.comparison(actual, case.expected)
        assert not errors, f"{name}: " + "; ".join(errors)


def verify_production_large_valid_keys_and_kernel_sizes(build_rulebook):
    build_rulebook, apply_rulebook = _answer_functions(build_rulebook)
    # This grid cannot be allocated densely. Neighbor arithmetic must still be
    # correct near int64's limit, and output columns retain the unsorted order.
    coords = torch.tensor([[0, 0, _INT64_MAX - 1], [0, 0, _INT64_MAX - 2]])
    actual = build_rulebook(coords, (1, _INT64_MAX), 1)
    expected = torch.full((9, 2), -1, dtype=torch.int64)
    expected[3] = torch.tensor([1, -1])
    expected[4] = torch.tensor([0, 1])
    expected[5] = torch.tensor([-1, 0])
    torch.testing.assert_close(actual, expected)

    for spatial_shape in ((3, 4), (3, 3, 4)):
        dims = len(spatial_shape)
        coords = torch.tensor([
            [0, *([1] * (dims - 1)), 3],
            [0, *([1] * (dims - 1)), 0],
            [0, *([1] * (dims - 1)), 1],
        ])
        features = torch.arange(6, dtype=torch.float64).reshape(3, 2)
        for kernel_size in (1, 5):
            rulebook = build_rulebook(coords, spatial_shape, 1, kernel_size)
            weight = _asymmetric_weight(kernel_size ** dims, 2, 3)
            actual = apply_rulebook(features, weight, rulebook)
            expected = _dense_active_oracle(coords, features, weight, spatial_shape, 1)
            torch.testing.assert_close(actual, expected)


def verify_production_empty_feature_and_weight_gradients(build_rulebook):
    build_rulebook, apply_rulebook = _answer_functions(build_rulebook)
    for spatial_shape in ((2, 3), (2, 3, 4)):
        dims = len(spatial_shape)
        coords = torch.empty((0, dims + 1), dtype=torch.int64)
        features = torch.empty((0, 2), dtype=torch.float64, requires_grad=True)
        weight = torch.ones((3 ** dims, 2, 3), dtype=torch.float64, requires_grad=True)
        rulebook = build_rulebook(coords, spatial_shape, 1)
        output = apply_rulebook(features, weight, rulebook)
        assert output.shape == (0, 3)
        output.sum().backward()
        torch.testing.assert_close(features.grad, torch.zeros_like(features))
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


def verify_production_dense_oracle_validates_before_allocation(build_rulebook):
    _answer_functions(build_rulebook)
    coords = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    features = torch.ones((1, 1), dtype=torch.float64)
    weight = torch.ones((9, 1, 1), dtype=torch.float64)
    _expect_value_error(
        "dense oracle int64 key-space overflow",
        lambda: _dense_active_oracle(coords, features, weight, (2 ** 62, 4), 1),
    )

# %%
from torch_judge import evaluate
if not evaluate("0121_submanifold_conv_eval.py", build_rulebook):
    raise SystemExit(1)
