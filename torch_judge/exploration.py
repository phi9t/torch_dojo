"""Inspectable case data for the edit, run, compare, replay loop.

Factories stay in eval-script verifier cells; this module knows no exercises.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import math
import numbers

from torch_judge.engine import _find_eval_file, _load_verifier_context, _STAGE_RANK


def _snapshot(value: Any, memo=None):
    """Detach recorded tensors; retain shared objects within one snapshot."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    # Optional at import time: judge still works in a lightweight Python env.
    import torch
    if isinstance(value, torch.Tensor):
        # Clone one storage once, then rebuild views. This preserves aliases
        # even when distinct tensors point at overlapping slices of a case.
        storage = value.untyped_storage()
        storage_key = ("tensor_storage", str(value.device), storage.data_ptr(), value.dtype)
        if storage_key not in memo:
            flat = torch.empty(0, dtype=value.dtype, device=value.device).set_(
                storage, 0, (storage.nbytes() // value.element_size(),), (1,))
            memo[storage_key] = flat.detach().clone()
        result = memo[storage_key].as_strided(value.shape, value.stride(), value.storage_offset()).detach()
        if value.is_conj():
            result = result.conj()
        result.requires_grad_(value.requires_grad)
        memo[id(value)] = result
        return result
    if isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update((key, _snapshot(item, memo)) for key, item in value.items())
        return result
    if isinstance(value, (list, tuple)):
        items = [_snapshot(item, memo) for item in value]
        result = tuple(items) if isinstance(value, tuple) else items
        memo[id(value)] = result
        return result
    return deepcopy(value, memo)


def compare_values(actual: Any, expected: Any, *, atol=1e-6, rtol=1e-5, path='output') -> list[str]:
    """Compare nested outputs, retaining useful numerical diagnostics."""
    import numpy as np
    import torch
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return [f'{path}: expected Tensor, received {type(actual).__name__}']
        errors = []
        for attribute in ('shape', 'dtype', 'device'):
            if getattr(actual, attribute) != getattr(expected, attribute):
                errors.append(f'{path}: {attribute} {getattr(actual, attribute)} != {getattr(expected, attribute)}')
        if actual.shape != expected.shape or actual.device != expected.device:
            return errors
        a, e = actual.detach(), expected.detach()
        if not e.is_floating_point() and not e.is_complex():
            if not torch.equal(a, e):
                errors.append(f'{path}: integer/bool values differ (exact comparison)')
            return errors
        comparison_dtype = torch.complex128 if a.is_complex() or e.is_complex() else torch.float64
        if not torch.allclose(a.to(comparison_dtype), e.to(comparison_dtype), atol=atol, rtol=rtol, equal_nan=True):
            delta = (a.to(comparison_dtype) - e.to(comparison_dtype)).abs()
            errors.append(f'{path}: values differ; max absolute error={delta.max().item() if delta.numel() else 0:g}; atol={atol:g}, rtol={rtol:g}')
        return errors
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return [f'{path}: expected ndarray, received {type(actual).__name__}']
        errors = []
        if actual.dtype != expected.dtype:
            errors.append(f'{path}: dtype {actual.dtype} != {expected.dtype}')
        if actual.shape != expected.shape:
            return errors + [f'{path}: shape {actual.shape} != {expected.shape}']
        equal = (np.array_equal(actual, expected) if expected.dtype.kind in 'biu'
                 else np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True))
        if not equal:
            errors.append(f'{path}: values differ; max absolute error={np.max(np.abs(actual - expected)) if actual.size else 0:g}')
        return errors
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return [f'{path}: dictionary keys differ']
        return [error for key in expected for error in compare_values(actual[key], expected[key], atol=atol, rtol=rtol, path=f'{path}.{key}')]
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            return [f'{path}: expected {type(expected).__name__} of length {len(expected)}']
        return [error for i, (a, e) in enumerate(zip(actual, expected)) for error in compare_values(a, e, atol=atol, rtol=rtol, path=f'{path}[{i}]')]
    # Discrete scalars must never pass through floating-point coercion: a
    # one-count change above 2**53 can otherwise disappear even at zero tolerance.
    if isinstance(expected, (bool, np.bool_)) or isinstance(actual, (bool, np.bool_)):
        equal = (isinstance(expected, (bool, np.bool_))
                 and isinstance(actual, (bool, np.bool_))
                 and bool(actual) == bool(expected))
        return [] if equal else [f'{path}: boolean scalar {actual!r} != {expected!r} (exact comparison)']
    if isinstance(expected, numbers.Integral) or isinstance(actual, numbers.Integral):
        equal = (isinstance(expected, numbers.Integral)
                 and isinstance(actual, numbers.Integral)
                 and int(actual) == int(expected))
        return [] if equal else [f'{path}: integral scalar {actual!r} != {expected!r} (exact comparison)']
    if isinstance(expected, numbers.Real) and isinstance(actual, numbers.Real):
        return [] if math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol) else [f'{path}: {actual!r} != {expected!r}']
    return [] if actual == expected else [f'{path}: {actual!r} != {expected!r}']


def _input_changes(actual, expected, path='inputs'):
    import torch
    if isinstance(expected, torch.nn.Module):
        if not isinstance(actual, torch.nn.Module) or type(actual) is not type(expected):
            return [f'{path}: module type changed']
        return compare_values(actual.state_dict(), expected.state_dict(), atol=0, rtol=0, path=path) + ([] if actual.training == expected.training else [f'{path}: training flag changed'])
    if isinstance(expected, dict) and isinstance(actual, dict) and actual.keys() == expected.keys():
        return [error for key in expected for error in _input_changes(actual[key], expected[key], f'{path}.{key}')]
    if isinstance(expected, (list, tuple)) and isinstance(actual, type(expected)) and len(actual) == len(expected):
        return [error for i, (a, e) in enumerate(zip(actual, expected)) for error in _input_changes(a, e, f'{path}[{i}]')]
    return compare_values(actual, expected, atol=0, rtol=0, path=path)


@dataclass
class EvalCase:
    name: str
    stage: str
    purpose: str
    inputs: dict[str, Any]
    expected: Any = None
    expected_exception: type[Exception] | tuple[type[Exception], ...] | None = None
    comparison: Callable[[Any, Any], list[str]] = compare_values
    generation: dict[str, Any] | None = None
    _original_inputs: dict[str, Any] = field(init=False, repr=False)
    _original_expected: Any = field(init=False, repr=False)

    def __post_init__(self):
        if self.stage not in _STAGE_RANK:
            raise ValueError(f'unknown stage {self.stage!r}')
        self._original_inputs = _snapshot(self.inputs)
        self._original_expected = _snapshot(self.expected)


@dataclass
class EvalResult:
    name: str
    inputs: dict[str, Any]
    actual: Any
    expected: Any
    passed: bool
    diagnostics: list[str]
    exception: Exception | None = None
    expected_exception: Any = None
    expectation_inputs: dict[str, Any] | None = None
    observed_inputs: dict[str, Any] | None = None


def compare(case: EvalCase, actual: Any = None, *, exception: Exception | None = None) -> EvalResult:
    """Record an outcome against the original fixture, rejecting stale data.

    After modifying case inputs, rebuild the case via its factory before comparing.
    Expected display objects may be edited; private snapshots retain the oracle.
    Manual results record inputs observed at comparison time, plus the original
    expectation inputs. Replay additionally records inputs before the call, so
    a mutating answer can be diagnosed from both snapshots.
    """
    diagnostics = _input_changes(case.inputs, case._original_inputs)
    if diagnostics:
        diagnostics.insert(0, 'Inputs changed: rebuild the case to obtain matching expectations.')
    if case.expected_exception is not None:
        if exception is None:
            diagnostics.append(f'Expected {case.expected_exception}, implementation returned normally')
        elif not isinstance(exception, case.expected_exception):
            diagnostics.append(f'Expected {case.expected_exception}, received {type(exception).__name__}: {exception}')
    elif exception is not None:
        diagnostics.append(f'Unexpected {type(exception).__name__}: {exception}')
    else:
        diagnostics.extend(case.comparison(actual, case._original_expected))
    return EvalResult(case.name, _snapshot(case.inputs), _snapshot(actual),
                      _snapshot(case._original_expected), not diagnostics, diagnostics,
                      exception, case.expected_exception,
                      expectation_inputs=_snapshot(case._original_inputs),
                      observed_inputs=_snapshot(case.inputs))


def get_eval_data(eval_file: str | Path, *, stage: str | None = None) -> dict[str, EvalCase]:
    """Retrieve fresh examples from a reference-free make_eval_data() factory."""
    if stage is not None and stage not in _STAGE_RANK:
        raise ValueError(f'unknown stage {stage!r}')
    path = _find_eval_file(eval_file)
    _, _, namespace = _load_verifier_context(path)
    factory = namespace.get('make_eval_data')
    if not callable(factory):
        raise ValueError(f'{path.name}: verifier section needs make_eval_data()')
    data = factory()
    if not isinstance(data, dict) or any(not isinstance(case, EvalCase) or name != case.name for name, case in data.items()):
        raise ValueError('make_eval_data() must return a dictionary keyed by EvalCase.name')
    return {name: case for name, case in data.items() if stage is None or _STAGE_RANK[case.stage] <= _STAGE_RANK[stage]}


def replay(eval_file: str | Path, implementation: Callable, *, names=None, stage=None) -> dict[str, EvalResult]:
    """Call the current answer on fresh cases; keep failures instead of raising."""
    data = get_eval_data(eval_file, stage=stage)
    selected = list(data) if names is None else list(names)
    unknown = set(selected) - data.keys()
    if unknown:
        raise KeyError(f'unknown or excluded eval cases: {sorted(unknown)}')
    results = {}
    for name in selected:
        # Reconstruct each case independently, including modules and their state.
        case = get_eval_data(eval_file, stage=stage)[name]
        before = _snapshot(case.inputs)
        try:
            actual = implementation(**case.inputs)
        except Exception as error:
            results[name] = compare(case, exception=error)
        else:
            results[name] = compare(case, actual)
        results[name].inputs = before
    return results
