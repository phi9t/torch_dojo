"""Run one browser-gym answer against a generated canonical checker bundle."""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
import types
import builtins
from contextlib import contextmanager
from pathlib import Path

STAGES = {"baseline": 0, "followup": 1, "production": 2}
MLP_CHECKPOINTS = {
    "init": 0,
    "forward": 1,
    "backward": 2,
    "train": 3,
    "inference": 4,
    "evaluate": 5,
    "state": 6,
    "diagnose": 7,
}
_MISSING = object()
_BLOCKED_MLP_REFERENCE_MODULES = (
    "browser_gym.labs.mlp.reference",
    "browser_gym.labs.mlp.checks",
)
_REFERENCE_IMPORT_MESSAGE = (
    "MLP reference components are not importable from submitted source; "
    "declare REFERENCE_OVERRIDES for supplied components instead"
)


def run_check(
    source: str,
    exercise_id: str,
    bundle_dir: Path,
    *,
    stage: str = "baseline",
    reference_overrides: set[str] | None = None,
) -> dict:
    matches = list((bundle_dir / "checkers").glob(f"{exercise_id}_*.json"))
    if len(matches) != 1:
        raise ValueError(f"{exercise_id}: expected one checker bundle")
    bundle = json.loads(matches[0].read_text())
    return _run_bundle(source, bundle, runtime="python", stage=stage, reference_overrides=reference_overrides)


def run_check_bundle(
    source: str,
    bundle: dict,
    stage: str = "baseline",
    reference_overrides: set[str] | None = None,
) -> dict:
    """Run the same checker from a bundle already fetched by a browser worker."""
    return _run_bundle(source, bundle, stage=stage, reference_overrides=reference_overrides)


def _package_identity(runtime: str) -> dict[str, str]:
    identity = {"python": platform.python_version(), "runtime": runtime}
    try:
        import numpy as np
    except Exception:
        pass
    else:
        identity["numpy"] = np.__version__
    return identity


def _pack_value(value, depth=0, budget=None):
    """Bounded, JSON-safe snapshots. Unsupported values are never repr/eval replayed."""
    import numpy as np
    if budget is None:
        budget = [2048]
    budget[0] -= 1
    if depth > 5 or budget[0] < 0:
        return {"kind": "preview", "text": "Nested value omitted"}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufUS" or value.size > 256:
            return {"kind": "preview", "text": f"array {value.shape} · {value.dtype} (preview only)"}
        return {"kind": "array", "shape": list(value.shape), "dtype": str(value.dtype),
                "data": _pack_value(value.tolist(), depth + 1, budget)}
    if isinstance(value, np.generic):
        return _pack_value(value.item(), depth, budget)
    if isinstance(value, float) and not np.isfinite(value):
        return {"kind": "float", "value": str(value)}
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return {"kind": "int", "value": str(value)}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 2000 else {"kind": "preview", "text": value[:2000] + "…"}
    if isinstance(value, (list, tuple)) and len(value) <= 256:
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [_pack_value(item, depth + 1, budget) for item in value]}
    if isinstance(value, dict) and len(value) <= 32 and all(isinstance(key, str) for key in value):
        return {"kind": "dict", "items": {key: _pack_value(item, depth + 1, budget) for key, item in value.items()}}
    return {"kind": "preview", "text": f"{type(value).__name__} (preview only)"}


def _unpack_value(value):
    import numpy as np
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind == "array":
        return np.array(_unpack_value(value["data"]), dtype=value["dtype"]).reshape(value["shape"])
    if kind == "float":
        return float(value["value"])
    if kind == "int":
        return int(value["value"])
    if kind in ("list", "tuple"):
        items = [_unpack_value(item) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {key: _unpack_value(item) for key, item in value["items"].items()}
    raise ValueError("This sample contains preview-only inputs")


class _CallSamples:
    """Keep the first three and most recent three actual calls per verifier."""
    def __init__(self, candidate, name):
        self.candidate = candidate
        self.__globals__ = getattr(candidate, "__globals__", {})
        self.name = name
        self.calls = []
        self.total = 0
        import inspect
        try:
            self.argument_names = list(inspect.signature(candidate).parameters)
        except (TypeError, ValueError):
            self.argument_names = []

    def __call__(self, *args, **kwargs):
        self.total += 1
        record = {"function": self.name, "call": self.total,
                  "args": _pack_value(args), "kwargs": _pack_value(kwargs),
                  "argument_names": self.argument_names}
        try:
            output = self.candidate(*args, **kwargs)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"[:2000]
            raise
        else:
            record["actual"] = _pack_value(output)
            return output
        finally:
            self.calls.append(record)
            if len(self.calls) > 6:
                del self.calls[3]

    def finish(self, reference):
        for record in self.calls:
            try:
                args, kwargs = _unpack_value(record["args"]), _unpack_value(record["kwargs"])
            except ValueError:
                record["replayable"] = False
                continue
            record["replayable"] = True
            if reference is None:
                continue
            try:
                record["expected"] = _pack_value(reference(*args, **kwargs))
            except Exception as error:
                record["expected_error"] = f"{type(error).__name__}: {error}"[:2000]
        return self.calls


def _run_bundle(
    source: str,
    bundle: dict,
    runtime: str = "pyodide",
    stage: str = "baseline",
    reference_overrides: set[str] | None = None,
) -> dict:
    if bundle.get("kind") == "mlp_lab":
        return _run_mlp_bundle(source, bundle, runtime, stage, reference_overrides=reference_overrides)
    if stage not in STAGES:
        raise ValueError(f"unknown check stage: {stage}")
    result = {"status": "failed", "checks": [], "error": None,
              "checker_version": bundle["checker_version"], "runtime": runtime,
              "package_identity": _package_identity(runtime),
              "fixture_id": f"canonical-{bundle['exercise_id']}", "stage": stage}
    candidate_ns = {"__name__": "__dojo_candidate__"}
    try:
        exec(compile(source, "solution.py", "exec"), candidate_ns)
        candidate_name = bundle["candidate"]
        if candidate_name not in candidate_ns:
            raise ValueError(f"Missing candidate function: {candidate_name}")
        candidate = candidate_ns[candidate_name]
        verifier_ns = {"__name__": "__dojo_verifiers__"}
        if "program" in bundle:
            exec(compile(bundle["program"], "checker_program.py", "exec"), verifier_ns)
        else:  # Bundles retained by an older offline site.
            exec(compile(bundle["imports"], "checker_imports.py", "exec"), verifier_ns)
            exec(compile(bundle.get("helpers", ""), "checker_helpers.py", "exec"), verifier_ns)
        reference_ns = {"__name__": "__dojo_expected__"}
        if bundle.get("reference_source"):
            exec(compile(bundle["reference_source"], "expected.py", "exec"), reference_ns)
        reference = reference_ns.get(candidate_name)
        for item in bundle["verifiers"]:
            verifier_stage = next((name for name in STAGES if item["name"].startswith(f"verify_{name}_")), None)
            if verifier_stage is not None and STAGES[verifier_stage] > STAGES[stage]:
                continue
            if "program" not in bundle:
                exec(compile(item["source"], "checker.py", "exec"), verifier_ns)
            started = time.perf_counter()
            samples = _CallSamples(candidate, candidate_name)
            try:
                verifier_ns[item["name"]](samples)
            except Exception as error:
                result["checks"].append({"name": item["name"], "status": "failed",
                                         "error": f"{type(error).__name__}: {error}",
                                         # The innermost frames: where the check or the answer failed.
                                         "traceback": traceback.format_exc(limit=-10),
                                         "locals": _failing_check_locals(error),
                                         "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
            else:
                result["checks"].append({"name": item["name"], "status": "passed",
                                         "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
            result["checks"][-1]["samples"] = samples.finish(reference)
            result["checks"][-1]["sample_calls"] = samples.total
        if not result["checks"]:
            raise ValueError(f"no verifiers available for {stage!r} stage")
        if all(item["status"] == "passed" for item in result["checks"]):
            result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc(limit=6)
    return result


def _failing_check_locals(error: BaseException) -> dict[str, str]:
    """The failing check's variables at the line that failed, as short reprs:
    the example's inputs and the values it compared."""
    frame_locals = None
    tb = error.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename in ("checker.py", "checker_helpers.py", "checker_program.py"):
            frame_locals = tb.tb_frame.f_locals
        tb = tb.tb_next
    shown: dict[str, str] = {}
    for name, value in (frame_locals or {}).items():
        if name.startswith("_") or callable(value) or isinstance(value, types.ModuleType):
            continue
        try:
            text = repr(value)
        except Exception:
            continue
        shown[name] = text if len(text) <= 600 else text[:600] + " …"
    return shown


def _ensure_package(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(_ensure_package(parent), child, module)
    return module


def _install_module(name: str, source: str) -> types.ModuleType:
    parent, _, child = name.rpartition(".")
    if parent:
        _ensure_package(parent)
    module = types.ModuleType(name)
    module.__file__ = f"{name.replace('.', '/')}.py"
    module.__package__ = parent
    sys.modules[name] = module
    if parent:
        setattr(sys.modules[parent], child, module)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _mlp_module_names(bundle: dict) -> tuple[str, ...]:
    modules = bundle.get("modules", {})
    return tuple(
        name for name in (
            "browser_gym.labs.mlp.fixtures",
            "browser_gym.labs.mlp.reference",
            "browser_gym.labs.mlp.checks",
        )
        if name in modules
    )


def _mlp_candidate_module_names(bundle: dict) -> tuple[str, ...]:
    modules = bundle.get("modules", {})
    return tuple(name for name in ("browser_gym.labs.mlp.fixtures",) if name in modules)


def _mlp_checker_module_names(bundle: dict) -> tuple[str, ...]:
    modules = bundle.get("modules", {})
    return tuple(
        name for name in (
            "browser_gym.labs.mlp.reference",
            "browser_gym.labs.mlp.checks",
        )
        if name in modules
    )


def _snapshot_modules(names: tuple[str, ...]) -> tuple[dict[str, object], dict[tuple[str, str], object]]:
    module_snapshot = {name: sys.modules.get(name, _MISSING) for name in names}
    parent_snapshot: dict[tuple[str, str], object] = {}
    for name in names:
        parent, _, child = name.rpartition(".")
        parent_module = sys.modules.get(parent)
        if parent_module is not None:
            parent_snapshot[(parent, child)] = getattr(parent_module, child, _MISSING)
    return module_snapshot, parent_snapshot


def _restore_modules(
    module_snapshot: dict[str, object],
    parent_snapshot: dict[tuple[str, str], object],
) -> None:
    for name, module in module_snapshot.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    for (parent, child), value in parent_snapshot.items():
        parent_module = sys.modules.get(parent)
        if parent_module is None:
            continue
        if value is _MISSING:
            try:
                delattr(parent_module, child)
            except AttributeError:
                pass
        else:
            setattr(parent_module, child, value)


def _install_mlp_bundle_modules(bundle: dict, names: tuple[str, ...] | None = None) -> None:
    _ensure_package("browser_gym")
    _ensure_package("browser_gym.labs")
    _ensure_package("browser_gym.labs.mlp")
    modules = bundle.get("modules", {})
    for name in _mlp_module_names(bundle) if names is None else names:
        _install_module(name, modules[name])


def _is_blocked_mlp_reference_name(name: str) -> bool:
    return any(
        name == blocked or name.startswith(f"{blocked}.")
        for blocked in _BLOCKED_MLP_REFERENCE_MODULES
    )


def _is_blocked_mlp_import_request(name: str, fromlist: object) -> bool:
    if _is_blocked_mlp_reference_name(name):
        return True
    if name == "browser_gym.labs.mlp":
        return any(item in {"reference", "checks"} for item in fromlist or ())
    return False


def _mlp_candidate_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and _is_blocked_mlp_import_request(name, fromlist):
        raise ImportError(_REFERENCE_IMPORT_MESSAGE)
    return builtins.__import__(name, globals, locals, fromlist, level)


def _mlp_candidate_builtins() -> dict[str, object]:
    candidate_builtins = dict(vars(builtins))
    candidate_builtins["__import__"] = _mlp_candidate_import
    return candidate_builtins


class _BlockedMlpReferenceFinder:
    def find_spec(self, fullname, path=None, target=None):
        if _is_blocked_mlp_reference_name(fullname):
            raise ImportError(_REFERENCE_IMPORT_MESSAGE)
        return None


@contextmanager
def _hidden_modules(names: tuple[str, ...]):
    module_snapshot, parent_snapshot = _snapshot_modules(names)
    try:
        for name in names:
            sys.modules.pop(name, None)
            parent, _, child = name.rpartition(".")
            parent_module = sys.modules.get(parent)
            if parent_module is not None:
                try:
                    delattr(parent_module, child)
                except AttributeError:
                    pass
        yield
    finally:
        _restore_modules(module_snapshot, parent_snapshot)


@contextmanager
def _blocked_mlp_reference_imports():
    finder = _BlockedMlpReferenceFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def _source_reference_overrides(candidate_ns: dict[str, object]) -> set[str]:
    value = candidate_ns.get("REFERENCE_OVERRIDES", ())
    if value is None:
        return set()
    if isinstance(value, str):
        raise TypeError("REFERENCE_OVERRIDES must be an iterable of function names, not a string")
    try:
        return {str(item) for item in value}
    except TypeError as error:
        raise TypeError("REFERENCE_OVERRIDES must be an iterable of function names") from error


def _run_mlp_bundle(
    source: str,
    bundle: dict,
    runtime: str,
    stage: str,
    reference_overrides: set[str] | None = None,
) -> dict:
    if stage not in MLP_CHECKPOINTS:
        raise ValueError(f"unknown MLP checkpoint: {stage}")
    result = {"status": "failed", "checks": [], "error": None,
              "checker_version": bundle["checker_version"], "runtime": runtime,
              "package_identity": _package_identity(runtime),
              "fixture_id": bundle.get("fixture_id", f"canonical-{bundle['exercise_id']}"),
              "stage": stage, "checkpoint": stage, "evidence": {},
              "reference_overrides": [], "substitutions": []}
    candidate_ns = {"__name__": "__dojo_candidate__", "__builtins__": _mlp_candidate_builtins()}
    module_snapshot, parent_snapshot = _snapshot_modules(_mlp_module_names(bundle))
    try:
        _install_mlp_bundle_modules(bundle, _mlp_candidate_module_names(bundle))
        with _hidden_modules(_BLOCKED_MLP_REFERENCE_MODULES), _blocked_mlp_reference_imports():
            exec(compile(source, "solution.py", "exec"), candidate_ns)
        candidate = {
            name: candidate_ns[name]
            for name in bundle.get("candidate_functions", ())
            if name in candidate_ns
        }
        overrides = set(reference_overrides or set()) | _source_reference_overrides(candidate_ns)
        _install_mlp_bundle_modules(bundle, _mlp_checker_module_names(bundle))
        checks = sys.modules["browser_gym.labs.mlp.checks"]
        collectors = {name: _CallSamples(fn, name) for name, fn in candidate.items() if callable(fn)}
        observed_candidate = {**candidate, **collectors}
        with _hidden_modules(_BLOCKED_MLP_REFERENCE_MODULES), _blocked_mlp_reference_imports():
            checkpoint_result = checks.run_checkpoint(stage, observed_candidate, reference_overrides=overrides)
        reference_module = sys.modules["browser_gym.labs.mlp.reference"]
        owned_check = checkpoint_result["checks"][-1]
        owned_check["samples"] = [sample for name, collector in collectors.items()
                                  for sample in collector.finish(getattr(reference_module, name, None))]
        owned_check["sample_calls"] = sum(collector.total for collector in collectors.values())
        result.update({
            "status": checkpoint_result["status"],
            "checks": checkpoint_result["checks"],
            "fixture_id": checkpoint_result["fixture_id"],
            "evidence": checkpoint_result["evidence"],
            "reference_overrides": checkpoint_result["reference_overrides"],
            "substitutions": checkpoint_result["substitutions"],
            "owned_checkpoint_passed": checkpoint_result["owned_checkpoint_passed"],
        })
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc(limit=6)
    finally:
        _restore_modules(module_snapshot, parent_snapshot)
    return result
