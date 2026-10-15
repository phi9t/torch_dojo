"""Run native Python verifiers from an exercise's reference eval script."""

from __future__ import annotations

import inspect
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from torch_judge.scripts import TaggedScript


def _find_eval_file(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    caller = inspect.currentframe().f_back.f_back
    caller_file = caller.f_globals.get("__file__") if caller else None
    roots = [Path(caller_file).parent] if caller_file else []
    roots.extend((Path.cwd(), Path.cwd() / "dojo"))
    for root in roots:
        candidate = root / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"eval script not found: {value}")


_STAGE_RANK = {"baseline": 0, "followup": 1, "production": 2}


def _verifier_stage(name: str) -> str | None:
    for stage in _STAGE_RANK:
        if name.startswith(f"verify_{stage}_"):
            return stage
    return None


def _selected_verifiers(verifiers: list[tuple[str, Any]], stage: str | None) -> list[tuple[str, Any]]:
    if stage is None:
        return verifiers
    if stage not in _STAGE_RANK:
        raise ValueError(f"unknown stage {stage!r}; choose baseline, followup, or production")
    selected = []
    for name, verifier in verifiers:
        verifier_stage = _verifier_stage(name)
        if verifier_stage is None or _STAGE_RANK[verifier_stage] <= _STAGE_RANK[stage]:
            selected.append((name, verifier))
    if not selected:
        raise ValueError(f"no verifiers available for {stage!r} stage")
    return selected


def _load_verifier_context(path: Path) -> tuple[str, list[tuple[str, Any]], dict[str, Any]]:
    """Load the shared selected program without running reference code."""
    program = TaggedScript(path.read_text(), path).verifiers()
    namespace: dict[str, Any] = {"__name__": "__dojo_verifiers__", "__file__": str(path)}
    exec(program.compile(), namespace)
    return program.candidate, [(name, namespace[name]) for name in program.verifier_names], namespace


def _load_verifiers(path: Path) -> tuple[str, list[tuple[str, Any]]]:
    name, verifiers, _ = _load_verifier_context(path)
    return name, verifiers


def _user_namespace() -> dict[str, Any]:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            return shell.user_ns
    except ImportError:
        pass
    frame = inspect.currentframe()
    return frame.f_back.f_back.f_globals if frame and frame.f_back and frame.f_back.f_back else {}


def check(eval_file: str | Path, implementation: Any | None = None, *, stage: str | None = None) -> bool:
    """Test a live answer or explicit reference against native verifiers."""
    path = _find_eval_file(eval_file)
    name, all_verifiers = _load_verifiers(path)
    verifiers = _selected_verifiers(all_verifiers, stage)
    if implementation is None:
        namespace = _user_namespace()
        if name not in namespace:
            print(f"Missing {name}; run the implementation section first.")
            return False
        implementation = namespace[name]

    passed = 0
    print(f"Testing {path.stem.removesuffix('_eval')}: {len(verifiers)} checks")
    for label, verifier in verifiers:
        started = time.perf_counter()
        try:
            verifier(implementation)
        except Exception as error:
            print(f"  FAIL {label}: {type(error).__name__}: {error}")
            frames = traceback.extract_tb(error.__traceback__)
            local_frames = [frame for frame in frames if Path(frame.filename).parent == path.parent]
            if local_frames:
                frame = local_frames[-1]
                source_lines = Path(frame.filename).read_text().splitlines()
                line = source_lines[frame.lineno - 1].strip() if frame.lineno <= len(source_lines) else ""
                if line:
                    print(f"    {Path(frame.filename).name}:{frame.lineno}: {line}")
            for note in getattr(error, "__notes__", ()):
                print(f"    {note}")
        else:
            passed += 1
            print(f"  PASS {label} ({(time.perf_counter() - started) * 1000:.1f} ms)")
    print(f"{passed}/{len(verifiers)} passed")
    return passed == len(verifiers)


def _candidate_from_question(eval_path: Path, name: str) -> Any:
    """Load saved setup and answer cells while skipping teaching probes."""
    question = eval_path.with_name(eval_path.name.removesuffix("_eval.py") + ".py")
    script = TaggedScript(question.read_text(), question)
    namespace: dict[str, Any] = {"__name__": "__dojo_candidate__", "__file__": str(question)}
    exec(script.answer().compile(), namespace)
    if name not in namespace:
        raise ValueError(f"{question.name}:1: answer section does not define {name}")
    return namespace[name]


def evaluate(
    eval_file: str | Path,
    reference: Any,
    *,
    candidate: bool | None = None,
    stage: str | None = None,
) -> bool:
    """Run eval-file verifiers on the reference or saved question answer.

    Direct script execution accepts ``--candidate``; explicit ``candidate=``
    also works from a kernel or another caller.
    """
    path = _find_eval_file(eval_file)
    if candidate is None:
        candidate = "--candidate" in sys.argv[1:]
    if stage is None:
        for index, argument in enumerate(sys.argv[1:], start=1):
            if argument == "--stage":
                if index + 1 >= len(sys.argv):
                    raise ValueError("--stage needs baseline, followup, or production")
                stage = sys.argv[index + 1]
                break
            if argument.startswith("--stage="):
                stage = argument.partition("=")[2]
                break
    if candidate:
        name, verifiers = _load_verifiers(path)
        if stage is None and any(_verifier_stage(label) is not None for label, _ in verifiers):
            stage = "baseline"
        implementation = _candidate_from_question(path, name)
    else:
        implementation = reference
    return check(path, implementation, stage=stage)
