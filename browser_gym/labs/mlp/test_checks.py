from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from browser_gym.build_catalog import build
from browser_gym.check_runtime import run_check
from browser_gym.labs.mlp import checks, fixtures, mutants, reference, starter


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = ("init", "forward", "backward", "train", "inference", "evaluate", "state", "diagnose")
LEARNER_FUNCTIONS = (
    "init_params",
    "forward",
    "loss_and_grads",
    "train_epoch",
    "predict_proba",
    "evaluate_model",
    "save_state",
    "restore_state",
    "diagnose_mlp_defect",
)


def _candidate(module) -> dict[str, object]:
    return {name: getattr(module, name) for name in LEARNER_FUNCTIONS}


def test_mlp_reference_checkpoints():
    data = fixtures.load_dataset()
    assert fixtures.FIXTURE_ID == "mlp2d-complete-v1"
    assert data["train"][0].shape == (96, 2)
    assert data["valid"][0].shape == (48, 2)
    assert data["test"][0].shape == (48, 2)
    assert data["train"][0].dtype == np.float64
    assert data["train"][1].dtype == np.int64
    assert np.allclose(data["train"][0][0], np.array([-0.219205, 0.916334]), atol=1e-6)
    assert int(data["train"][1][0]) == 1

    candidate = _candidate(reference)
    for checkpoint in CHECKPOINTS:
        result = checks.run_checkpoint(checkpoint, candidate)
        assert result["status"] == "passed", json.dumps(result, indent=2)
        assert result["checkpoint"] == checkpoint
        assert result["fixture_id"] == fixtures.FIXTURE_ID
        assert not result["reference_overrides"]
        assert result["evidence"]

    init_result = checks.run_checkpoint("init", candidate)
    assert init_result["evidence"]["param_shapes"] == {
        "W1": [2, 5],
        "b1": [5],
        "W2": [5, 3],
        "b2": [3],
    }

    forward_result = checks.run_checkpoint("forward", candidate)
    assert forward_result["evidence"]["logits_shape"] == [8, 3]
    assert forward_result["evidence"]["hidden_shape"] == [8, 5]

    backward_result = checks.run_checkpoint("backward", candidate)
    blocks = {item["param"] for item in backward_result["evidence"]["gradient_checks"]}
    assert blocks == {"W1", "b1", "W2", "b2"}
    assert max(item["abs_error"] for item in backward_result["evidence"]["gradient_checks"]) < 1e-6

    train_result = checks.run_checkpoint("train", candidate)
    train_evidence = train_result["evidence"]
    assert train_evidence["batch_size"] == 16
    assert train_evidence["batch_count"] == 6
    assert train_evidence["final_train_loss"] < train_evidence["initial_train_loss"] - 0.01
    assert 0.0 <= train_evidence["valid_accuracy"] <= 1.0
    assert train_evidence["param_delta_norm"] > 0.01

    inference_result = checks.run_checkpoint("inference", candidate)
    inference_evidence = inference_result["evidence"]
    assert inference_evidence["probabilities_shape"] == [48, 3]
    assert inference_evidence["max_batch_difference"] < 1e-12
    assert inference_evidence["max_row_sum_error"] < 1e-12

    evaluate_result = checks.run_checkpoint("evaluate", candidate)
    evaluate_evidence = evaluate_result["evidence"]
    assert evaluate_evidence["split"] == "test"
    assert evaluate_evidence["n"] == 48
    assert evaluate_evidence["parameter_mutation"] is False
    assert 0.0 <= evaluate_evidence["accuracy"] <= 1.0
    assert evaluate_evidence["loss"] > 0.0

    state_result = checks.run_checkpoint("state", candidate)
    state_evidence = state_result["evidence"]
    assert state_evidence["parameters_restored"] is True
    assert state_evidence["optimizer_state_restored"] is True
    assert state_evidence["snapshot_is_independent"] is True

    diagnose_result = checks.run_checkpoint("diagnose", candidate)
    diagnose_evidence = diagnose_result["evidence"]
    assert diagnose_evidence["defect_id"] == "omitted_batch_mean"
    assert diagnose_evidence["candidate_max_abs_error"] < 1e-6
    assert diagnose_evidence["broken_max_abs_error"] > 1.0
    assert diagnose_evidence["visible_symptom"]["broken_final_valid_loss"] > (
        diagnose_evidence["visible_symptom"]["corrected_final_valid_loss"] + 1.0
    )
    assert diagnose_evidence["diagnosis"]["defect_id"] == "omitted_batch_mean"
    assert "mini-batch" in diagnose_evidence["repair_task"]


def test_mlp_mutant_and_substitution():
    for checkpoint in CHECKPOINTS:
        result = checks.run_checkpoint(checkpoint, _candidate(starter))
        assert result["status"] == "failed"
        assert result["checks"][0]["status"] == "failed"

    result = checks.run_checkpoint("backward", _candidate(mutants))
    assert result["status"] == "failed"
    failed = [item for item in result["checks"] if item["status"] == "failed"]
    assert failed
    assert any("gradient magnitude" in item["error"] for item in failed)
    assert result["evidence"]["gradient_checks"][0]["abs_error"] > 0.01
    assert "mini-batch mean" in mutants.OMITTED_BATCH_MEAN_DIAGNOSIS["diagnosis"]
    assert "Directional finite differences" in mutants.OMITTED_BATCH_MEAN_DIAGNOSIS["failed_invariant"]
    assert "backward checkpoint" in mutants.OMITTED_BATCH_MEAN_DIAGNOSIS["regression_check"]

    backward_result = checks.run_checkpoint(
        "backward",
        {"loss_and_grads": starter.loss_and_grads},
        reference_overrides={"loss_and_grads"},
    )

    assert backward_result["status"] == "passed"
    assert backward_result["reference_overrides"] == ["loss_and_grads"]
    assert backward_result["owned_checkpoint_passed"] is False
    assert backward_result["substitutions"] == [
        {"function": "loss_and_grads", "checkpoint": "backward", "independent_credit": False}
    ]
    assert any(item["name"] == "substitution_loss_and_grads" for item in backward_result["checks"])

    train_result = checks.run_checkpoint(
        "train",
        {"train_epoch": reference.train_epoch},
        reference_overrides={"loss_and_grads"},
    )
    assert train_result["status"] == "passed"
    assert train_result["reference_overrides"] == ["loss_and_grads"]
    assert train_result["owned_checkpoint_passed"] is True
    assert train_result["substitutions"] == [
        {"function": "loss_and_grads", "checkpoint": "backward", "independent_credit": False}
    ]


def test_mlp_diagnosis_checkpoint_rejects_planted_defect_and_records_substitutions():
    planted = checks.run_checkpoint("diagnose", _candidate(mutants))

    assert planted["status"] == "failed"
    assert planted["checkpoint"] == "diagnose"
    assert planted["evidence"]["defect_id"] == "omitted_batch_mean"
    assert planted["evidence"]["candidate_max_abs_error"] > 1.0
    assert planted["evidence"]["broken_max_abs_error"] > 1.0
    assert planted["evidence"]["visible_symptom"]["broken_final_valid_loss"] > (
        planted["evidence"]["visible_symptom"]["corrected_final_valid_loss"] + 1.0
    )
    assert any("omitted mini-batch mean" in item["error"] for item in planted["checks"])
    assert "divide dlogits by the mini-batch size" in planted["evidence"]["repair_task"]

    corrected = checks.run_checkpoint("diagnose", _candidate(reference))
    assert corrected["status"] == "passed", json.dumps(corrected, indent=2)
    assert corrected["owned_checkpoint_passed"] is True
    assert corrected["evidence"]["candidate_max_abs_error"] < 1e-6

    substituted = checks.run_checkpoint(
        "diagnose",
        {"diagnose_mlp_defect": reference.diagnose_mlp_defect},
        reference_overrides={"loss_and_grads"},
    )
    assert substituted["status"] == "passed", json.dumps(substituted, indent=2)
    assert substituted["reference_overrides"] == ["loss_and_grads"]
    assert substituted["owned_checkpoint_passed"] is False
    assert substituted["substitutions"] == [
        {"function": "loss_and_grads", "checkpoint": "backward", "independent_credit": False}
    ]


def test_mlp_browser_bundle_runs_in_isolated_runtime(tmp_path):
    entries = build(ROOT, tmp_path, ["mlp"])
    assert entries[0]["id"] == "mlp"

    reference_source = (ROOT / "browser_gym/labs/mlp/reference.py").read_text()
    starter_source = (ROOT / "browser_gym/labs/mlp/starter.py").read_text()
    mutant_source = (ROOT / "browser_gym/labs/mlp/mutants.py").read_text()

    for stage in CHECKPOINTS:
        passed = run_check(reference_source, "mlp", tmp_path, stage=stage)
        assert passed["status"] == "passed", json.dumps(passed, indent=2)
        assert passed["fixture_id"] == fixtures.FIXTURE_ID
        assert passed["checkpoint"] == stage

    backward = run_check(reference_source, "mlp", tmp_path, stage="backward")
    assert backward["checks"][0]["evidence"]["logits_shape"] == [8, 3]
    assert backward["checks"][0]["evidence"]["gradient_checks"]

    direct_reference = run_check(
        "from browser_gym.labs.mlp.reference import init_params\n",
        "mlp",
        tmp_path,
        stage="init",
    )
    assert direct_reference["status"] == "failed"
    assert "ImportError" in direct_reference["error"]
    assert "reference component" in direct_reference["error"]

    lazy_reference = run_check(
        """
def init_params(rng, input_dim, hidden_dim, classes):
    from browser_gym.labs.mlp.reference import init_params as ref_init_params
    return ref_init_params(rng, input_dim, hidden_dim, classes)
""",
        "mlp",
        tmp_path,
        stage="init",
    )
    assert lazy_reference["status"] == "failed"
    assert "ImportError" in lazy_reference["checks"][0]["error"]
    assert "reference component" in lazy_reference["checks"][0]["error"]

    importlib_reference = run_check(
        """
import importlib


def init_params(rng, input_dim, hidden_dim, classes):
    reference = importlib.import_module("browser_gym.labs.mlp.reference")
    return reference.init_params(rng, input_dim, hidden_dim, classes)
""",
        "mlp",
        tmp_path,
        stage="init",
    )
    assert importlib_reference["status"] == "failed"
    assert "ImportError" in importlib_reference["checks"][0]["error"]
    assert "reference component" in importlib_reference["checks"][0]["error"]

    continued = run_check(
        """
import numpy as np

REFERENCE_OVERRIDES = {"loss_and_grads"}
MOMENTUM = 0.9


def train_epoch(X, y, params, state, rng, batch_size, learning_rate):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    next_params = {name: np.asarray(value, dtype=np.float64).copy() for name, value in params.items()}
    raw_velocity = state.get("velocity", {}) if isinstance(state, dict) else {}
    step = int(state.get("step", 0)) if isinstance(state, dict) else 0
    velocity = {
        name: np.asarray(raw_velocity.get(name, np.zeros_like(value)), dtype=np.float64).copy()
        for name, value in next_params.items()
    }
    order = rng.permutation(len(X))
    batch_losses = []
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        loss, grads = loss_and_grads(X[indices], y[indices], next_params)
        batch_losses.append(float(loss))
        for name in next_params:
            velocity[name] = MOMENTUM * velocity[name] + np.asarray(grads[name], dtype=np.float64)
            next_params[name] = next_params[name] - learning_rate * velocity[name]
        step += 1
    return next_params, {"step": step, "velocity": velocity}, {
        "batch_losses": np.asarray(batch_losses, dtype=np.float64),
        "mean_loss": float(np.mean(batch_losses)),
        "batch_count": len(batch_losses),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
    }
""",
        "mlp",
        tmp_path,
        stage="train",
    )
    assert continued["status"] == "passed", json.dumps(continued, indent=2)
    assert continued["reference_overrides"] == ["loss_and_grads"]
    assert continued["substitutions"][0]["checkpoint"] == "backward"
    assert continued["owned_checkpoint_passed"] is True

    failed = run_check(starter_source, "mlp", tmp_path, stage="forward")
    assert failed["status"] == "failed"
    assert "NotImplementedError" in failed["checks"][0]["error"]

    mutant = run_check(mutant_source, "mlp", tmp_path, stage="backward")
    assert mutant["status"] == "failed"
    assert "gradient magnitude" in mutant["checks"][0]["error"]

    planted = run_check(mutant_source, "mlp", tmp_path, stage="diagnose")
    assert planted["status"] == "failed"
    assert "omitted mini-batch mean" in planted["checks"][0]["error"]
    assert planted["evidence"]["defect_id"] == "omitted_batch_mean"
    assert planted["evidence"]["candidate_max_abs_error"] > 1.0

    corrected = run_check(reference_source, "mlp", tmp_path, stage="diagnose")
    assert corrected["status"] == "passed", json.dumps(corrected, indent=2)
    assert corrected["evidence"]["candidate_max_abs_error"] < 1e-6

    substituted = run_check(
        """
REFERENCE_OVERRIDES = {"loss_and_grads"}


def diagnose_mlp_defect():
    return {
        "defect_id": "omitted_batch_mean",
        "diagnosis": "The planted backward pass omits the mini-batch mean scale for dlogits.",
        "failed_invariant": "Directional finite differences must match every parameter-block gradient.",
        "evidence": "The broken analytic directional derivatives are about the mini-batch size larger than finite differences.",
        "repair": "Divide dlogits by the mini-batch size before computing W1, b1, W2, and b2 gradients.",
        "regression_check": "Run the backward checkpoint and require abs_error below 1e-6 for every block.",
    }
""",
        "mlp",
        tmp_path,
        stage="diagnose",
    )
    assert substituted["status"] == "passed", json.dumps(substituted, indent=2)
    assert substituted["reference_overrides"] == ["loss_and_grads"]
    assert substituted["substitutions"] == [
        {"function": "loss_and_grads", "checkpoint": "backward", "independent_credit": False}
    ]

    isolated = run_check("print(shape_probe)\n" + reference_source, "mlp", tmp_path, stage="init")
    assert isolated["status"] == "failed"
    assert "NameError" in isolated["error"]


def test_mlp_notebook_is_derived_from_canonical_sources(tmp_path):
    entries = build(ROOT, tmp_path, ["mlp"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())

    assert notebook["metadata"]["dojo"]["exercise_id"] == "mlp"
    solution_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    exploration_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    ]
    assert len(solution_cells) == 1
    assert "".join(solution_cells[0]["source"]).strip() == (
        ROOT / "browser_gym/labs/mlp/starter.py"
    ).read_text().strip()
    assert all(len(cell["source"]) <= 3 for cell in exploration_cells)
    lesson_source = (tmp_path / "lessons" / "mlp-0.py").read_text()
    assert 'tensor_card("train_X"' in lesson_source
    assert "Gradient block norms" in lesson_source
    assert "data-testid=\"dojo-mlp-loss-run\"" in lesson_source
    assert "data-testid=\"dojo-mlp-boundary-run\"" in lesson_source
    assert "data-testid=\"dojo-mlp-confidence-run\"" in lesson_source
    assert "data-testid=\"dojo-mlp-gradient-run\"" in lesson_source
    assert "data-testid=\"dojo-mlp-diagnosis-run\"" in lesson_source
    assert "data-testid=\"dojo-mlp-diagnosis-plot\"" in lesson_source
    assert "Planted defect: omitted_batch_mean" in lesson_source
    assert "shape_probe" in lesson_source


def test_documented_native_check_command_works():
    readme = (ROOT / "browser_gym/labs/mlp/README.md").read_text()
    command = "python -m browser_gym.labs.mlp.checks --candidate reference --checkpoint all"
    assert command in readme

    completed = subprocess.run(
        [sys.executable, "-m", "browser_gym.labs.mlp.checks", "--candidate", "reference", "--checkpoint", "all"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "init: passed" in completed.stdout
    assert "forward: passed" in completed.stdout
    assert "backward: passed" in completed.stdout
    assert "train: passed" in completed.stdout
    assert "inference: passed" in completed.stdout
    assert "evaluate: passed" in completed.stdout
    assert "state: passed" in completed.stdout
    assert "diagnose: passed" in completed.stdout
