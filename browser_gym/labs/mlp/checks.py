"""Native checkpoint checks for the NumPy MLP browser lab."""

from __future__ import annotations

import argparse
import importlib
import json
import traceback
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import numpy as np

from browser_gym.labs.mlp import fixtures, reference


CHECKPOINTS = ("init", "forward", "backward", "train", "inference", "evaluate", "state", "diagnose")
PARAM_SHAPES = {
    "W1": (fixtures.INPUT_DIM, fixtures.HIDDEN_DIM),
    "b1": (fixtures.HIDDEN_DIM,),
    "W2": (fixtures.HIDDEN_DIM, fixtures.CLASSES),
    "b2": (fixtures.CLASSES,),
}
OWNED_FUNCTIONS = {
    "init": ("init_params",),
    "forward": ("forward",),
    "backward": ("loss_and_grads",),
    "train": ("train_epoch",),
    "inference": ("predict_proba",),
    "evaluate": ("evaluate_model",),
    "state": ("save_state", "restore_state"),
    "diagnose": ("loss_and_grads", "diagnose_mlp_defect"),
}
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
FUNCTION_CHECKPOINT: dict[str, str] = {}
for checkpoint, function_names in OWNED_FUNCTIONS.items():
    for function_name in function_names:
        FUNCTION_CHECKPOINT.setdefault(function_name, checkpoint)
TRAIN_BATCH_SIZE = fixtures.TRAIN_BATCH_SIZE
TRAIN_LEARNING_RATE = fixtures.TRAIN_LEARNING_RATE
DIAGNOSIS_DEFECT_ID = "omitted_batch_mean"
DIAGNOSIS_REPAIR_TASK = (
    "Repair the planted backward pass: divide dlogits by the mini-batch "
    "size before computing W1, b1, W2, and b2 gradients, then rerun the "
    "diagnose checkpoint."
)


class CheckFailure(AssertionError):
    def __init__(self, message: str, evidence: dict[str, object] | None = None):
        super().__init__(message)
        self.evidence = evidence


def reference_candidate() -> dict[str, object]:
    return {function_name: getattr(reference, function_name) for function_name in LEARNER_FUNCTIONS}


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value, dtype=np.float64).copy() for name, value in params.items()}


def _assert_param_dict(params: Any) -> dict[str, np.ndarray]:
    if not isinstance(params, dict):
        raise CheckFailure("init_params must return a dict with W1, b1, W2, and b2")
    result: dict[str, np.ndarray] = {}
    for name, shape in PARAM_SHAPES.items():
        if name not in params:
            raise CheckFailure(f"missing parameter block {name}")
        array = np.asarray(params[name])
        if array.shape != shape:
            raise CheckFailure(f"{name} has shape {array.shape}, expected {shape}")
        if array.dtype != np.float64:
            raise CheckFailure(f"{name} has dtype {array.dtype}, expected float64")
        if not np.isfinite(array).all():
            raise CheckFailure(f"{name} contains non-finite values")
        result[name] = array.copy()
    return result


def _assert_same_params(before: dict[str, np.ndarray], after: dict[str, np.ndarray], where: str) -> None:
    for name, expected in before.items():
        if not np.array_equal(expected, after[name]):
            raise CheckFailure(f"{where} mutated parameter block {name}")


def _assert_optimizer_state(state: Any, params: dict[str, np.ndarray]) -> dict[str, object]:
    if not isinstance(state, dict):
        raise CheckFailure("optimizer state must be a dict")
    if "step" not in state:
        raise CheckFailure("optimizer state is missing step")
    try:
        step = int(state["step"])
    except Exception as error:
        raise CheckFailure("optimizer state step must be an integer") from error
    velocity = state.get("velocity")
    if not isinstance(velocity, dict):
        raise CheckFailure("optimizer state is missing velocity dict")
    copied_velocity = {}
    for name, param in params.items():
        if name not in velocity:
            raise CheckFailure(f"optimizer velocity is missing {name}")
        array = np.asarray(velocity[name])
        if array.shape != param.shape:
            raise CheckFailure(f"optimizer velocity {name} has shape {array.shape}, expected {param.shape}")
        if array.dtype != np.float64:
            raise CheckFailure(f"optimizer velocity {name} has dtype {array.dtype}, expected float64")
        if not np.isfinite(array).all():
            raise CheckFailure(f"optimizer velocity {name} contains non-finite values")
        copied_velocity[name] = array.copy()
    return {"step": step, "velocity": copied_velocity}


def _assert_metric_dict(metrics: Any) -> dict[str, object]:
    if not isinstance(metrics, dict):
        raise CheckFailure("training metrics must be a dict")
    return metrics


def _assert_probabilities(probabilities: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(probabilities)
    if array.shape != expected_shape:
        raise CheckFailure(f"probabilities have shape {array.shape}, expected {expected_shape}")
    if array.dtype != np.float64:
        raise CheckFailure(f"probabilities have dtype {array.dtype}, expected float64")
    if not np.isfinite(array).all():
        raise CheckFailure("probabilities contain non-finite values")
    if (array < -1e-12).any() or (array > 1.0 + 1e-12).any():
        raise CheckFailure("probabilities must stay in [0, 1]")
    return array


def _param_delta_norm(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum((after[name] - before[name]) ** 2)) for name in PARAM_SHAPES)))


def _fixture_params() -> dict[str, np.ndarray]:
    return reference.init_params(
        fixtures.seeded_rng(),
        fixtures.INPUT_DIM,
        fixtures.HIDDEN_DIM,
        fixtures.CLASSES,
    )


def _trained_reference_params(epochs: int = 8) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    X, y = fixtures.load_dataset()["train"]
    params = _fixture_params()
    state: dict[str, object] | None = None
    rng = np.random.default_rng(fixtures.TRAIN_SEED)
    for _ in range(epochs):
        params, state, _ = reference.train_epoch(
            X,
            y,
            params,
            state,
            rng,
            batch_size=TRAIN_BATCH_SIZE,
            learning_rate=TRAIN_LEARNING_RATE,
        )
    assert state is not None
    return params, state


def _check_init(candidate: dict[str, object]) -> dict[str, object]:
    fn = candidate["init_params"]
    params = _assert_param_dict(fn(
        fixtures.seeded_rng(),
        fixtures.INPUT_DIM,
        fixtures.HIDDEN_DIM,
        fixtures.CLASSES,
    ))
    expected = _fixture_params()
    for name in PARAM_SHAPES:
        if not np.allclose(params[name], expected[name], atol=1e-12, rtol=0.0):
            raise CheckFailure(f"{name} is not reproducible from the supplied RNG")
    return {
        "param_shapes": {name: list(params[name].shape) for name in PARAM_SHAPES},
        "param_norms": {name: round(float(np.linalg.norm(params[name])), 8) for name in PARAM_SHAPES},
        "seed": fixtures.INIT_SEED,
    }


def _check_forward(candidate: dict[str, object]) -> dict[str, object]:
    fn = candidate["forward"]
    batch = fixtures.checkpoint_batch()
    params = _fixture_params()
    before = _copy_params(params)
    activations = fn(batch.X, params)
    _assert_same_params(before, params, "forward")
    if not isinstance(activations, dict):
        raise CheckFailure("forward must return a dict with hidden_pre, hidden, and logits")
    expected = reference.forward(batch.X, params)
    for name, shape in {"hidden_pre": (8, fixtures.HIDDEN_DIM), "hidden": (8, fixtures.HIDDEN_DIM), "logits": (8, fixtures.CLASSES)}.items():
        if name not in activations:
            raise CheckFailure(f"forward output is missing {name}")
        observed = np.asarray(activations[name])
        if observed.shape != shape:
            raise CheckFailure(f"{name} has shape {observed.shape}, expected {shape}")
        if observed.dtype != np.float64:
            raise CheckFailure(f"{name} has dtype {observed.dtype}, expected float64")
        if not np.allclose(observed, expected[name], atol=1e-10, rtol=1e-10):
            raise CheckFailure(f"{name} values differ from the reference forward pass")
    return {
        "input_shape": list(batch.X.shape),
        "hidden_shape": list(np.asarray(activations["hidden"]).shape),
        "logits_shape": list(np.asarray(activations["logits"]).shape),
        "logits_preview": np.round(np.asarray(activations["logits"])[0], 6).tolist(),
    }


def _loss_with_param(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    return float(reference.loss_and_grads(X, y, params)[0])


def _direction(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    value = rng.normal(size=shape)
    norm = float(np.linalg.norm(value))
    return value / norm


def _check_backward(candidate: dict[str, object]) -> dict[str, object]:
    evidence = _gradient_diagnostics(candidate["loss_and_grads"])
    for item in evidence["gradient_checks"]:
        if item["abs_error"] > 1e-6:
            raise CheckFailure(
                f"gradient magnitude mismatch for {item['param']}: directional analytic "
                f"{item['analytic_directional']:.6g} vs finite-difference {item['finite_difference']:.6g}",
                evidence={
                    **evidence,
                    "hint": "Check that dlogits is divided by the mini-batch size before propagating gradients.",
                },
            )
    return {
        "loss": evidence["loss"],
        "input_shape": evidence["input_shape"],
        "hidden_shape": evidence["hidden_shape"],
        "logits_shape": evidence["logits_shape"],
        "gradient_checks": evidence["gradient_checks"],
    }


def _broken_loss_and_grads_omitted_batch_mean(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[float, dict[str, np.ndarray]]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    activations = reference.forward(X, params)
    logits = activations["logits"]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    row = np.arange(len(X))
    loss = float(np.mean(np.log(exp.sum(axis=1)) - shifted[row, y]))

    dlogits = probabilities.copy()
    dlogits[row, y] -= 1.0
    dW2 = activations["hidden"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dhidden = dlogits @ params["W2"].T
    dhidden_pre = dhidden * (activations["hidden_pre"] > 0.0)
    dW1 = X.T @ dhidden_pre
    db1 = dhidden_pre.sum(axis=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def _gradient_diagnostics(loss_and_grads) -> dict[str, object]:
    batch = fixtures.checkpoint_batch()
    params = _fixture_params()
    reference_activations = reference.forward(batch.X, params)
    before = _copy_params(params)
    loss, grads = loss_and_grads(batch.X, batch.y, params)
    _assert_same_params(before, params, "loss_and_grads")
    expected_loss, expected_grads = reference.loss_and_grads(batch.X, batch.y, params)
    if not np.isscalar(loss) or not np.isfinite(loss):
        raise CheckFailure("loss_and_grads must return a finite scalar loss")
    if not np.isclose(float(loss), expected_loss, atol=1e-10, rtol=1e-10):
        raise CheckFailure(f"loss {float(loss):.12g} differs from reference {expected_loss:.12g}")
    if not isinstance(grads, dict):
        raise CheckFailure("loss_and_grads must return a gradient dict")

    gradient_checks = []
    rng = np.random.default_rng(fixtures.GRADIENT_SEED)
    eps = 1e-5
    for name, shape in PARAM_SHAPES.items():
        if name not in grads:
            raise CheckFailure(f"missing gradient block {name}")
        grad = np.asarray(grads[name])
        if grad.shape != shape:
            raise CheckFailure(f"gradient {name} has shape {grad.shape}, expected {shape}")
        if grad.dtype != np.float64:
            raise CheckFailure(f"gradient {name} has dtype {grad.dtype}, expected float64")
        if not np.isfinite(grad).all():
            raise CheckFailure(f"gradient {name} contains non-finite values")
        direction = _direction(shape, rng)
        plus = _copy_params(params)
        minus = _copy_params(params)
        plus[name] += eps * direction
        minus[name] -= eps * direction
        numeric = (_loss_with_param(plus, batch.X, batch.y) - _loss_with_param(minus, batch.X, batch.y)) / (2.0 * eps)
        analytic = float(np.sum(grad * direction))
        expected = float(np.sum(expected_grads[name] * direction))
        gradient_checks.append({
            "param": name,
            "analytic_directional": round(analytic, 10),
            "finite_difference": round(float(numeric), 10),
            "reference_directional": round(expected, 10),
            "abs_error": round(float(abs(analytic - numeric)), 10),
            "grad_norm": round(float(np.linalg.norm(grad)), 10),
            "reference_grad_norm": round(float(np.linalg.norm(expected_grads[name])), 10),
        })
    return {
        "loss": round(float(loss), 10),
        "input_shape": list(batch.X.shape),
        "hidden_shape": list(reference_activations["hidden"].shape),
        "logits_shape": list(reference_activations["logits"].shape),
        "gradient_checks": gradient_checks,
        "max_abs_error": max(item["abs_error"] for item in gradient_checks),
    }


def _diagnosis_training_curve(loss_and_grads, epochs: int = 6) -> list[dict[str, object]]:
    train_X, train_y = fixtures.load_dataset()["train"]
    valid_X, valid_y = fixtures.load_dataset()["valid"]
    params = _fixture_params()
    velocity = {name: np.zeros_like(value) for name, value in params.items()}
    rng = np.random.default_rng(fixtures.TRAIN_SEED)
    rows = []
    for epoch in range(epochs + 1):
        train_metrics = reference.evaluate_model(train_X, train_y, params)
        valid_metrics = reference.evaluate_model(valid_X, valid_y, params)
        rows.append({
            "epoch": epoch,
            "train_loss": round(float(train_metrics["loss"]), 6),
            "valid_loss": round(float(valid_metrics["loss"]), 6),
            "valid_accuracy": round(float(valid_metrics["accuracy"]), 6),
        })
        if epoch == epochs:
            continue
        order = rng.permutation(len(train_X))
        for start in range(0, len(order), TRAIN_BATCH_SIZE):
            indices = order[start:start + TRAIN_BATCH_SIZE]
            _, grads = loss_and_grads(train_X[indices], train_y[indices], params)
            for name in params:
                velocity[name] = 0.9 * velocity[name] + np.asarray(grads[name], dtype=np.float64)
                params[name] = params[name] - TRAIN_LEARNING_RATE * velocity[name]
    return rows


def diagnosis_comparison() -> dict[str, object]:
    """Return deterministic broken-vs-corrected evidence for the diagnosis task."""

    broken = _gradient_diagnostics(_broken_loss_and_grads_omitted_batch_mean)
    corrected = _gradient_diagnostics(reference.loss_and_grads)
    broken_curve = _diagnosis_training_curve(_broken_loss_and_grads_omitted_batch_mean)
    corrected_curve = _diagnosis_training_curve(reference.loss_and_grads)
    return {
        "defect_id": DIAGNOSIS_DEFECT_ID,
        "repair_task": DIAGNOSIS_REPAIR_TASK,
        "broken_gradient_checks": broken["gradient_checks"],
        "corrected_gradient_checks": corrected["gradient_checks"],
        "broken_max_abs_error": broken["max_abs_error"],
        "corrected_max_abs_error": corrected["max_abs_error"],
        "visible_symptom": {
            "metric": "validation loss after six deterministic mini-batch epochs",
            "broken_loss_curve": broken_curve,
            "corrected_loss_curve": corrected_curve,
            "broken_final_valid_loss": broken_curve[-1]["valid_loss"],
            "corrected_final_valid_loss": corrected_curve[-1]["valid_loss"],
        },
        "checkpoint_comparison": {
            "planted_defect": {
                "checkpoint": "backward",
                "status": "failed",
                "max_abs_error": broken["max_abs_error"],
            },
            "corrected_reference": {
                "checkpoint": "backward",
                "status": "passed",
                "max_abs_error": corrected["max_abs_error"],
            },
        },
    }


def _assert_diagnosis_report(report: Any) -> dict[str, str]:
    if not isinstance(report, dict):
        raise CheckFailure("diagnose_mlp_defect must return a diagnosis dict")
    required = ("defect_id", "diagnosis", "failed_invariant", "evidence", "repair", "regression_check")
    missing = [key for key in required if key not in report]
    if missing:
        raise CheckFailure(f"diagnosis is missing {', '.join(missing)}")
    normalized = {key: str(report[key]).strip() for key in required}
    if any(not value for value in normalized.values()):
        raise CheckFailure("diagnosis fields must be non-empty")
    if sum(len(value) for value in normalized.values()) > 1200:
        raise CheckFailure("diagnosis should be short enough to audit in a receipt")
    if normalized["defect_id"] != DIAGNOSIS_DEFECT_ID:
        raise CheckFailure(f"defect_id must be {DIAGNOSIS_DEFECT_ID!r}")
    checks = {
        "diagnosis": ("mini-batch", "dlogits"),
        "failed_invariant": ("finite", "gradient"),
        "evidence": ("directional", "finite"),
        "repair": ("divide", "batch"),
        "regression_check": ("backward", "abs_error"),
    }
    for key, terms in checks.items():
        lowered = normalized[key].lower()
        for term in terms:
            if term not in lowered:
                raise CheckFailure(f"diagnosis field {key!r} must mention {term!r}")
    return normalized


def _check_diagnose(candidate: dict[str, object]) -> dict[str, object]:
    report = _assert_diagnosis_report(candidate["diagnose_mlp_defect"]())
    comparison = diagnosis_comparison()
    candidate_diagnostics = _gradient_diagnostics(candidate["loss_and_grads"])
    evidence = {
        **comparison,
        "diagnosis": report,
        "candidate_gradient_checks": candidate_diagnostics["gradient_checks"],
        "candidate_max_abs_error": candidate_diagnostics["max_abs_error"],
    }
    if float(candidate_diagnostics["max_abs_error"]) > 1e-6:
        raise CheckFailure(
            "diagnose checkpoint rejects unresolved omitted mini-batch mean defect: "
            f"candidate max abs_error {candidate_diagnostics['max_abs_error']:.6g}",
            evidence=evidence,
        )
    return evidence


def _check_train(candidate: dict[str, object]) -> dict[str, object]:
    fn = candidate["train_epoch"]
    train_X, train_y = fixtures.load_dataset()["train"]
    valid_X, valid_y = fixtures.load_dataset()["valid"]
    params = _fixture_params()
    before = _copy_params(params)
    rng = np.random.default_rng(fixtures.TRAIN_SEED)
    output = fn(
        train_X.copy(),
        train_y.copy(),
        params,
        None,
        rng,
        batch_size=TRAIN_BATCH_SIZE,
        learning_rate=TRAIN_LEARNING_RATE,
    )
    _assert_same_params(before, params, "train_epoch")
    if not isinstance(output, tuple) or len(output) != 3:
        raise CheckFailure("train_epoch must return (params, state, metrics)")
    trained_params = _assert_param_dict(output[0])
    trained_state = _assert_optimizer_state(output[1], trained_params)
    metrics = _assert_metric_dict(output[2])
    expected_batch_count = int(np.ceil(len(train_X) / TRAIN_BATCH_SIZE))
    if trained_state["step"] != expected_batch_count:
        raise CheckFailure(
            f"optimizer step is {trained_state['step']}, expected one update per mini-batch ({expected_batch_count})",
            evidence={"hint": "Visit each shuffled mini-batch once and increment optimizer state after every batch."},
        )
    observed_losses = np.asarray(metrics.get("batch_losses", []), dtype=np.float64)
    if observed_losses.shape != (expected_batch_count,):
        raise CheckFailure(f"batch_losses has shape {observed_losses.shape}, expected ({expected_batch_count},)")
    if not np.isfinite(observed_losses).all() or (observed_losses <= 0.0).any():
        raise CheckFailure("batch_losses must be finite positive values")
    metric_batch_count = int(metrics.get("batch_count", -1))
    if metric_batch_count != expected_batch_count:
        raise CheckFailure(f"metrics batch_count is {metric_batch_count}, expected {expected_batch_count}")

    initial_train = reference.evaluate_model(train_X, train_y, before)
    final_train = reference.evaluate_model(train_X, train_y, trained_params)
    valid = reference.evaluate_model(valid_X, valid_y, trained_params)
    delta_norm = _param_delta_norm(before, trained_params)
    if not (0.01 < delta_norm < 5.0):
        raise CheckFailure(
            "one mini-batch epoch should update parameters by a bounded amount",
            evidence={"param_delta_norm": round(delta_norm, 10)},
        )
    if final_train["loss"] >= initial_train["loss"] - 0.01:
        raise CheckFailure(
            "one mini-batch epoch should reduce training loss by a visible margin",
            evidence={
                "initial_train_loss": round(float(initial_train["loss"]), 10),
                "final_train_loss": round(float(final_train["loss"]), 10),
            },
        )
    return {
        "split": "train",
        "batch_size": TRAIN_BATCH_SIZE,
        "batch_count": expected_batch_count,
        "learning_rate": TRAIN_LEARNING_RATE,
        "initial_train_loss": round(float(initial_train["loss"]), 10),
        "final_train_loss": round(float(final_train["loss"]), 10),
        "valid_loss": round(float(valid["loss"]), 10),
        "valid_accuracy": round(float(valid["accuracy"]), 10),
        "param_delta_norm": round(delta_norm, 10),
        "batch_losses": np.round(observed_losses, 10).tolist(),
    }


def _check_inference(candidate: dict[str, object]) -> dict[str, object]:
    fn = candidate["predict_proba"]
    valid_X, _ = fixtures.load_dataset()["valid"]
    params, _ = _trained_reference_params()
    before = _copy_params(params)
    small_batches = _assert_probabilities(fn(valid_X.copy(), params, batch_size=7), (len(valid_X), fixtures.CLASSES))
    _assert_same_params(before, params, "predict_proba")
    full_batch = _assert_probabilities(fn(valid_X.copy(), params, batch_size=len(valid_X) + 5), (len(valid_X), fixtures.CLASSES))
    expected = reference.predict_proba(valid_X, params, batch_size=13)
    max_batch_difference = float(np.max(np.abs(small_batches - full_batch)))
    max_reference_difference = float(np.max(np.abs(small_batches - expected)))
    max_row_sum_error = float(np.max(np.abs(small_batches.sum(axis=1) - 1.0)))
    if max_batch_difference > 1e-12:
        raise CheckFailure(
            "predict_proba must be invariant to the batch size",
            evidence={"max_batch_difference": round(max_batch_difference, 12)},
        )
    if max_reference_difference > 1e-10:
        raise CheckFailure(
            "predict_proba values differ from the reference probabilities",
            evidence={"max_reference_difference": round(max_reference_difference, 12)},
        )
    if max_row_sum_error > 1e-12:
        raise CheckFailure(
            "predict_proba rows must sum to one",
            evidence={"max_row_sum_error": round(max_row_sum_error, 12)},
        )
    return {
        "split": "valid",
        "probabilities_shape": list(small_batches.shape),
        "batch_sizes_checked": [7, len(valid_X) + 5],
        "max_batch_difference": round(max_batch_difference, 12),
        "max_reference_difference": round(max_reference_difference, 12),
        "max_row_sum_error": round(max_row_sum_error, 12),
        "confidence_preview": np.round(small_batches.max(axis=1)[:5], 6).tolist(),
    }


def _check_evaluate(candidate: dict[str, object]) -> dict[str, object]:
    fn = candidate["evaluate_model"]
    test_X, test_y = fixtures.load_dataset()["test"]
    params, _ = _trained_reference_params()
    before = _copy_params(params)
    metrics = fn(test_X.copy(), test_y.copy(), params)
    _assert_same_params(before, params, "evaluate_model")
    if not isinstance(metrics, dict):
        raise CheckFailure("evaluate_model must return a metrics dict")
    expected = reference.evaluate_model(test_X, test_y, params)
    predictions = np.asarray(metrics.get("predictions"))
    if predictions.shape != expected["predictions"].shape:
        raise CheckFailure(f"predictions have shape {predictions.shape}, expected {expected['predictions'].shape}")
    if predictions.dtype.kind not in {"i", "u"}:
        raise CheckFailure("predictions must be an integer array")
    if not np.array_equal(predictions, expected["predictions"]):
        raise CheckFailure("evaluate_model predictions differ from the reference")
    loss = float(metrics.get("loss"))
    accuracy = float(metrics.get("accuracy"))
    count = int(metrics.get("n", -1))
    if not np.isclose(loss, float(expected["loss"]), atol=1e-10, rtol=1e-10):
        raise CheckFailure(f"evaluation loss {loss:.12g} differs from reference {float(expected['loss']):.12g}")
    if not np.isclose(accuracy, float(expected["accuracy"]), atol=1e-12, rtol=0.0):
        raise CheckFailure(f"evaluation accuracy {accuracy:.12g} differs from reference {float(expected['accuracy']):.12g}")
    if count != len(test_X):
        raise CheckFailure(f"evaluation count is {count}, expected {len(test_X)}")
    return {
        "split": "test",
        "n": count,
        "loss": round(loss, 10),
        "accuracy": round(accuracy, 10),
        "parameter_mutation": False,
        "prediction_preview": predictions[:8].astype(int).tolist(),
    }


def _check_state(candidate: dict[str, object]) -> dict[str, object]:
    save_fn = candidate["save_state"]
    restore_fn = candidate["restore_state"]
    params, state = _trained_reference_params(epochs=2)
    expected_params = _copy_params(params)
    expected_state = _assert_optimizer_state(state, params)
    snapshot = save_fn(params, state)
    if not isinstance(snapshot, dict):
        raise CheckFailure("save_state must return a snapshot dict")

    params["W1"][0, 0] += 100.0
    state["velocity"]["W1"][0, 0] += 100.0
    restored = restore_fn(snapshot)
    if not isinstance(restored, tuple) or len(restored) != 2:
        raise CheckFailure("restore_state must return (params, state)")
    restored_params = _assert_param_dict(restored[0])
    restored_state = _assert_optimizer_state(restored[1], restored_params)
    for name in PARAM_SHAPES:
        if not np.array_equal(restored_params[name], expected_params[name]):
            raise CheckFailure(f"restore_state did not restore parameter block {name}")
        if not np.array_equal(restored_state["velocity"][name], expected_state["velocity"][name]):
            raise CheckFailure(f"restore_state did not restore optimizer velocity {name}")
    if restored_state["step"] != expected_state["step"]:
        raise CheckFailure(f"restore_state step is {restored_state['step']}, expected {expected_state['step']}")

    snapshot["params"]["W1"][0, 0] += 50.0
    snapshot["optimizer"]["velocity"]["W1"][0, 0] += 50.0
    snapshot_is_independent = (
        restored_params["W1"][0, 0] == expected_params["W1"][0, 0]
        and restored_state["velocity"]["W1"][0, 0] == expected_state["velocity"]["W1"][0, 0]
    )
    if not snapshot_is_independent:
        raise CheckFailure("restore_state must return arrays independent from the saved snapshot")
    return {
        "parameters_restored": True,
        "optimizer_state_restored": True,
        "snapshot_is_independent": True,
        "optimizer_step": int(restored_state["step"]),
        "velocity_norms": {
            name: round(float(np.linalg.norm(restored_state["velocity"][name])), 10)
            for name in PARAM_SHAPES
        },
    }


def _run_owned(checkpoint: str, candidate: dict[str, object]) -> dict[str, object]:
    if checkpoint == "init":
        return _check_init(candidate)
    if checkpoint == "forward":
        return _check_forward(candidate)
    if checkpoint == "backward":
        return _check_backward(candidate)
    if checkpoint == "train":
        return _check_train(candidate)
    if checkpoint == "inference":
        return _check_inference(candidate)
    if checkpoint == "evaluate":
        return _check_evaluate(candidate)
    if checkpoint == "state":
        return _check_state(candidate)
    if checkpoint == "diagnose":
        return _check_diagnose(candidate)
    raise ValueError(f"unknown checkpoint: {checkpoint}")


@contextmanager
def _patched_reference_globals(
    candidate: dict[str, object],
    reference_overrides: Iterable[str],
):
    patches: list[tuple[dict[str, object], str, object]] = []
    missing = object()
    try:
        for function_name in reference_overrides:
            reference_function = getattr(reference, function_name)
            for value in candidate.values():
                globals_dict = getattr(value, "__globals__", None)
                if not isinstance(globals_dict, dict):
                    continue
                old = globals_dict.get(function_name, missing)
                patches.append((globals_dict, function_name, old))
                globals_dict[function_name] = reference_function
        yield
    finally:
        for globals_dict, function_name, old in reversed(patches):
            if old is missing:
                globals_dict.pop(function_name, None)
            else:
                globals_dict[function_name] = old


def run_checkpoint(
    name: str,
    candidate: dict[str, object],
    reference_overrides: set[str] | None = None,
) -> dict[str, object]:
    """Grade one MLP checkpoint and return structured evidence."""

    if name not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint: {name}")
    overrides = sorted(set(reference_overrides or set()))
    effective = dict(candidate)
    checks: list[dict[str, object]] = []
    for function_name in overrides:
        if function_name not in LEARNER_FUNCTIONS:
            raise ValueError(f"unknown reference override: {function_name}")
        effective[function_name] = getattr(reference, function_name)
        checks.append({
            "name": f"substitution_{function_name}",
            "status": "passed",
            "evidence": {
                "function": function_name,
                "checkpoint": FUNCTION_CHECKPOINT[function_name],
                "independent_credit": False,
            },
        })

    owned = OWNED_FUNCTIONS[name]
    result: dict[str, object] = {
        "checkpoint": name,
        "status": "failed",
        "checks": checks,
        "evidence": {},
        "fixture_id": fixtures.FIXTURE_ID,
        "reference_overrides": overrides,
        "substitutions": [
            {
                "function": function_name,
                "checkpoint": FUNCTION_CHECKPOINT[function_name],
                "independent_credit": False,
            }
            for function_name in overrides
        ],
        "owned_checkpoint_passed": False,
    }
    try:
        missing = [function_name for function_name in owned if function_name not in effective]
        if missing:
            raise CheckFailure(f"missing learner function(s): {', '.join(missing)}")
        with _patched_reference_globals(effective, overrides):
            evidence = _run_owned(name, effective)
    except Exception as error:
        evidence = getattr(error, "evidence", {})
        result["evidence"] = deepcopy(evidence)
        result["checks"].append({
            "name": f"checkpoint_{name}",
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=6),
            "evidence": deepcopy(evidence),
        })
        return result

    result["evidence"] = evidence
    result["checks"].append({"name": f"checkpoint_{name}", "status": "passed", "evidence": evidence})
    result["status"] = "passed"
    result["owned_checkpoint_passed"] = not any(function_name in overrides for function_name in owned)
    return result


def _load_candidate(name: str) -> dict[str, object]:
    module = importlib.import_module(f"browser_gym.labs.mlp.{name}")
    return {function_name: getattr(module, function_name) for function_name in LEARNER_FUNCTIONS}


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("reference", "starter", "mutants"), default="reference")
    parser.add_argument("--checkpoint", choices=(*CHECKPOINTS, "all"), default="all")
    parser.add_argument("--json", action="store_true", help="print full JSON receipts")
    args = parser.parse_args(argv)

    candidate = _load_candidate(args.candidate)
    checkpoint_names = CHECKPOINTS if args.checkpoint == "all" else (args.checkpoint,)
    results = [run_checkpoint(checkpoint, candidate) for checkpoint in checkpoint_names]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(f"{result['checkpoint']}: {result['status']} ({result['fixture_id']})")
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
