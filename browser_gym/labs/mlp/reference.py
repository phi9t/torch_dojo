"""Reference implementation for the NumPy MLP modeling lab."""

import numpy as np


MOMENTUM = 0.9
EPSILON = 1e-12


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value, dtype=np.float64).copy() for name, value in params.items()}


def _optimizer_state(
    params: dict[str, np.ndarray],
    state: dict | None = None,
) -> dict[str, object]:
    raw_velocity = {}
    raw_step = 0
    if isinstance(state, dict):
        raw_velocity = state.get("velocity", {}) if isinstance(state.get("velocity", {}), dict) else {}
        raw_step = int(state.get("step", 0))
    velocity = {
        name: np.asarray(raw_velocity.get(name, np.zeros_like(value)), dtype=np.float64).copy()
        for name, value in params.items()
    }
    return {"step": raw_step, "velocity": velocity}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def init_params(
    rng: np.random.Generator,
    input_dim: int,
    hidden_dim: int,
    classes: int,
) -> dict[str, np.ndarray]:
    """Initialize a two-layer ReLU MLP with deterministic RNG consumption."""

    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(np.float64),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "W2": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, classes)).astype(np.float64),
        "b2": np.zeros(classes, dtype=np.float64),
    }


def forward(X: np.ndarray, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return hidden activations and logits for a two-layer ReLU MLP."""

    X = np.asarray(X, dtype=np.float64)
    hidden_pre = X @ params["W1"] + params["b1"]
    hidden = np.maximum(hidden_pre, 0.0)
    logits = hidden @ params["W2"] + params["b2"]
    return {"hidden_pre": hidden_pre, "hidden": hidden, "logits": logits}


def loss_and_grads(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[float, dict[str, np.ndarray]]:
    """Return mean softmax cross-entropy and gradients for every parameter."""

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    activations = forward(X, params)
    logits = activations["logits"]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    row = np.arange(len(X))
    loss = float(np.mean(np.log(exp.sum(axis=1)) - shifted[row, y]))

    dlogits = probabilities.copy()
    dlogits[row, y] -= 1.0
    dlogits /= len(X)
    dW2 = activations["hidden"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dhidden = dlogits @ params["W2"].T
    dhidden_pre = dhidden * (activations["hidden_pre"] > 0.0)
    dW1 = X.T @ dhidden_pre
    db1 = dhidden_pre.sum(axis=0)

    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train_epoch(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    state: dict | None,
    rng: np.random.Generator,
    batch_size: int,
    learning_rate: float,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    """Run one deterministic shuffled mini-batch SGD epoch with momentum."""

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of examples")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    next_params = _copy_params(params)
    next_state = _optimizer_state(next_params, state)
    velocity = next_state["velocity"]
    order = rng.permutation(len(X))
    losses: list[float] = []

    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        loss, grads = loss_and_grads(X[indices], y[indices], next_params)
        losses.append(float(loss))
        for name in next_params:
            velocity[name] = MOMENTUM * velocity[name] + np.asarray(grads[name], dtype=np.float64)
            next_params[name] = next_params[name] - learning_rate * velocity[name]
        next_state["step"] = int(next_state["step"]) + 1

    return next_params, next_state, {
        "batch_losses": np.asarray(losses, dtype=np.float64),
        "mean_loss": float(np.mean(losses)),
        "batch_count": int(len(losses)),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
    }


def predict_proba(
    X: np.ndarray,
    params: dict[str, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    """Return class probabilities, computing logits in bounded batches."""

    X = np.asarray(X, dtype=np.float64)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    outputs = []
    for start in range(0, len(X), batch_size):
        logits = forward(X[start:start + batch_size], params)["logits"]
        outputs.append(_softmax(logits))
    if not outputs:
        return np.empty((0, params["b2"].shape[0]), dtype=np.float64)
    return np.vstack(outputs)


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
) -> dict[str, object]:
    """Return loss, accuracy, and predictions without updating model state."""

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    probabilities = predict_proba(X, params, batch_size=32)
    row = np.arange(len(X))
    loss = float(np.mean(-np.log(np.clip(probabilities[row, y], EPSILON, 1.0))))
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    accuracy = float(np.mean(predictions == y))
    return {
        "loss": loss,
        "accuracy": accuracy,
        "n": int(len(X)),
        "predictions": predictions,
    }


def save_state(params: dict[str, np.ndarray], state: dict | None) -> dict[str, object]:
    """Return a deep snapshot of parameters and optimizer state."""

    params_copy = _copy_params(params)
    return {
        "params": params_copy,
        "optimizer": _optimizer_state(params_copy, state),
    }


def restore_state(snapshot: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Restore copied parameters and optimizer state from `save_state` output."""

    params = _copy_params(snapshot["params"])
    state = _optimizer_state(params, snapshot.get("optimizer", {}))
    return params, state


def diagnose_mlp_defect() -> dict[str, str]:
    """Return the short diagnosis for the planted backward-pass defect."""

    return {
        "defect_id": "omitted_batch_mean",
        "diagnosis": "The planted backward pass omits the mini-batch mean scale for dlogits.",
        "failed_invariant": "Directional finite differences must match every parameter-block gradient.",
        "evidence": (
            "The broken analytic directional gradients are much larger than the finite-difference "
            "directions on the fixed batch."
        ),
        "repair": "Divide dlogits by the batch size before computing W1, b1, W2, and b2 gradients.",
        "regression_check": "Run the backward checkpoint and require abs_error below 1e-6 for every block.",
    }
