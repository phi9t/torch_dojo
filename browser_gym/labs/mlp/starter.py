import numpy as np


def init_params(
    rng: np.random.Generator,
    input_dim: int,
    hidden_dim: int,
    classes: int,
) -> dict[str, np.ndarray]:
    raise NotImplementedError("initialize W1, b1, W2, and b2")


def forward(X: np.ndarray, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    raise NotImplementedError("compute hidden_pre, hidden, and logits")


def loss_and_grads(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[float, dict[str, np.ndarray]]:
    raise NotImplementedError("return mean loss and gradients for every parameter")


def train_epoch(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    state: dict | None,
    rng: np.random.Generator,
    batch_size: int,
    learning_rate: float,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    raise NotImplementedError("run one shuffled mini-batch training epoch")


def predict_proba(
    X: np.ndarray,
    params: dict[str, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    raise NotImplementedError("return batched class probabilities")


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
) -> dict[str, object]:
    raise NotImplementedError("return loss, accuracy, and predictions without updating")


def save_state(params: dict[str, np.ndarray], state: dict | None) -> dict[str, object]:
    raise NotImplementedError("snapshot parameters and optimizer state")


def restore_state(snapshot: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    raise NotImplementedError("restore parameters and optimizer state")


def diagnose_mlp_defect() -> dict[str, str]:
    raise NotImplementedError("diagnose and repair the planted omitted-batch-mean gradient defect")
