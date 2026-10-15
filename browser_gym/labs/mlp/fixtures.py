"""Deterministic fixtures for the NumPy MLP browser lab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FIXTURE_ID = "mlp2d-complete-v1"
DATASET_SEED = 62006
INIT_SEED = 62007
GRADIENT_SEED = 62008
TRAIN_SEED = 62009
INPUT_DIM = 2
HIDDEN_DIM = 5
CLASSES = 3
TRAIN_BATCH_SIZE = 16
TRAIN_LEARNING_RATE = 0.08


@dataclass(frozen=True)
class BatchFixture:
    X: np.ndarray
    y: np.ndarray


def _make_split(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=count)
    radii = 1.05 + 0.32 * np.sin(3.0 * angles) + rng.normal(0.0, 0.08, size=count)
    X = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    X += rng.normal(0.0, 0.07, size=X.shape)
    twisted = angles + 0.55 * np.sin(2.0 * radii) + 0.18 * X[:, 0] * X[:, 1]
    y = np.floor(((twisted + np.pi) / (2.0 * np.pi)) * CLASSES).astype(np.int64) % CLASSES
    return X.astype(np.float64), y


def load_dataset() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return fixed train, validation, and test splits for a small 2D problem."""

    train = _make_split(DATASET_SEED, 96)
    valid = _make_split(DATASET_SEED + 1, 48)
    test = _make_split(DATASET_SEED + 2, 48)
    train_X = train[0].copy()
    train_y = train[1].copy()
    train_X[0] = np.array([-0.219205, 0.916334], dtype=np.float64)
    train_y[0] = np.int64(1)
    return {
        "train": (train_X, train_y),
        "valid": (valid[0].copy(), valid[1].copy()),
        "test": (test[0].copy(), test[1].copy()),
    }


def checkpoint_batch() -> BatchFixture:
    """Return the small batch used by forward and backward checkpoint checks."""

    X, y = load_dataset()["train"]
    return BatchFixture(X=X[:8].copy(), y=y[:8].copy())


def seeded_rng() -> np.random.Generator:
    return np.random.default_rng(INIT_SEED)
