"""sklearn model factory and utilities for baseline experiments.

Provides:
- create_model(name, random_state) — returns unfitted sklearn estimator
- predict_proba_full(model, X, num_classes) — maps sklearn's subset of classes
  to full [n, num_classes] probability array for GNN-compatible metrics.
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier


def create_model(name: str, random_state: int = 42):
    """Create an unfitted sklearn classifier for baseline experiments.

    Args:
        name: One of 'rf', 'gbdt', 'mlp'.
        random_state: Seed for reproducibility.

    Returns:
        An unfitted sklearn estimator with a .fit(), .predict(), .predict_proba()
        interface.
    """
    name = name.lower()

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )

    if name == "gbdt":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            max_depth=None,
            min_samples_leaf=20,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        )

    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate="adaptive",
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=500,
            random_state=random_state,
        )

    raise ValueError(f"Unknown model name: {name!r}. Choose from: rf, gbdt, mlp.")


def predict_proba_full(model, X: np.ndarray, num_classes: int = 44) -> np.ndarray:
    """Return a [n, num_classes] probability array.

    sklearn's predict_proba only includes classes seen during training.
    This function scatters those probabilities into a full-size zero array
    so that metrics expecting all 44 classes (e.g. confusion matrix) work
    correctly.
    """
    proba = model.predict_proba(X)  # [n, n_seen]
    full = np.zeros((X.shape[0], num_classes))
    for i, cls in enumerate(model.classes_):
        full[:, int(cls)] = proba[:, i]
    return full