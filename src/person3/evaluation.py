"""Evaluation helpers for trajectory classification."""

from collections import Counter
from typing import Dict, Iterable, List, Optional


def class_distribution(
    labels: Iterable[str],
) -> Dict[str, int]:
    return dict(Counter(labels))


def classification_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Optional[float]]:
    """Calculate standard classification metrics."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if not y_true:
        return {
            "accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
        }

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }
