"""Metrics and threshold selection helpers."""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, accuracy_score, balanced_accuracy_score


def metric_row(y_true, probabilities, threshold):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def select_threshold_for_f2(y_true, probabilities):
    # Validation-only threshold tuning; test set is never used for this choice.
    thresholds = np.arange(0.20, 0.801, 0.01)
    scores = [fbeta_score(y_true, np.asarray(probabilities) >= t, beta=2, zero_division=0) for t in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])
