"""Temperature calibration and validation-selected abstention."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss

from intentlab.common import wilson


def scale_probabilities(probabilities, temperature):
    return softmax(np.log(np.clip(probabilities, 1e-12, 1)) / temperature, axis=1)


def fit_temperature(y, probabilities, classes):
    result = minimize_scalar(
        lambda t: log_loss(y, scale_probabilities(probabilities, t), labels=classes),
        bounds=(0.2, 5),
        method="bounded",
    )
    return float(result.x)


def routing_metrics(y, probabilities, classes, threshold):
    predicted = classes[np.argmax(probabilities, axis=1)]
    accepted = probabilities.max(axis=1) >= threshold
    count = int(accepted.sum())
    correct = int((predicted[accepted] == np.asarray(y)[accepted]).sum())
    return {
        "threshold": float(threshold),
        "accepted": count,
        "total": len(y),
        "coverage": float(accepted.mean()),
        "accuracy": correct / count if count else None,
        "accuracy_wilson_95": wilson(correct, count) if count else None,
    }


def select_threshold(y, probabilities, classes, target=0.95):
    rows = [routing_metrics(y, probabilities, classes, t) for t in np.linspace(0, 1, 101)]
    eligible = [r for r in rows if r["coverage"] >= 0.1 and r["accuracy_wilson_95"][0] >= target]
    # If evidence is insufficient, disable automatic routing.
    threshold = min(eligible, key=lambda r: r["threshold"])["threshold"] if eligible else 1.01
    return threshold, rows


def scores(y, probabilities, classes):
    predicted = classes[np.argmax(probabilities, axis=1)]
    correct = predicted == np.asarray(y)
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "log_loss": float(log_loss(y, probabilities, labels=classes)),
        "ece_10_bins": float(ece),
        "accuracy_wilson_95": wilson(int(correct.sum()), len(y)),
    }


def add_typo(text):
    """One deterministic transposition in the first sufficiently long word."""
    words = text.split()
    for i, word in enumerate(words):
        if len(word) >= 5:
            words[i] = word[0] + word[2] + word[1] + word[3:]
            break
    return " ".join(words)
