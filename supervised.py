"""Held-out digit classification; normalization is fixed, never fitted."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass
class ClassificationResult:
    model: LogisticRegression
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    predictions: np.ndarray
    accuracy: float
    confusion: np.ndarray
    seconds: float


def train_classifier():
    start = perf_counter()
    digits = load_digits()
    features = digits.data / 16.0
    train_indices, test_indices = train_test_split(
        np.arange(len(features)), test_size=.2, stratify=digits.target, random_state=42)
    model = LogisticRegression(max_iter=2000)
    model.fit(features[train_indices], digits.target[train_indices])
    test_features = features[test_indices]
    test_labels = digits.target[test_indices]
    predictions = model.predict(test_features)
    return ClassificationResult(
        model, train_indices, test_indices, test_features, test_labels, predictions,
        float(accuracy_score(test_labels, predictions)),
        confusion_matrix(test_labels, predictions, labels=np.arange(10)),
        perf_counter() - start)
