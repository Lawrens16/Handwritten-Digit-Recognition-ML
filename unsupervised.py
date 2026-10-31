"""Cluster pixel vectors without accepting or loading target labels."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringResult:
    centers: np.ndarray
    assignments: np.ndarray
    counts: np.ndarray
    silhouette: float
    seconds: float


def train_clusters(features):
    start = perf_counter()
    model = KMeans(n_clusters=10, n_init=10, random_state=42)
    assignments = model.fit_predict(features)
    return ClusteringResult(
        model.cluster_centers_.reshape(10, 8, 8), assignments,
        np.bincount(assignments, minlength=10),
        float(silhouette_score(features, assignments)), perf_counter() - start)
