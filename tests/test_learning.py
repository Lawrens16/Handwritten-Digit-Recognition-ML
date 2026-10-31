import importlib.util
import numpy as np
from sklearn.datasets import load_digits


def test_supervised_split_and_predictions():
    assert importlib.util.find_spec('supervised'), 'supervised training is missing'
    from supervised import train_classifier
    result = train_classifier()
    assert len(result.train_indices) == 1437
    assert len(result.test_indices) == 360
    assert not set(result.train_indices) & set(result.test_indices)
    assert result.confusion.sum() == 360
    assert result.accuracy >= .90
    digits = load_digits()
    np.testing.assert_array_equal(result.test_labels, digits.target[result.test_indices])
    np.testing.assert_allclose(result.test_features, digits.data[result.test_indices] / 16)
    np.testing.assert_array_equal(result.model.predict(result.test_features), result.predictions)
    # Fit independently on only the specified training rows to catch leakage.
    from sklearn.linear_model import LogisticRegression
    reference = LogisticRegression(max_iter=2000).fit(
        digits.data[result.train_indices] / 16, digits.target[result.train_indices])
    np.testing.assert_allclose(result.model.coef_, reference.coef_)


def test_clustering_uses_only_features():
    assert importlib.util.find_spec('unsupervised'), 'unlabeled clustering is missing'
    from unsupervised import train_clusters
    features = load_digits().data / 16
    result = train_clusters(features)
    assert result.centers.shape == (10, 8, 8)
    assert result.counts.sum() == 1797
    np.testing.assert_array_equal(result.counts, np.bincount(result.assignments, minlength=10))
    assert -1 <= result.silhouette <= 1
    # Real membership must agree with nearest learned center.
    nearest = ((features[:, None] - result.centers.reshape(10, 64)) ** 2).sum(axis=2).argmin(axis=1)
    np.testing.assert_array_equal(result.assignments, nearest)


def test_q_learning_and_frozen_evaluation():
    assert importlib.util.find_spec('reinforcement'), 'Q-learning is missing'
    from reinforcement import train_agent, evaluate_policy, rollout
    result = train_agent()
    assert result.q_table.shape == (16, 4)
    assert np.any(result.q_table > 0)
    assert len(result.rewards) == 10000
    before = result.q_table.copy()
    assert evaluate_policy(result.q_table) == 1.0
    path, status = rollout(result.q_table)
    assert path[0] == 0 and path[-1] == 15 and status == 'goal'
    assert len(path) <= 101
    np.testing.assert_array_equal(before, result.q_table)
    path, status = rollout(np.zeros((16, 4)))
    assert status == 'loop'


def test_terminal_update_ignores_future_values():
    assert importlib.util.find_spec('reinforcement'), 'Q update is missing'
    from reinforcement import update_q
    table = np.ones((16, 4)) * 10
    update_q(table, 0, 1, 1, 5, True)
    assert table[0, 1] == 9.1
    update_q(table, 1, 1, 0, 5, False)
    assert table[1, 1] == 9.95
