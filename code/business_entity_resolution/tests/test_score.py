import numpy as np
from ber.score import best_threshold, entity_f05, macro_f05


def test_entity_f05_worked_example():
    labels = np.array([1, 1, 0])
    preds = np.array([1, 1, 1])
    assert abs(entity_f05(labels, preds) - 0.714285) < 1e-5


def test_entity_f05_singleton_rule():
    empty = np.array([], dtype=int)
    assert entity_f05(empty, np.array([], dtype=int)) == 1.0
    assert entity_f05(empty, np.array([1], dtype=int)) == 0.0


def test_macro_f05_groups_and_thresholds():
    groups = np.array([0, 0, 1, 1, 2])
    labels = np.array([1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.2, 0.1, 0.8, 0.1])
    assert abs(macro_f05(groups, labels, scores, 0.5) - 1.0) < 1e-9
    f, th = best_threshold(groups, labels, scores, grid=np.array([0.5]))
    assert (f, th) == (1.0, 0.5)
