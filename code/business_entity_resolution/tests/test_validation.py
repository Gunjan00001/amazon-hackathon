import numpy as np

from ber.validation import _macro_from_counts


def test_macro_from_counts_rules():
    ntrue = np.array([2, 0, 1, 0])
    npred = np.array([3, 0, 0, 1])
    tp = np.array([2, 0, 0, 0])
    scores = _macro_from_counts(ntrue, npred, tp)
    assert round(scores[0], 3) == 0.714
    assert scores[1] == 1.0
    assert scores[2] == 0.0
    assert scores[3] == 0.0
    assert round(float(scores.mean()), 6) == round((0.7142857142857142 + 1 + 0 + 0) / 4, 6)
