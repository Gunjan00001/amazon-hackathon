import numpy as np
import pandas as pd
import pytest

from ber.evaluate import macro_f05
from ber.postprocess import one_to_one
from ber.threshold import entity_f05, sweep_threshold


def test_macro_f05_worked_example():
    truth = {"S1-1": {"S2-1", "S3-1"}}
    pred = {"S1-1": {"S2-1", "S2-2", "S3-1"}}
    assert round(macro_f05(truth, pred), 3) == 0.714


def test_macro_f05_singleton_rules():
    assert macro_f05({"S1-1": set()}, {"S1-1": set()}) == 1.0
    assert macro_f05({"S1-1": set()}, {"S1-1": {"S2-1"}}) == 0.0


def test_entity_f05_singleton_rule():
    groups = ["S1-1", "S1-1", "S1-2"]
    labels = [1, 0, 0]
    probs = [0.9, 0.1, 0.1]
    assert entity_f05(groups, labels, probs, thresholds=0.5) == 1.0
    assert entity_f05(groups, labels, [0.9, 0.1, 0.9], thresholds=0.5) == 0.5


def test_sweep_threshold_matches_reference():
    rng = np.random.default_rng(0)
    n = 800
    groups = np.array([f"S1-{v}" for v in rng.integers(0, 120, n)])
    labels = rng.integers(0, 2, n)
    probs = rng.random(n)

    score, threshold = sweep_threshold(probs, groups, labels)
    reference = max(
        (entity_f05(groups, labels, probs, thresholds=t), float(t))
        for t in np.arange(0.05, 0.95 + 1e-9, 0.025)
    )
    assert score == pytest.approx(reference[0], abs=1e-9)
    assert threshold == pytest.approx(reference[1], abs=1e-9)


def test_one_to_one_keeps_best_assignment():
    pairs = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-2"],
            "cand_id": ["S2-1", "S2-1"],
            "prob": [0.9, 0.6],
        }
    )
    keep = one_to_one(pairs["prob"], pairs)
    assert keep.tolist() == [True, False]
