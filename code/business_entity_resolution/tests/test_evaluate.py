import numpy as np

from ber.evaluate import entity_macro_counts, entity_macro_f05, evaluate_solution, grouped_split, holdout_mask


def _case():
    groups = np.array([0, 0, 1, 1, 2])
    labels = np.array([1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.2, 0.1, 0.8, 0.1])
    countries = np.array(["us", "us", "india", "india", "india"])
    return groups, labels, scores, countries


def test_evaluate_solution_returns_threshold_and_holdout():
    groups, labels, scores, countries = _case()
    out = evaluate_solution(groups, labels, scores, countries, holdout_country="us", grid=np.array([0.5]))
    assert out["threshold"] == 0.5
    assert abs(out["macro_f05"] - 1.0) < 1e-9
    assert abs(out["macro_f05_holdout"] - 1.0) < 1e-9
    assert out["holdout_country"] == "us"


def test_evaluate_solution_without_country_has_no_holdout_key():
    groups, labels, scores, _ = _case()
    out = evaluate_solution(groups, labels, scores, grid=np.array([0.5]))
    assert "macro_f05_holdout" not in out


def test_grouped_split_disjoint_by_group():
    groups = np.repeat(np.arange(100), 3)
    tr, va = grouped_split(groups, val_frac=0.2, seed=0)
    assert not (set(groups[tr]) & set(groups[va]))
    assert len(set(groups[va])) == 20


def test_holdout_mask_selects_country():
    m = holdout_mask(np.array(["us", "india", "us", "france"]), "us")
    assert m.tolist() == [True, False, True, False]


def test_entity_macro_f05_includes_missing_and_singletons():
    true = {0: {1, 2}, 1: set(), 2: {5}}
    pred = {0: {1}, 2: set()}
    expected = (0.8333333333333334 + 1.0 + 0.0) / 3
    assert abs(entity_macro_f05(true, pred, universe=[0, 1, 2]) - expected) < 1e-9
    assert abs(entity_macro_f05(true, pred) - expected) < 1e-9


def test_entity_macro_counts_matches_dict_version():
    n_true = np.array([2, 0, 1])
    n_pred = np.array([1, 0, 0])
    n_hit = np.array([1, 0, 0])
    expected = (0.8333333333333334 + 1.0 + 0.0) / 3
    assert abs(entity_macro_counts(n_true, n_pred, n_hit) - expected) < 1e-9
