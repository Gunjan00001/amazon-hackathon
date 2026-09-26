import numpy as np

from ber.calibration import (
    apply_calibration,
    load_calibration,
    save_calibration,
    singleton_decision,
    tune_per_country,
    tune_singleton_tau,
)
from ber.threshold import sweep_threshold


def _country_data():
    probs, groups, labels, countries = [], [], [], []
    for i in range(2):
        groups += [f"US-{i}"] * 2
        labels += [1, 0]
        probs += [0.9 - 0.1 * i, 0.7]
        countries += ["us", "us"]
    for i in range(2):
        groups += [f"IN-{i}"] * 2
        labels += [1, 0]
        probs += [0.5 - 0.2 * i, 0.1]
        countries += ["india", "india"]
    return np.array(probs), np.array(groups), np.array(labels), np.array(countries)


def test_tune_per_country_recovers_distinct_thresholds():
    probs, groups, labels, countries = _country_data()
    result = tune_per_country(probs, groups, labels, countries, global_default=0.5, min_entities=2)
    us_mask = countries == "us"
    india_mask = countries == "india"
    _, us_expected = sweep_threshold(probs[us_mask], groups[us_mask], labels[us_mask])
    _, india_expected = sweep_threshold(probs[india_mask], groups[india_mask], labels[india_mask])
    assert result["us"] == us_expected
    assert result["india"] == india_expected
    assert result["us"] > result["india"]


def test_tune_per_country_single_country_matches_global_path():
    probs, groups, labels, _ = _country_data()
    countries = np.array(["us"] * len(probs))
    result = tune_per_country(probs, groups, labels, countries, global_default=None, min_entities=2)
    _, expected = sweep_threshold(probs, groups, labels)
    assert result["us"] == expected


def test_tune_per_country_falls_back_for_sparse_country():
    probs, groups, labels, countries = _country_data()
    probs = np.append(probs, [0.6, 0.4])
    groups = np.append(groups, ["FR-0", "FR-0"])
    labels = np.append(labels, [1, 0])
    countries = np.append(countries, ["france", "france"])
    result = tune_per_country(probs, groups, labels, countries, global_default=0.7, min_entities=2)
    assert result["france"] == 0.7


def test_singleton_decision_suppresses_low_max_groups():
    probs = np.array([0.4, 0.2, 0.05, 0.03])
    groups = np.array(["A", "A", "B", "B"])
    keep = singleton_decision(probs, groups, tau=0.3)
    assert keep.tolist() == [True, True, False, False]


def test_tune_singleton_tau_suppresses_false_merges():
    probs = np.array([0.9, 0.35, 0.30])
    groups = np.array(["E1", "E2", "E2"])
    labels = np.array([1, 0, 0])
    thresholds = np.array([0.5, 0.2, 0.2])
    score, tau = tune_singleton_tau(probs, groups, labels, thresholds, low=0.3, high=0.99, step=0.01)
    assert tau > 0.35
    assert score == 1.0


def test_apply_calibration_without_tau_is_plain_threshold():
    probs = np.array([0.9, 0.4, 0.2])
    groups = np.array(["A", "B", "B"])
    thresholds = np.array([0.5, 0.5, 0.5])
    mask = apply_calibration(probs, groups, thresholds)
    assert mask.tolist() == [True, False, False]


def test_calibration_roundtrip_and_backward_compat(tmp_path):
    path = tmp_path / "threshold.json"
    save_calibration(
        path,
        global_threshold=0.8,
        by_country={"us": 0.85, "india": 0.6},
        singleton_tau=0.7,
        use_one_to_one=True,
    )
    loaded = load_calibration(path)
    assert loaded["global"] == 0.8
    assert loaded["by_country"] == {"us": 0.85, "india": 0.6}
    assert loaded["singleton_tau"] == 0.7
    assert loaded["use_one_to_one"] is True

    path.write_text('{"global": 0.7, "use_one_to_one": false}', encoding="utf-8")
    old = load_calibration(path)
    assert old["global"] == 0.7
    assert old["by_country"] == {}
    assert old["singleton_tau"] is None
    assert old["use_one_to_one"] is False
