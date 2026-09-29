import pandas as pd

from ber.search import expand_space, search_blocking, successive_halving


def test_expand_space_cartesian_product():
    space = {"a": [1, 2], "b": ["x"]}
    got = expand_space(space)
    assert {tuple(sorted(d.items())) for d in got} == {
        (("a", 1), ("b", "x")),
        (("a", 2), ("b", "x")),
    }


def test_successive_halving_keeps_best():
    configs = [{"id": i} for i in range(8)]
    calls = []

    def evaluate(c):
        calls.append(c["id"])
        return float(c["id"])

    ranked = successive_halving(configs, evaluate, keep_frac=0.5, min_keep=1, max_rounds=3)
    assert ranked[0]["id"] == 7
    assert len(ranked) == 1
    assert len(calls) == 8 + 4 + 2


def test_successive_halving_records_scores():
    configs = [{"id": 0}, {"id": 1}]
    ranked = successive_halving(configs, lambda c: c["id"] * 0.5, keep_frac=0.5, min_keep=1, max_rounds=2)
    assert ranked[0]["score"] == 0.5


def test_search_blocking_reports_true_recall_not_intersection():
    s1 = pd.DataFrame({
        "entity_id": ["S1-a", "S1-b"],
        "business_name": ["Zephyr Trading", "Quixotic Industries"],
        "business_address": ["", ""],
        "country": ["India", "India"],
    })
    mid = pd.DataFrame({
        "entity_id": ["S2-a", "S2-b"],
        "business_name": ["Zephyr Trading Company", "Gamma Delta Foods"],
        "business_address": ["", ""],
        "country": ["India", "India"],
    })
    labels = pd.DataFrame({
        "source1_entity_id": ["S1-a", "S1-b"],
        "matched_entity_ids": ["S2-a", "S2-b"],
    })
    res = search_blocking(s1, mid, labels, [{"max_postings": 1000, "max_candidates": 40}])
    assert res[0]["recall_ceiling"] == 0.5
    assert res[0]["positives"] == 1
    assert res[0]["true_positives"] == 2
