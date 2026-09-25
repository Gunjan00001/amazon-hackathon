import numpy as np
import pandas as pd
from ber.decision import matches_from_scores, one_to_one_assign, prune_top_k, to_entity_ids


def _df():
    return pd.DataFrame({
        "s1_idx": np.array([0, 0, 0, 1, 1], dtype="int32"),
        "mid_idx": np.array([10, 11, 12, 10, 20], dtype="int32"),
        "score": np.array([0.9, 0.4, 0.8, 0.85, 0.7]),
    })


def test_prune_top_k_per_s1():
    out = prune_top_k(_df(), k=2)
    assert len(out) == 4
    assert set(out.loc[out.s1_idx == 0, "mid_idx"]) == {10, 12}


def test_matches_from_scores_threshold():
    got = matches_from_scores(_df(), 0.75)
    assert got == {0: [10, 12], 1: [10]}


def test_one_to_one_assign_resolves_conflict():
    got = one_to_one_assign(_df(), 0.5)
    assert got[0] == [10, 12]
    assert got[1] == [20]


def test_to_entity_ids_maps_indexes():
    got = to_entity_ids({0: [10]}, ["S1-1", "S1-2"], {10: "S2-7", 11: "S3-3"})
    assert got == {"S1-1": ["S2-7"], "S1-2": []}
