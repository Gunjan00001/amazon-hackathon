import numpy as np
import pandas as pd

from ber.pairs import build_training_pairs, grouped_split

CFG = type("C", (), {"seed": 42, "neg_ratio": 2})()


def _cands():
    return pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-1"],
            "cand_id": ["S2-1", "S2-2", "S2-3"],
            "is_s2": [True, True, True],
            "pass_id": [1, 3, 1],
            "block_score": [1.0, 0.5, 1.0],
        }
    )


def _gt():
    return pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1"]})


def test_build_training_pairs_labels_and_ratio():
    out = build_training_pairs(_cands(), _gt(), CFG)
    assert out.loc[out["cand_id"] == "S2-1", "label"].iat[0] == 1
    assert out.loc[out["cand_id"] != "S2-1", "label"].max() == 0
    assert len(out) >= 3


def test_grouped_split_has_no_group_leakage():
    pairs = pd.DataFrame(
        {
            "s1_id": [f"S1-{i}" for i in range(10)] * 2,
            "cand_id": [f"S2-{i}" for i in range(20)],
            "label": [0] * 20,
        }
    )
    train_mask, val_mask = grouped_split(pairs, 0.5, 42)
    assert not (set(pairs.loc[train_mask, "s1_id"]) & set(pairs.loc[val_mask, "s1_id"]))
    assert np.isclose(val_mask.mean(), 0.5, atol=0.11)
