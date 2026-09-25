import numpy as np
import pandas as pd
from ber.features import build_pair_frame, classical_features, structural_features


def test_classical_features_identical_strings_are_max():
    frame = pd.DataFrame({
        "n1": ["acme trading"], "n2": ["acme trading"],
        "a1": ["12 mg road"], "a2": ["12 mg road"],
        "c1": ["india"], "c2": ["india"],
        "is_s2": [True], "s1_idx": [0], "mid_idx": [0], "pass": ["rare_token"],
    })
    X = classical_features(frame)
    assert X.shape == (1, 15)
    assert X[0, 0] == 100.0
    assert X[0, 5] == 1.0


def test_build_pair_frame_joins_columns():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1"], "business_name": ["Acme Pvt Ltd"],
        "business_address": ["12 MG Rd"], "country": ["India"],
    })
    mid = pd.DataFrame({
        "entity_id": ["S2-9"], "business_name": ["Acme Private Limited"],
        "business_address": ["12 MG Road"], "country": ["India"],
    })
    pairs = pd.DataFrame({
        "s1_idx": np.array([0], dtype="int32"),
        "mid_idx": np.array([0], dtype="int32"),
        "pass": ["rare_token"],
    })
    frame = build_pair_frame(s1, mid, pairs)
    assert frame.loc[0, "n1"] == "acme private limited"
    assert frame.loc[0, "n2"] == "acme private limited"
    assert set(["s1_idx", "mid_idx", "pass", "is_s2"]) <= set(frame.columns)


def test_structural_features_postal_and_phonetic():
    frame = pd.DataFrame({
        "n1": ["acme"], "n2": ["acme"],
        "n1_fold": ["acme"], "n2_fold": ["acme"],
        "a1": ["12 mg road bengaluru 560001"], "a2": ["12 mg road bengaluru 560001"],
        "c1": ["india"], "c2": ["india"],
        "is_s2": [True],
    })
    S = structural_features(frame)
    assert S.shape[0] == 1 and S[0, 0] == 1.0
