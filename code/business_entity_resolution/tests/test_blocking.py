import pandas as pd
from ber.blocking import (blocking_recall, build_index, generate_candidates,
                          generate_candidates_from_keys, load_keys, prepare_keys, save_keys, score_keys)


def _s1():
    return pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["Zenith Trading Company", "Orchid Diagnostics"],
        "business_address": ["12 MG Road Bengaluru 560001", "44 Park Ave Austin 78701"],
        "country": ["India", "US"],
    })


def _mid():
    return pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S3-3"],
        "business_name": ["Zenith Trading Pvt Ltd", "Orchid Diagnostics LLC", "Zebra Foods"],
        "business_address": ["12 MG Rd Bengaluru 560001", "44 Park Avenue Austin 78701", "9 Beach Rd Goa 403001"],
        "country": ["India", "US", "India"],
    })


def test_generates_expected_pairs_and_country_gate():
    pairs = generate_candidates(_s1(), _mid(), max_candidates_per_s1=50, max_postings=100)
    got = set(zip(pairs["s1_idx"], pairs["mid_idx"]))
    assert (0, 0) in got and (1, 1) in got
    assert (0, 2) not in got
    assert pairs["pass"].notna().all()


def test_prepare_keys_reuse_matches_dataframe_path():
    s1, mid = _s1(), _mid()
    a = generate_candidates(s1, mid, max_candidates_per_s1=50, max_postings=100)
    s1_keys = prepare_keys(s1)
    mid_keys = prepare_keys(mid)
    assert len(s1_keys) == len(s1) and len(mid_keys) == len(mid)
    b = generate_candidates_from_keys(s1_keys, mid_keys, max_candidates_per_s1=50, max_postings=100)
    pd.testing.assert_frame_equal(a, b)


def test_prebuilt_index_matches_generate_candidates():
    s1, mid = _s1(), _mid()
    a = generate_candidates(s1, mid, max_candidates_per_s1=50, max_postings=100)
    index = build_index(prepare_keys(mid), max_postings=100)
    b = score_keys(prepare_keys(s1), index, max_candidates_per_s1=50)
    pd.testing.assert_frame_equal(a, b)


def test_keys_roundtrip_through_parquet(tmp_path):
    keys = prepare_keys(_mid())
    path = tmp_path / "mid_keys.parquet"
    save_keys(keys, path)
    assert load_keys(path) == keys


def test_recall_helper_counts_hits():
    labels = {(0, 0), (1, 1)}
    pairs = pd.DataFrame({"s1_idx": [0, 1], "mid_idx": [0, 2]})
    assert blocking_recall(pairs, labels) == 0.5


def test_cap_prefers_rare_shared_token():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Zephyr Trading"],
        "business_address": [""],
        "country": ["India"],
    })
    names = ["Orchid Trading", "Pine Trading"] + [f"Maple{i} Trading" for i in range(18)] + ["Foxtrot Zephyr"]
    mid = pd.DataFrame({
        "entity_id": [f"S2-{i}" for i in range(len(names))],
        "business_name": names,
        "business_address": [""] * len(names),
        "country": ["India"] * len(names),
    })
    true_idx = len(names) - 1
    pairs = generate_candidates(s1, mid, max_candidates_per_s1=1, max_postings=1000)
    assert true_idx in set(pairs["mid_idx"][pairs["s1_idx"] == 0])
