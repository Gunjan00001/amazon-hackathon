import pandas as pd
from ber.blocking import blocking_recall, generate_candidates


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


def test_recall_helper_counts_hits():
    labels = {(0, 0), (1, 1)}
    pairs = pd.DataFrame({"s1_idx": [0, 1], "mid_idx": [0, 2]})
    assert blocking_recall(pairs, labels) == 0.5
