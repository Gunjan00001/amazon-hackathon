import pandas as pd

from ber.audit import audit_candidates


def test_audit_candidates_recall():
    cands = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-2"],
            "cand_id": ["S2-1", "S3-9", "S2-7"],
            "pass_id": [1, 3, 1],
            "block_score": [1.0, 0.6, 1.0],
            "is_s2": [True, False, True],
        }
    )
    gt = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1,S3-9", "S2-7", ""],
        }
    )
    meta = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"], "country": ["US", "India", "US"]})
    rep = audit_candidates(cands, gt, meta)
    assert rep["recall"] == 1.0
    assert rep["recall_by_country"]["US"] == 1.0
    assert rep["singleton_candidates_mean"] == 0.0
