import json

import pandas as pd
import pytest

from ber.audit import audit_candidates, run_audit


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


def _seed(cfg):
    data = cfg.data_dir
    (data / "processed").mkdir(parents=True, exist_ok=True)
    (data / "candidates").mkdir(parents=True, exist_ok=True)
    dataset = cfg.dataset_dir / "train"
    dataset.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"]}).to_parquet(
        data / "processed" / "train_source1.parquet", index=False
    )
    pd.DataFrame({"entity_id": ["S2-1", "S2-7"]}).to_parquet(
        data / "processed" / "train_source2.parquet", index=False
    )
    pd.DataFrame({"entity_id": ["S3-9"]}).to_parquet(
        data / "processed" / "train_source3.parquet", index=False
    )
    gt = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1,S3-9", "S2-7", ""],
        }
    )
    gt.to_parquet(data / "processed" / "train_ground_truth.parquet", index=False)
    cands = pd.DataFrame(
        {
            "s1_id": ["S1-1", "S1-1", "S1-2"],
            "cand_id": ["S2-1", "S3-9", "S2-7"],
            "pass_id": [1, 3, 1],
            "block_score": [1.0, 0.6, 1.0],
            "is_s2": [True, False, True],
        }
    )
    cands.to_parquet(data / "candidates" / "train_candidates.parquet", index=False)

    tsv = dataset / "train_source1.tsv"
    tsv.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tA\t1 rd\tUS\n"
        "S1-2\tB\t2 rd\tIndia\n"
        "S1-3\tC\t3 rd\tUS\n",
        encoding="utf-8",
    )
    return cands, gt


def test_run_audit_matches_pandas_reference(cfg):
    cands, gt = _seed(cfg)
    meta = pd.DataFrame(
        {"entity_id": ["S1-1", "S1-2", "S1-3"], "country": ["US", "India", "US"]}
    )
    n_cand_records = 3

    duck = run_audit(cfg, "train")
    ref = audit_candidates(cands, gt, meta, n_cand_records)

    assert set(duck) == set(ref)
    assert duck["recall"] == pytest.approx(ref["recall"])
    assert duck["recall_by_country"] == pytest.approx(ref["recall_by_country"])
    assert duck["reduction_ratio"] == pytest.approx(ref["reduction_ratio"])
    assert duck["candidates_per_s1_mean"] == pytest.approx(ref["candidates_per_s1_mean"])
    assert duck["singleton_candidates_mean"] == pytest.approx(ref["singleton_candidates_mean"])
    assert duck["per_pass_recall"] == pytest.approx(ref["per_pass_recall"])
    assert duck["truth_pairs"] == ref["truth_pairs"]
    assert duck["found_pairs"] == ref["found_pairs"]

    report = cfg.data_dir / "reports" / "train_blocking_audit.json"
    assert json.loads(report.read_text(encoding="utf-8")) == duck
