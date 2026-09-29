import json

import pandas as pd

from ber.search import finalize, run_search


def _write_clean(base, n=200):
    base.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, labels = [], [], [], []
    for i in range(n):
        sid = f"S1-{i:05d}"
        if i % 2 == 0:
            s1.append((sid, f"Zyn{i}x Global Traders", f"12 Main Road City{i} 56{i:04d}", "India"))
            label = f"S2-{i:05d}"
        else:
            s1.append((sid, f"Lone{i}y Ventures", f"77 Park Avenue Town{i} 70{i:04d}", "India"))
            label = ""
        s2.append((f"S2-{i:05d}", f"Zyn{i}x Global Traders Private Limited", f"12 Main Rd City{i} 56{i:04d}", "India"))
        s3.append((f"S3-{i:05d}", f"Sunrise Foods {i}", f"9 Beach Road Town{i} 40{i:04d}", "India"))
        labels.append((sid, label))
    cols = ["entity_id", "business_name", "business_address", "country"]
    pd.DataFrame(s1, columns=cols).to_parquet(base / "train_s1.parquet", index=False)
    pd.DataFrame(s2, columns=cols).to_parquet(base / "train_s2.parquet", index=False)
    pd.DataFrame(s3, columns=cols).to_parquet(base / "train_s3.parquet", index=False)
    pd.DataFrame(s1, columns=cols).to_parquet(base / "test_s1.parquet", index=False)
    pd.DataFrame(s2, columns=cols).to_parquet(base / "test_s2.parquet", index=False)
    pd.DataFrame(s3, columns=cols).to_parquet(base / "test_s3.parquet", index=False)
    pd.DataFrame(labels, columns=["source1_entity_id", "matched_entity_ids"]).to_parquet(
        base / "labels.parquet", index=False)


def test_run_search_and_finalize_end_to_end(tmp_path):
    clean = tmp_path / "clean"
    _write_clean(clean)
    out = tmp_path / "search"
    space = {"max_postings": [1000], "max_candidates": [10], "num_leaves": [15],
             "learning_rate": [0.1], "top_k": [5], "one_to_one": [False]}
    best = run_search(clean, out, space=space, n_block=1, n_gbdt=1, s1_limit=0, budget_s=300)
    assert best["macro_f05"] > 0.7
    assert (out / "best.json").exists()
    assert (out / "trials.json").exists()

    metrics = finalize(clean, out, best, train_cap=10000)
    assert metrics["rows"] == 200
    assert metrics["matches"] > 0
    assert (out / "matching_results.tsv").exists()

    text = (out / "matching_results.tsv").read_text(encoding="utf-8").splitlines()
    assert text[0] == "source1_entity_id\tmatched_entity_ids"
    row = dict(line.split("\t") for line in text[1:])
    assert row["S1-00000"] == "S2-00000"
    assert row["S1-00001"] == ""
