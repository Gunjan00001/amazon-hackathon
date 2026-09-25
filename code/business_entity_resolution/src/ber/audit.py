from pathlib import Path

import pandas as pd

from ber.io_utils import read_source


def load_country_map(path) -> dict:
    mapping = {}
    for chunk in read_source(path):
        mapping.update(zip(chunk["entity_id"].tolist(), chunk["country"].tolist()))
    return mapping


def _explode_truth(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s1, mids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if mids:
            for mid in mids.split(","):
                if mid:
                    rows.append((s1, mid))
    return pd.DataFrame(rows, columns=["s1_id", "cand_id"])


def audit_candidates(candidates: pd.DataFrame, gt: pd.DataFrame, s1_meta: pd.DataFrame, n_cand_records: int = None) -> dict:
    truth = _explode_truth(gt)
    cand_pairs = set(map(tuple, candidates[["s1_id", "cand_id"]].to_numpy()))
    truth_pairs = list(map(tuple, truth[["s1_id", "cand_id"]].to_numpy()))
    total = len(truth_pairs)
    found = sum(1 for pair in truth_pairs if pair in cand_pairs)

    country_by_s1 = dict(zip(s1_meta["entity_id"], s1_meta["country"]))
    truth["country"] = truth["s1_id"].map(country_by_s1)
    by_country = {}
    for country, group in truth.groupby("country"):
        pairs = list(map(tuple, group[["s1_id", "cand_id"]].to_numpy()))
        hit = sum(1 for pair in pairs if pair in cand_pairs)
        by_country[str(country)] = (hit / len(pairs)) if pairs else 0.0

    counts = candidates.groupby("s1_id").size()
    all_s1 = s1_meta["entity_id"].tolist()
    counts = counts.reindex(all_s1, fill_value=0)
    singleton_ids = set(gt.loc[gt["matched_entity_ids"] == "", "source1_entity_id"])
    singleton_counts = counts[counts.index.isin(singleton_ids)]

    per_pass = {}
    for pass_id, group in candidates.groupby("pass_id"):
        pairs = set(map(tuple, group[["s1_id", "cand_id"]].to_numpy()))
        hit = sum(1 for pair in truth_pairs if pair in pairs)
        per_pass[str(int(pass_id))] = (hit / total) if total else 0.0

    reduction = None
    if n_cand_records:
        reduction = len(candidates) / n_cand_records

    return {
        "recall": found / total if total else 0.0,
        "recall_by_country": by_country,
        "reduction_ratio": reduction,
        "candidates_per_s1_mean": float(counts.mean()),
        "singleton_candidates_mean": float(singleton_counts.mean()) if len(singleton_counts) else 0.0,
        "per_pass_recall": per_pass,
        "truth_pairs": total,
        "found_pairs": found,
    }


def run_audit(cfg, split: str) -> dict:
    data = Path(cfg.data_dir)
    candidates = pd.read_parquet(data / "candidates" / f"{split}_candidates.parquet")
    gt = pd.read_parquet(data / "processed" / "train_ground_truth.parquet")
    meta = pd.read_parquet(data / "processed" / f"{split}_source1.parquet", columns=["entity_id"])
    country_path = Path(cfg.dataset_dir) / split / f"{split}_source1.tsv"
    country_map = load_country_map(country_path)
    meta = meta.assign(country=meta["entity_id"].map(country_map))
    n_cand_records = 0
    for source in (2, 3):
        n_cand_records += len(
            pd.read_parquet(data / "processed" / f"{split}_source{source}.parquet", columns=["entity_id"])
        )
    report = audit_candidates(candidates, gt, meta, n_cand_records)
    reports = data / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    import json

    (reports / f"{split}_blocking_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
