import json
from pathlib import Path

import duckdb
import numpy as np

from ber.validation import _country_of_source1, val_s1_ids_path


def oracle_macro(truth, cand):
    ids = set(truth) | set(cand)
    scores = []
    for s1_id in ids:
        true_set = set(truth.get(s1_id, ()))
        pred = true_set & set(cand.get(s1_id, ()))
        if not true_set:
            scores.append(1.0 if not pred else 0.0)
            continue
        if not pred:
            scores.append(0.0)
            continue
        precision = len(pred) / len(pred)
        recall = len(pred) / len(true_set)
        scores.append((1.25 * precision * recall) / (0.25 * precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def _oracle_scores(ntrue, nfound):
    has_true = ntrue > 0
    scores = np.ones(len(ntrue), dtype=np.float64)
    scores[has_true & (nfound == 0)] = 0.0
    valid = has_true & (nfound > 0)
    recall = nfound[valid] / ntrue[valid]
    scores[valid] = (1.25 * recall) / (0.25 + recall)
    return scores


def run_diagnostic(cfg, candidates_path=None, out_path=None, label=None):
    data = Path(cfg.data_dir)
    if candidates_path is not None:
        candidates = Path(candidates_path).as_posix()
    else:
        candidates = (data / "candidates" / "train_candidates.parquet").as_posix()
    gt = (data / "processed" / "train_ground_truth.parquet").as_posix()
    valids = val_s1_ids_path(cfg)

    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(data / 'tmp').as_posix()}'")
    con.execute("PRAGMA max_temp_directory_size='50GiB'")
    con.execute(f"CREATE TEMP TABLE valids AS SELECT s1_id FROM read_parquet('{valids.as_posix()}')")
    n_cand_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{candidates}')").fetchone()[0]
    n_cand_val = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{candidates}') c SEMI JOIN valids v ON v.s1_id = c.s1_id"
    ).fetchone()[0]
    con.execute(
        f"""
        CREATE TEMP TABLE tcount AS
        SELECT s1_id, COUNT(DISTINCT cand_id) AS ntrue FROM (
            SELECT g.source1_entity_id AS s1_id,
                   UNNEST(string_split(g.matched_entity_ids, ',')) AS cand_id
            FROM read_parquet('{gt}') g SEMI JOIN valids v ON v.s1_id = g.source1_entity_id
            WHERE g.matched_entity_ids <> ''
        ) GROUP BY s1_id
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE found AS
        SELECT t.s1_id, COUNT(DISTINCT t.cand_id) AS nfound
        FROM (
            SELECT g.source1_entity_id AS s1_id, UNNEST(string_split(g.matched_entity_ids, ',')) AS cand_id
            FROM read_parquet('{gt}') g SEMI JOIN valids v ON v.s1_id = g.source1_entity_id
            WHERE g.matched_entity_ids <> ''
        ) t
        JOIN (SELECT DISTINCT s1_id, cand_id FROM read_parquet('{candidates}')) c
          ON c.s1_id = t.s1_id AND c.cand_id = t.cand_id
        GROUP BY t.s1_id
        """
    )
    rows = con.execute(
        """
        SELECT v.s1_id, COALESCE(tc.ntrue, 0) AS ntrue, COALESCE(f.nfound, 0) AS nfound
        FROM valids v
        LEFT JOIN tcount tc ON tc.s1_id = v.s1_id
        LEFT JOIN found f ON f.s1_id = v.s1_id
        """
    ).fetchall()
    con.close()

    ids = np.array([r[0] for r in rows], dtype=object)
    ntrue = np.array([int(r[1]) for r in rows], dtype=np.int64)
    nfound = np.array([int(r[2]) for r in rows], dtype=np.int64)

    scores = _oracle_scores(ntrue, nfound)
    has_true = ntrue > 0
    total_truth = int(ntrue.sum())
    found_truth = int(nfound.sum())

    country_by_s1 = _country_of_source1(cfg, "train")
    per_country = {}
    recall_by_country = {}
    for country in sorted(set(country_by_s1.get(sid, "") for sid in ids)):
        mask = np.array([country_by_s1.get(sid, "") == country for sid in ids])
        if mask.any():
            per_country[country] = float(scores[mask].mean())
            n_t = int(ntrue[mask].sum())
            n_f = int(nfound[mask].sum())
            recall_by_country[country] = (n_f / n_t) if n_t else 0.0

    candidates_per_s1 = (n_cand_val / len(ids)) if len(ids) else 0.0
    report = {
        "val_s1": int(len(ids)),
        "oracle_macro_f05": float(scores.mean()),
        "oracle_by_country": per_country,
        "pair_recall": (found_truth / total_truth) if total_truth else 0.0,
        "recall_by_country": recall_by_country,
        "total_truth_pairs": total_truth,
        "found_truth_pairs": found_truth,
        "entities_with_truth": int(has_true.sum()),
        "entities_with_zero_found": int((has_true & (nfound == 0)).sum()),
        "share_entities_with_zero_found": float((has_true & (nfound == 0)).sum() / has_true.sum()) if has_true.any() else 0.0,
        "mean_entities_recall": float((nfound[has_true] / ntrue[has_true]).mean()) if has_true.any() else 0.0,
        "total_candidates": int(n_cand_total),
        "heldout_candidates": int(n_cand_val),
        "candidates_per_s1": float(candidates_per_s1),
        "baseline_heldout_f05": 0.8577,
        "note": "oracle = ground-truth pairs intersected with candidates (prediction precision fixed at 1.0); held-out S1 groups",
    }
    if label:
        report["label"] = label
    out = Path(out_path) if out_path is not None else (data / "reports" / "eval_oracle.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[diagnose] oracle_macro_f05={report['oracle_macro_f05']:.4f} "
        f"(baseline 0.8577, headroom {report['oracle_macro_f05'] - 0.8577:+.4f}) "
        f"pair_recall={report['pair_recall']:.4f} "
        f"cand_per_s1={candidates_per_s1:.1f} "
        f"zero_found={report['entities_with_zero_found']:,}/{report['entities_with_truth']:,} "
        f"({report['share_entities_with_zero_found']:.3%}) "
        f"by_country={ {k: round(v, 4) for k, v in per_country.items()} }",
        flush=True,
    )
    return report
