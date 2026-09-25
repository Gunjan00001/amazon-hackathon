import json
import time
from pathlib import Path

import duckdb
import pandas as pd

from ber.io_utils import read_source

MEMORY_LIMIT = "8GB"
THREADS = 8
MAX_TEMP = "50GiB"


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
    """Reference (in-memory pandas) implementation. Kept for tests/cross-checks."""
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


class _Progress:
    def __init__(self, split):
        self.split = split
        self._last = time.perf_counter()
        self._start = self._last

    def log(self, message):
        now = time.perf_counter()
        print(
            f"[audit:{self.split}] {message} "
            f"(+{now - self._last:.1f}s, total {now - self._start:.1f}s)",
            flush=True,
        )
        self._last = now


def _sql_path(path) -> str:
    return Path(path).as_posix()


def _connect(cfg):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path(cfg.data_dir) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{_sql_path(tmp)}'")
    con.execute(f"PRAGMA max_temp_directory_size='{MAX_TEMP}'")
    return con


def _truth_sql(gt_path) -> str:
    return (
        "SELECT s1_id, cand_id FROM ("
        "  SELECT source1_entity_id AS s1_id, "
        "         UNNEST(string_split(matched_entity_ids, ',')) AS cand_id "
        f"  FROM read_parquet('{_sql_path(gt_path)}') WHERE matched_entity_ids <> ''"
        ") WHERE cand_id <> ''"
    )


def run_audit(cfg, split: str) -> dict:
    progress = _Progress(split)
    data = Path(cfg.data_dir)
    cand_path = data / "candidates" / f"{split}_candidates.parquet"
    gt_path = data / "processed" / "train_ground_truth.parquet"
    s1_path = data / "processed" / f"{split}_source1.parquet"
    country_path = Path(cfg.dataset_dir) / split / f"{split}_source1.tsv"

    con = _connect(cfg)

    con.execute(
        "CREATE TEMP TABLE cand AS "
        f"SELECT s1_id, cand_id, pass_id FROM read_parquet('{_sql_path(cand_path)}')"
    )
    n_cand_rows = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
    progress.log(f"loaded candidates: {n_cand_rows:,} rows")

    n_cand_records = 0
    for source in (2, 3):
        n_cand_records += con.execute(
            "SELECT COUNT(*) FROM read_parquet("
            f"'{_sql_path(data / 'processed' / f'{split}_source{source}.parquet')}')"
        ).fetchone()[0]
    progress.log(f"counted source2/3 records: {n_cand_records:,}")

    country_map = load_country_map(country_path)
    country_df = pd.DataFrame(
        {"s1_id": list(country_map.keys()), "country": list(country_map.values())}
    )
    con.register("country_map_df", country_df)
    con.execute(
        "CREATE TEMP TABLE s1_meta AS "
        "SELECT p.entity_id AS s1_id, m.country AS country "
        f"FROM read_parquet('{_sql_path(s1_path)}') p "
        "LEFT JOIN country_map_df m ON p.entity_id = m.s1_id"
    )
    n_s1 = con.execute("SELECT COUNT(*) FROM s1_meta").fetchone()[0]
    progress.log(f"built source1 metadata + country map: {n_s1:,} entities")

    con.execute(
        "CREATE TEMP TABLE truth AS "
        "SELECT t.s1_id, t.cand_id, m.country AS country "
        f"FROM ({_truth_sql(gt_path)}) t "
        "LEFT JOIN s1_meta m ON t.s1_id = m.s1_id"
    )
    total = con.execute("SELECT COUNT(*) FROM truth").fetchone()[0]
    progress.log(f"exploded ground truth: {total:,} truth pairs")

    con.execute(
        "CREATE TEMP TABLE matched AS "
        "SELECT t.s1_id, t.cand_id, t.country, c.pass_id "
        "FROM cand c JOIN truth t ON c.s1_id = t.s1_id AND c.cand_id = t.cand_id"
    )
    found = con.execute("SELECT COUNT(*) FROM matched").fetchone()[0]
    progress.log(f"matched candidates against truth: {found:,} found pairs")

    by_country = {
        str(country): (hits / n) if n else 0.0
        for country, n, hits in con.execute(
            "SELECT t.country, COUNT(*), "
            "       SUM(CASE WHEN m.s1_id IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM truth t "
            "LEFT JOIN (SELECT DISTINCT s1_id, cand_id FROM matched) m "
            "  ON t.s1_id = m.s1_id AND t.cand_id = m.cand_id "
            "WHERE t.country IS NOT NULL "
            "GROUP BY t.country"
        ).fetchall()
    }

    hit_by_pass = dict(
        con.execute("SELECT pass_id, COUNT(*) FROM matched GROUP BY pass_id").fetchall()
    )
    per_pass = {
        str(int(pass_id)): ((hit_by_pass.get(pass_id, 0) / total) if total else 0.0)
        for (pass_id,) in con.execute("SELECT DISTINCT pass_id FROM cand").fetchall()
    }

    cand_in_meta = con.execute(
        "SELECT COUNT(*) FROM cand c SEMI JOIN s1_meta m ON c.s1_id = m.s1_id"
    ).fetchone()[0]
    candidates_per_s1_mean = float(cand_in_meta / n_s1) if n_s1 else 0.0

    con.execute(
        "CREATE TEMP TABLE singleton AS "
        f"SELECT DISTINCT source1_entity_id AS s1_id FROM read_parquet('{_sql_path(gt_path)}') "
        "WHERE matched_entity_ids = ''"
    )
    singleton_denom = con.execute(
        "SELECT COUNT(*) FROM singleton s SEMI JOIN s1_meta m ON s.s1_id = m.s1_id"
    ).fetchone()[0]
    singleton_num = con.execute(
        "SELECT COUNT(*) FROM cand c SEMI JOIN singleton s ON c.s1_id = s.s1_id"
    ).fetchone()[0]
    singleton_candidates_mean = (
        float(singleton_num / singleton_denom) if singleton_denom else 0.0
    )

    reduction = (n_cand_rows / n_cand_records) if n_cand_records else None

    report = {
        "recall": (found / total) if total else 0.0,
        "recall_by_country": by_country,
        "reduction_ratio": reduction,
        "candidates_per_s1_mean": candidates_per_s1_mean,
        "singleton_candidates_mean": singleton_candidates_mean,
        "per_pass_recall": per_pass,
        "truth_pairs": int(total),
        "found_pairs": int(found),
    }

    reports = data / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{split}_blocking_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    progress.log("wrote report")
    con.close()
    return report
