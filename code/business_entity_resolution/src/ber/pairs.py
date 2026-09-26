from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def _explode_truth(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s1, mids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if mids:
            for mid in mids.split(","):
                if mid:
                    rows.append((s1, mid))
    return pd.DataFrame(rows, columns=["s1_id", "cand_id"])


def build_training_pairs(candidates: pd.DataFrame, gt: pd.DataFrame, cfg) -> pd.DataFrame:
    truth = _explode_truth(gt)
    pos_pairs = set(map(tuple, truth[["s1_id", "cand_id"]].to_numpy()))
    labels = [
        1 if (s1, cid) in pos_pairs else 0
        for s1, cid in zip(candidates["s1_id"], candidates["cand_id"])
    ]
    out = candidates.copy()
    out["label"] = labels
    positives = out[out["label"] == 1]
    negatives = out[out["label"] == 0]
    target = int(cfg.neg_ratio * max(len(positives), 1))
    if len(negatives) > target:
        rng = np.random.default_rng(cfg.seed)
        keep = rng.choice(len(negatives), size=target, replace=False)
        negatives = negatives.iloc[keep]
    return pd.concat([positives, negatives], ignore_index=True)


def grouped_split(pairs: pd.DataFrame, val_frac: float, seed: int):
    codes, uniques = pd.factorize(pairs["s1_id"].to_numpy())
    codes = np.asarray(codes)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniques))
    n_val = int(len(uniques) * val_frac)
    val_codes = np.zeros(len(uniques), dtype=bool)
    val_codes[order[:n_val]] = True
    val_mask = val_codes[codes]
    return ~val_mask, val_mask


def write_training_pairs(cfg, split="train"):
    data = Path(cfg.data_dir)
    candidates = (data / "candidates" / f"{split}_candidates.parquet").as_posix()
    gt = (data / "processed" / "train_ground_truth.parquet").as_posix()
    out = data / "pairs" / f"{split}_pairs.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(data / 'tmp').as_posix()}'")
    con.execute("PRAGMA max_temp_directory_size='50GiB'")
    con.execute(
        f"CREATE TABLE cand AS SELECT s1_id, cand_id, is_s2, pass_id, block_score "
        f"FROM read_parquet('{candidates}')"
    )
    con.execute(
        f"CREATE TABLE truth AS SELECT source1_entity_id AS s1_id, UNNEST(string_split(matched_entity_ids, ',')) AS cand_id "
        f"FROM read_parquet('{gt}') WHERE matched_entity_ids <> ''"
    )
    pos = con.execute("SELECT COUNT(*) FROM cand JOIN truth USING (s1_id, cand_id)").fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
    target = int(cfg.neg_ratio * max(pos, 1))
    pct = max(min(target / max(total, 1) * 100.0, 100.0), 0.0001)
    sql = f"""
    WITH pos AS (
        SELECT c.*, 1 AS label FROM cand c JOIN truth t ON t.s1_id = c.s1_id AND t.cand_id = c.cand_id
    ), neg AS (
        SELECT c.*, 0 AS label FROM cand c
        ANTI JOIN truth t ON t.s1_id = c.s1_id AND t.cand_id = c.cand_id
        WHERE hash(c.s1_id || '|' || c.cand_id) % 1000000 < {int(pct * 10000)}
    )
    SELECT * FROM pos UNION ALL SELECT * FROM neg
    """
    con.execute(f"COPY ({sql}) TO '{out.as_posix()}' (FORMAT PARQUET)")
    counts = con.execute(
        f"SELECT label, COUNT(*) FROM read_parquet('{out.as_posix()}') GROUP BY label"
    ).fetchall()
    con.close()
    counts = {int(k): int(v) for k, v in counts}
    return {"positives": counts.get(1, 0), "negatives": counts.get(0, 0), "path": str(out)}


def write_inference_pairs(cfg, split="test"):
    data = Path(cfg.data_dir)
    candidates = (data / "candidates" / f"{split}_candidates.parquet").as_posix()
    out = data / "pairs" / f"{split}_pairs.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(data / 'tmp').as_posix()}'")
    con.execute("PRAGMA max_temp_directory_size='50GiB'")
    con.execute(
        f"COPY (SELECT s1_id, cand_id, is_s2, pass_id, block_score "
        f"FROM read_parquet('{candidates}')) TO '{out.as_posix()}' (FORMAT PARQUET)"
    )
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    con.close()
    return {"rows": int(rows), "path": str(out)}
