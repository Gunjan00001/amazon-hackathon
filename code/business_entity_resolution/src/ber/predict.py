import json
import shutil
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _load_booster(cfg):
    models = Path(cfg.models_dir)
    booster = lgb.Booster(model_file=str(models / "lgbm.txt"))
    features = json.loads((models / "feature_list.json").read_text(encoding="utf-8"))
    threshold = float(json.loads((models / "threshold.json").read_text(encoding="utf-8"))["global"])
    return booster, features, threshold


def _feature_sources(cfg, split):
    data = Path(cfg.data_dir)
    parts_dir = data / "tmp" / f"{split}_features"
    if parts_dir.exists() and any(parts_dir.glob("*.parquet")):
        return sorted(parts_dir.glob("*.parquet"))
    combined = data / "features" / f"{split}.parquet"
    if combined.exists():
        return [combined]
    raise FileNotFoundError(f"no feature parts or combined features for split={split}")


def predict_parts(cfg, split, booster, features, threshold, out_dir=None):
    data = Path(cfg.data_dir)
    pred_dir = Path(out_dir) if out_dir is not None else data / "tmp" / f"{split}_pred"
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    sources = _feature_sources(cfg, split)
    total = 0
    for i, part in enumerate(sources):
        writer = None
        part_rows = 0
        for batch in pq.ParquetFile(part).iter_batches(
            batch_size=2_000_000, columns=features + ["s1_id", "cand_id"]
        ):
            frame = batch.to_pandas()
            probs = booster.predict(frame[features], num_iteration=booster.best_iteration)
            mask = probs >= threshold
            if mask.any():
                out = pd.DataFrame(
                    {
                        "s1_id": frame["s1_id"].to_numpy()[mask],
                        "cand_id": frame["cand_id"].to_numpy()[mask],
                        "prob": probs[mask].astype(np.float32),
                    }
                )
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(pred_dir / f"part_{i:04d}.parquet", table.schema)
                writer.write_table(table)
                part_rows += len(out)
        if writer is not None:
            writer.close()
        total += part_rows
        print(f"[predict] part {i + 1}/{len(sources)}: {part_rows:,} matches", flush=True)
    return pred_dir, total


def _write_tsv(cfg, split, use_one_to_one, n_buckets=64):
    data = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = (data / "pairs" / f"{split}_pairs.parquet").as_posix()
    preds = (data / "tmp" / f"{split}_pred" / "*.parquet").as_posix()
    s1_parquet = (data / "processed" / f"{split}_source1.parquet").as_posix()
    matching = out_dir / "matching_results.tsv"
    candidate = out_dir / "candidate_pairs.tsv"

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(data / 'tmp').as_posix()}'")
    con.execute("PRAGMA max_temp_directory_size='60GiB'")
    con.execute(
        f"CREATE TEMP TABLE s1list AS SELECT entity_id AS source1_entity_id FROM read_parquet('{s1_parquet}')"
    )

    if use_one_to_one:
        con.execute(
            f"""
            CREATE TEMP TABLE matches AS
            SELECT s1_id, cand_id FROM (
                SELECT s1_id, cand_id,
                       row_number() OVER (PARTITION BY cand_id ORDER BY prob DESC, s1_id ASC) AS rn
                FROM read_parquet('{preds}')
            ) WHERE rn = 1
            """
        )
    else:
        con.execute(
            f"CREATE TEMP TABLE matches AS SELECT DISTINCT s1_id, cand_id FROM read_parquet('{preds}')"
        )

    con.execute(
        f"""
        COPY (
            WITH agg AS (
                SELECT s1_id, string_agg(cand_id, ',' ORDER BY cand_id) AS ids
                FROM matches GROUP BY s1_id
            )
            SELECT s.source1_entity_id, coalesce(a.ids, '') AS matched_entity_ids
            FROM s1list s LEFT JOIN agg a ON a.s1_id = s.source1_entity_id
            ORDER BY s.source1_entity_id
        ) TO '{matching.as_posix()}' (FORMAT CSV, HEADER, DELIM '\t', QUOTE '')
        """
    )
    print("[predict] wrote matching_results.tsv", flush=True)

    bucket_dir = data / "tmp" / f"{split}_cand_buckets"
    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)
    con.execute(
        f"COPY (SELECT s1_id, cand_id, hash(s1_id) % {int(n_buckets)} AS __b FROM read_parquet('{pairs}')) "
        f"TO '{bucket_dir.as_posix()}' (FORMAT PARQUET, PARTITION_BY (__b))"
    )
    parts_dir = data / "tmp" / f"{split}_cand_tsv"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_files = []
    for b in range(n_buckets):
        part = parts_dir / f"part_{b:03d}.tsv"
        glob = (bucket_dir / f"__b={b}").as_posix() + "/*.parquet"
        con.execute(
            f"""
            COPY (
                WITH cand AS (SELECT s1_id, cand_id FROM read_parquet('{glob}')),
                     agg AS (
                         SELECT s1_id, string_agg(cand_id, ',' ORDER BY cand_id) AS ids
                         FROM cand GROUP BY s1_id
                     )
                SELECT s.source1_entity_id, coalesce(a.ids, '') AS candidate_entity_ids
                FROM (SELECT source1_entity_id FROM s1list WHERE hash(source1_entity_id) % {int(n_buckets)} = {b}) s
                LEFT JOIN agg a ON a.s1_id = s.source1_entity_id
            ) TO '{part.as_posix()}' (FORMAT CSV, HEADER false, DELIM '\t', QUOTE '')
            """
        )
        part_files.append(part)
    with open(candidate, "wb") as fout:
        fout.write(b"source1_entity_id\tcandidate_entity_ids\n")
        for part in part_files:
            fout.write(part.read_bytes())
    print("[predict] wrote candidate_pairs.tsv", flush=True)

    match_dist = con.execute(
        """
        WITH agg AS (SELECT s1_id, count(*) AS n FROM matches GROUP BY s1_id)
        SELECT coalesce(a.n, 0) AS n_matches, count(*) AS entities
        FROM s1list s LEFT JOIN agg a ON a.s1_id = s.source1_entity_id
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    n_matching = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{matching.as_posix()}', delim='\t', header=true)"
    ).fetchone()[0]
    n_candidate = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{candidate.as_posix()}', delim='\t', header=true)"
    ).fetchone()[0]
    con.close()
    return {
        "matching_rows": int(n_matching),
        "candidate_rows": int(n_candidate),
        "matching_path": matching.as_posix(),
        "candidate_path": candidate.as_posix(),
        "match_count_distribution": {int(k): int(v) for k, v in match_dist},
    }


def run_predict(cfg, split="test", use_one_to_one=False, threshold_override=None, reuse_predictions=False):
    booster, features, threshold = _load_booster(cfg)
    if threshold_override is not None:
        threshold = float(threshold_override)
    print(f"[predict] model={booster.num_trees()} trees, threshold={threshold}", flush=True)
    data = Path(cfg.data_dir)
    pred_dir = data / "tmp" / f"{split}_pred"
    if reuse_predictions and pred_dir.exists() and any(pred_dir.glob("*.parquet")):
        print("[predict] reusing existing prediction parts", flush=True)
        n_pred = None
    else:
        pred_dir, n_pred = predict_parts(cfg, split, booster, features, threshold)
    stats = _write_tsv(cfg, split, use_one_to_one)
    stats["pairs_above_threshold"] = int(n_pred) if n_pred is not None else "reused"
    stats["use_one_to_one"] = bool(use_one_to_one)
    stats["threshold"] = threshold
    return stats
