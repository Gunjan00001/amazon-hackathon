import json
from pathlib import Path

import duckdb
import jellyfish
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def compute_token_idf(files, min_idf, out_path):
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    files_sql = "[" + ",".join("'" + str(f).replace("\\", "/") + "'" for f in files) + "]"
    total = con.execute(f"SELECT COUNT(*) FROM read_parquet({files_sql})").fetchone()[0]
    counts = con.execute(
        "SELECT token, COUNT(DISTINCT entity_id) AS df FROM ("
        f"SELECT entity_id, UNNEST(name_idf_tokens) AS token FROM read_parquet({files_sql})"
        ") GROUP BY token"
    ).fetchall()
    rare = {
        token: float(np.log(total / df))
        for token, df in counts
        if df > 0 and np.log(total / df) >= min_idf
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": total, "min_idf": min_idf, "idf": rare}, ensure_ascii=False),
        encoding="utf-8",
    )
    con.close()
    return rare


def load_rare_tokens(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["idf"]


def _metaphone(token):
    if len(token) < 3:
        return ""
    try:
        return jellyfish.metaphone(token)
    except Exception:
        return ""


def block_keys(df, rare=None):
    eids = df["entity_id"].tolist()
    fold_col = df["name_fold"] if "name_fold" in df.columns else df["name_norm"]
    idf_col = df["name_idf_tokens"] if "name_idf_tokens" in df.columns else df["name_tokens"]
    norm_col = df["name_norm"].tolist()
    house_col = df["house_no"].tolist()
    street_col = df["street_tokens"].tolist()
    postal_col = df["postal"].tolist()
    state_col = df["state_key"].tolist()

    frames = [
        pd.DataFrame(
            {
                "entity_id": eids,
                "pass_id": 1,
                "key": normalize_keys(norm_col),
                "block_score": 1.0,
            }
        )
    ]

    pass2 = []
    pass3 = []
    pass4 = []
    pass5 = []
    pass6 = []
    rare = rare or {}
    for eid, fold, tokens, norm, house, street, postal, state in zip(
        eids, fold_col.tolist(), idf_col.tolist(), norm_col,
        house_col, street_col, postal_col, state_col,
    ):
        seen2 = set()
        for token in fold.split():
            if token in seen2:
                continue
            seen2.add(token)
            code = _metaphone(token)
            if code:
                pass2.append((eid, 2, code, 0.9))
        has_rare = False
        for token in set(tokens):
            score = rare.get(token)
            if score is not None:
                has_rare = True
                pass3.append((eid, 3, token, float(score)))
        if house and street:
            pass4.append((eid, 4, f"{house}|{street[0]}", 0.8))
        first_token = norm.split()[0] if norm.split() else ""
        if postal and first_token:
            pass5.append((eid, 5, f"{postal}|{first_token}", 0.7))
        if not has_rare and len(norm.split()) <= 2 and norm:
            pass6.append((eid, 6, f"{state}|{norm[:3]}", 0.5))

    frames.append(pd.DataFrame(pass2, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass3, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass4, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass5, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass6, columns=["entity_id", "pass_id", "key", "block_score"]))
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out = out[out["key"].astype(str).str.len() > 0].copy()
    out["key"] = out["key"].astype(str)
    out["block_score"] = out["block_score"].astype("float32")
    out["pass_id"] = out["pass_id"].astype("int8")
    return out


def normalize_keys(values):
    return pd.Series(values, dtype="string").fillna("").astype(str).tolist()


_JOIN_SQL = """
WITH s AS (SELECT entity_id, pass_id, key, block_score FROM {s1_src} WHERE key <> ''),
     c AS (SELECT entity_id, pass_id, key, block_score FROM {cand_src} WHERE key <> ''),
     valid AS (SELECT key FROM c GROUP BY key HAVING COUNT(*) <= {max_block}),
     j AS (
         SELECT s.entity_id AS s1_id, c.entity_id AS cand_id,
                s.pass_id AS pass_id, s.block_score AS block_score
         FROM s JOIN c USING (key) JOIN valid USING (key)
     ),
     dedup AS (
         SELECT s1_id, cand_id, MIN(pass_id) AS pass_id, MAX(block_score) AS block_score
         FROM j GROUP BY s1_id, cand_id
     ),
     ranked AS (
         SELECT *, ROW_NUMBER() OVER (
             PARTITION BY s1_id ORDER BY pass_id, block_score DESC, cand_id
         ) AS rn
         FROM dedup
     )
SELECT s1_id, cand_id, pass_id, block_score,
       CASE WHEN cand_id LIKE 'S2-%' THEN TRUE ELSE FALSE END AS is_s2
FROM ranked WHERE rn <= {cap}
"""


def _join_candidates(con, s1_src, cand_src, cap, max_block):
    sql = _JOIN_SQL.format(s1_src=s1_src, cand_src=cand_src, cap=int(cap), max_block=int(max_block))
    return con.execute(sql)


def generate_candidates(s1, cands, cfg):
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.register("s1k", block_keys(s1, getattr(cfg, "rare", None)))
    con.register("candk", block_keys(cands, getattr(cfg, "rare", None)))
    table = _join_candidates(con, "s1k", "candk", cfg.cap, cfg.max_block)
    table = table.to_arrow_table() if hasattr(table, "to_arrow_table") else table.fetch_arrow_table()
    out = table.to_pandas()
    con.close()
    return out


def _write_keys_chunked(parquet_path, key_path, rare):
    writer = None
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=1_000_000):
        df = batch.to_pandas()
        keys = block_keys(df, rare)
        table = __import__("pyarrow").Table.from_pandas(keys, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(key_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()


def run_block(cfg, split):
    processed = Path(cfg.data_dir) / "processed"
    reports = Path(cfg.data_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rare = compute_token_idf(
        [
            processed / f"{split}_source1.parquet",
            processed / f"{split}_source2.parquet",
            processed / f"{split}_source3.parquet",
        ],
        cfg.idf_min,
        reports / f"{split}_token_idf.json",
    )
    key_dir = Path(cfg.data_dir) / "keys"
    key_dir.mkdir(parents=True, exist_ok=True)
    _write_keys_chunked(processed / f"{split}_source1.parquet", key_dir / f"{split}_s1_keys.parquet", rare)
    writer_parts = []
    for source in (2, 3):
        part = key_dir / f"{split}_source{source}_keys.parquet"
        _write_keys_chunked(processed / f"{split}_source{source}.parquet", part, rare)
        writer_parts.append(part)
    cand_keys = key_dir / f"{split}_cand_keys.parquet"
    con = duckdb.connect()
    con.execute("SET memory_limit='12GB'")
    tmp = Path(cfg.data_dir) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    p1 = (key_dir / f"{split}_s1_keys.parquet").as_posix()
    p2 = writer_parts[0].as_posix()
    p3 = writer_parts[1].as_posix()
    con.execute(
        f"CREATE TABLE s1k AS SELECT * FROM read_parquet('{p1}')"
    )
    con.execute(
        f"CREATE TABLE candk AS SELECT * FROM read_parquet(['{p2}','{p3}'])"
    )
    out = Path(cfg.data_dir) / "candidates" / f"{split}_candidates.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    sql = _JOIN_SQL.format(s1_src="s1k", cand_src="candk", cap=int(cfg.cap), max_block=int(cfg.max_block))
    con.execute(f"COPY ({sql}) TO '{out.as_posix()}' (FORMAT PARQUET)")
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    con.close()
    return {"split": split, "candidates": int(count), "rare_tokens": len(rare)}
