import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_PASS_CAPS = {1: 5000, 3: 200, 4: 2000, 5: 1000, 6: 50, 7: 500, 8: 300, 9: 100, 10: 30}


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
    idf = {token: float(np.log(total / df)) for token, df in counts if df > 0}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": total, "min_idf": min_idf, "idf": idf}, ensure_ascii=False),
        encoding="utf-8",
    )
    con.close()
    return idf


def load_token_idf(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["idf"], float(payload["min_idf"])


def normalize_keys(values):
    return pd.Series(values, dtype="string").fillna("").astype(str).tolist()


def block_keys(df, idf_map=None, min_idf=4.0):
    idf_map = idf_map or {}
    eids = df["entity_id"].tolist()
    norm_col = df["name_norm"].tolist()
    idf_col = df["name_idf_tokens"] if "name_idf_tokens" in df.columns else df["name_tokens"]
    idf_col = [list(x) if x is not None else [] for x in idf_col.tolist()]
    house_col = df["house_no"].tolist()
    street_col = [list(x) if x is not None else [] for x in df["street_tokens"].tolist()]
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
    pass3 = []
    pass4 = []
    pass5 = []
    pass6 = []
    pass7 = []
    pass8 = []
    pass9 = []
    pass10 = []
    for eid, tokens, norm, house, street, postal, state in zip(
        eids, idf_col, norm_col, house_col, street_col, postal_col, state_col
    ):
        unique = set(tokens)
        has_rare = False
        for token in unique:
            score = idf_map.get(token, 0.0)
            if score >= min_idf:
                has_rare = True
                pass3.append((eid, 3, token, float(score)))
            if len(token) >= 5:
                pass7.append((eid, 7, token[:5], 0.75))
        rarest = sorted(((idf_map.get(t, 0.0), t) for t in unique), reverse=True)
        top = [t for _, t in rarest[:4]]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = sorted((top[i], top[j]))
                pass9.append((eid, 9, f"{a}|{b}", 0.65))
        if len(top) >= 3:
            a, b, c = sorted(top[:3])
            pass10.append((eid, 10, f"{a}|{b}|{c}", 0.6))
        if house and street:
            pass4.append((eid, 4, f"{house}|{street[0][:5]}", 0.8))
        for token in set(street):
            if len(token) >= 5:
                pass8.append((eid, 8, token[:5], 0.7))
        first_token = norm.split()[0] if norm.split() else ""
        if postal and first_token:
            pass5.append((eid, 5, f"{postal}|{first_token}", 0.7))
        if not has_rare and len(norm.split()) <= 2 and norm:
            pass6.append((eid, 6, f"{state}|{norm[:3]}", 0.5))

    frames.append(pd.DataFrame(pass3, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass4, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass5, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass6, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass7, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass8, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass9, columns=["entity_id", "pass_id", "key", "block_score"]))
    frames.append(pd.DataFrame(pass10, columns=["entity_id", "pass_id", "key", "block_score"]))
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out = out[out["key"].astype(str).str.len() > 0].copy()
    out["key"] = out["key"].astype(str)
    out["block_score"] = out["block_score"].astype("float32")
    out["pass_id"] = out["pass_id"].astype("int8")
    return out


def _pass_caps(cfg):
    caps = dict(DEFAULT_PASS_CAPS)
    for key, value in (getattr(cfg, "pass_caps", None) or {}).items():
        caps[int(key)] = int(value)
    return caps


def _valid_sql(s1_src, cand_src, caps):
    cases = " ".join(f"WHEN {int(p)} THEN {int(c)}" for p, c in sorted(caps.items()))
    return (
        "SELECT pass_id, key FROM {src} GROUP BY pass_id, key "
        "HAVING COUNT(*) <= CASE pass_id {cases} ELSE {fallback} END"
    ).format(src=cand_src, cases=cases, fallback=int(max(caps.values())))


def _join_sql(s1_src, cand_src, cap, caps):
    valid_expr = "(" + _valid_sql(s1_src, cand_src, caps) + ")"
    return f"""
WITH s AS (SELECT entity_id, pass_id, key, block_score FROM {s1_src} WHERE key <> ''),
     c AS (SELECT entity_id, pass_id, key, block_score FROM {cand_src} WHERE key <> ''),
     valid AS {valid_expr},
     j AS (
         SELECT s.entity_id AS s1_id, c.entity_id AS cand_id,
                s.pass_id AS pass_id, s.block_score AS block_score
         FROM s JOIN c ON s.key = c.key AND s.pass_id = c.pass_id
         JOIN valid v ON v.pass_id = s.pass_id AND v.key = s.key
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
FROM ranked WHERE rn <= {int(cap)}
"""


def generate_candidates(s1, cands, cfg):
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    idf_map = getattr(cfg, "idf_map", None)
    min_idf = getattr(cfg, "idf_min", 4.0)
    con.register("s1k", block_keys(s1, idf_map, min_idf))
    con.register("candk", block_keys(cands, idf_map, min_idf))
    sql = _join_sql("s1k", "candk", cfg.cap, _pass_caps(cfg))
    table = con.execute(sql)
    table = table.to_arrow_table() if hasattr(table, "to_arrow_table") else table.fetch_arrow_table()
    out = table.to_pandas()
    con.close()
    return out


def _write_keys_chunked(parquet_path, key_path, idf_map, min_idf):
    Path(key_path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=1_000_000):
        df = batch.to_pandas()
        keys = block_keys(df, idf_map, min_idf)
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
    idf_path = reports / f"{split}_token_idf.json"
    if idf_path.exists():
        idf_map, min_idf = load_token_idf(idf_path)
        min_idf = cfg.idf_min
    else:
        idf_map = compute_token_idf(
            [
                processed / f"{split}_source1.parquet",
                processed / f"{split}_source2.parquet",
                processed / f"{split}_source3.parquet",
            ],
            cfg.idf_min,
            idf_path,
        )
        min_idf = cfg.idf_min
    key_dir = Path(cfg.data_dir) / "keys"
    key_dir.mkdir(parents=True, exist_ok=True)
    s1_keys = key_dir / f"{split}_s1_keys.parquet"
    if not s1_keys.exists():
        _write_keys_chunked(processed / f"{split}_source1.parquet", s1_keys, idf_map, min_idf)
    cand_parts = []
    for source in (2, 3):
        part = key_dir / f"{split}_source{source}_keys.parquet"
        if not part.exists():
            _write_keys_chunked(processed / f"{split}_source{source}.parquet", part, idf_map, min_idf)
        cand_parts.append(part)

    con = duckdb.connect()
    con.execute("SET memory_limit='12GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path(cfg.data_dir) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute("PRAGMA max_temp_directory_size='50GiB'")
    con.execute(f"CREATE TABLE s1k AS SELECT * FROM read_parquet('{s1_keys.as_posix()}')")
    con.execute(
        "CREATE TABLE candk AS SELECT * FROM read_parquet("
        f"['{cand_parts[0].as_posix()}','{cand_parts[1].as_posix()}'])"
    )
    out = Path(cfg.data_dir) / "candidates" / f"{split}_candidates.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    sql = _join_sql("s1k", "candk", cfg.cap, _pass_caps(cfg))
    con.execute(f"COPY ({sql}) TO '{out.as_posix()}' (FORMAT PARQUET)")
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    con.close()
    return {"split": split, "candidates": int(count), "tokens": len(idf_map)}
