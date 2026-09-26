import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

S1_COLS = [
    "entity_id", "name_norm", "name_fold", "name_roman", "name_tokens", "name_stripped",
    "suffix_class", "addr_norm", "addr_raw_missing", "house_no", "street_tokens",
    "postal", "state_key", "landmark_flag", "name_script",
]

FEATURE_ORDER = [
    "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_wratio",
    "name_qratio", "name_jaro", "name_exact", "name_jaccard", "name_containment",
    "name_sorted_eq", "name_len_diff", "token_count_diff", "roman_ratio", "script_match",
    "addr_ratio", "addr_token_sort", "addr_jaccard", "addr_containment", "house_match",
    "street_jaccard", "postal_exact", "postal_prefix3", "state_match", "landmark",
    "addr_missing", "addr_len_diff", "same_country", "is_s2", "pass_id", "block_score",
    "s1_degree", "cand_degree",
]


def load_frame(path, columns=S1_COLS):
    return pd.read_parquet(path, columns=columns)


_COUNTRY_CACHE = {}


def _country_map(cfg, split):
    key = (str(cfg.dataset_dir), split)
    if key not in _COUNTRY_CACHE:
        mapping = {}
        for source in (1, 2, 3):
            path = Path(cfg.dataset_dir) / split / f"{split}_source{source}.tsv"
            if not path.exists():
                continue
            countries = pd.read_csv(
                path, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id", "country"]
            )
            mapping.update(zip(countries["entity_id"], countries["country"].str.strip().str.lower()))
        _COUNTRY_CACHE[key] = mapping
    return _COUNTRY_CACHE[key]


def attach_country(frame, cfg, split, id_col="entity_id"):
    mapping = _country_map(cfg, split)
    return frame.assign(country=frame[id_col].map(mapping).fillna(""))


def _tokens(value):
    if value is None:
        return frozenset()
    if isinstance(value, (list, tuple, np.ndarray)):
        return frozenset(str(v) for v in value)
    return frozenset(str(value).split())


def _pair_set_metrics(tokens1, tokens2):
    jaccard = []
    containment = []
    for a, b in zip(tokens1, tokens2):
        union = a | b
        inter = a & b
        jaccard.append(len(inter) / len(union) if union else 0.0)
        containment.append(len(inter) / min(len(a), len(b)) if (a and b) else 0.0)
    return np.asarray(jaccard, dtype=np.float32), np.asarray(containment, dtype=np.float32)


def _feature_block(merged):
    s1_name = merged["name_fold"].astype(str).tolist()
    c_name = merged["name_fold_2"].astype(str).tolist()
    s1_norm = merged["name_norm"].astype(str).tolist()
    c_norm = merged["name_norm_2"].astype(str).tolist()
    s1_roman = merged["name_roman"].astype(str).tolist()
    c_roman = merged["name_roman_2"].astype(str).tolist()
    s1_addr = merged["addr_norm"].astype(str).tolist()
    c_addr = merged["addr_norm_2"].astype(str).tolist()

    out = pd.DataFrame(index=merged.index)
    out["name_ratio"] = process.cpdist(s1_name, c_name, scorer=fuzz.ratio, dtype=np.float32)
    out["name_partial"] = process.cpdist(s1_name, c_name, scorer=fuzz.partial_ratio, dtype=np.float32)
    out["name_token_sort"] = process.cpdist(s1_name, c_name, scorer=fuzz.token_sort_ratio, dtype=np.float32)
    out["name_token_set"] = process.cpdist(s1_name, c_name, scorer=fuzz.token_set_ratio, dtype=np.float32)
    out["name_wratio"] = process.cpdist(s1_name, c_name, scorer=fuzz.WRatio, dtype=np.float32)
    out["name_qratio"] = process.cpdist(s1_name, c_name, scorer=fuzz.QRatio, dtype=np.float32)
    out["name_jaro"] = process.cpdist(s1_name, c_name, scorer=JaroWinkler.similarity, dtype=np.float32)
    out["name_exact"] = (merged["name_norm"].to_numpy() == merged["name_norm_2"].to_numpy()).astype(np.float32)
    s1_tokens = [_tokens(v) for v in merged["name_tokens"]]
    c_tokens = [_tokens(v) for v in merged["name_tokens_2"]]
    jacc, contain = _pair_set_metrics(s1_tokens, c_tokens)
    out["name_jaccard"] = jacc
    out["name_containment"] = contain
    out["name_sorted_eq"] = np.array(
        [1.0 if " ".join(sorted(a)) == " ".join(sorted(b)) and a else 0.0 for a, b in zip(s1_tokens, c_tokens)],
        dtype=np.float32,
    )
    len1 = np.array([len(s) for s in s1_norm], dtype=np.float32)
    len2 = np.array([len(s) for s in c_norm], dtype=np.float32)
    out["name_len_diff"] = np.abs(len1 - len2)
    out["token_count_diff"] = np.abs(
        np.array([len(s.split()) for s in s1_norm], dtype=np.float32)
        - np.array([len(s.split()) for s in c_norm], dtype=np.float32)
    )
    out["roman_ratio"] = process.cpdist(s1_roman, c_roman, scorer=fuzz.ratio, dtype=np.float32)
    out["script_match"] = (merged["name_script"].to_numpy() == merged["name_script_2"].to_numpy()).astype(np.float32)

    both_addr = (
        (merged["addr_norm"].astype(str) != "") & (merged["addr_norm_2"].astype(str) != "")
    )
    out["addr_ratio"] = process.cpdist(s1_addr, c_addr, scorer=fuzz.ratio, dtype=np.float32)
    out.loc[~both_addr, "addr_ratio"] = -1.0
    out["addr_token_sort"] = process.cpdist(s1_addr, c_addr, scorer=fuzz.token_sort_ratio, dtype=np.float32)
    out.loc[~both_addr, "addr_token_sort"] = -1.0
    a1 = [_tokens(v) for v in merged["street_tokens"]]
    a2 = [_tokens(v) for v in merged["street_tokens_2"]]
    ajacc, acontain = _pair_set_metrics(a1, a2)
    out["addr_jaccard"] = np.where(both_addr.to_numpy(), ajacc, -1.0).astype(np.float32)
    out["addr_containment"] = np.where(both_addr.to_numpy(), acontain, -1.0).astype(np.float32)
    out["house_match"] = (
        (merged["house_no"].astype(str) != "")
        & (merged["house_no"].astype(str) == merged["house_no_2"].astype(str))
    ).astype(np.float32)
    out["street_jaccard"] = ajacc
    out["postal_exact"] = (
        (merged["postal"].astype(str) != "") & (merged["postal"].astype(str) == merged["postal_2"].astype(str))
    ).astype(np.float32)
    p1 = merged["postal"].astype(str)
    p2 = merged["postal_2"].astype(str)
    out["postal_prefix3"] = (
        (p1.str.len() >= 3) & (p1.str.slice(0, 3) == p2.str.slice(0, 3))
    ).astype(np.float32)
    out["state_match"] = (
        (merged["state_key"].astype(str) != "") & (merged["state_key"].astype(str) == merged["state_key_2"].astype(str))
    ).astype(np.float32)
    out["landmark"] = merged["landmark_flag"].fillna(False).astype(bool).astype(np.float32)
    out["addr_missing"] = merged["addr_raw_missing"].fillna(False).astype(bool).astype(np.float32)
    a1_len = np.array([len(s) for s in merged["addr_norm"].astype(str)], dtype=np.float32)
    a2_len = np.array([len(s) for s in merged["addr_norm_2"].astype(str)], dtype=np.float32)
    out["addr_len_diff"] = np.abs(a1_len - a2_len)
    out["same_country"] = (
        (merged["country"].astype(str) != "") & (merged["country"].astype(str) == merged["country_2"].astype(str))
    ).astype(np.float32)
    out["is_s2"] = merged["is_s2"].astype(np.float32)
    out["pass_id"] = merged["pass_id"].astype(np.float32)
    out["block_score"] = merged["block_score"].astype(np.float32)
    out["s1_degree"] = merged["s1_degree"].astype(np.float32)
    out["cand_degree"] = merged["cand_degree"].astype(np.float32)
    return out


def compute_features(pairs, s1, cand, cfg, chunk_size=3_000_000):
    s1 = attach_country(s1, cfg, "train") if "country" not in s1.columns else s1
    cand = attach_country(cand, cfg, "train", id_col="entity_id") if "country" not in cand.columns else cand
    keep1 = ["entity_id", "name_norm", "name_fold", "name_roman", "name_tokens", "name_script",
             "addr_norm", "addr_raw_missing", "house_no", "street_tokens", "postal", "state_key",
             "landmark_flag", "country"]
    keep2 = keep1
    s1 = s1[[c for c in keep1 if c in s1.columns]]
    cand = cand[[c for c in keep2 if c in cand.columns]]

    degrees_s1 = pairs.groupby("s1_id").size()
    degrees_cand = pairs.groupby("cand_id").size()
    parts = []
    n = len(pairs)
    n_chunks = (n + chunk_size - 1) // chunk_size
    for start in range(0, n, chunk_size):
        print(
            f"[features] chunk {start // chunk_size + 1}/{n_chunks} "
            f"({start:,}/{n:,} pairs)",
            flush=True,
        )
        chunk = pairs.iloc[start:start + chunk_size].copy()
        chunk["s1_degree"] = chunk["s1_id"].map(degrees_s1).astype(np.float32)
        chunk["cand_degree"] = chunk["cand_id"].map(degrees_cand).astype(np.float32)
        a = s1[s1["entity_id"].isin(chunk["s1_id"])]
        b = cand[cand["entity_id"].isin(chunk["cand_id"])]
        merged = chunk.merge(a, left_on="s1_id", right_on="entity_id", how="left", suffixes=("", "_1"))
        merged = merged.merge(b, left_on="cand_id", right_on="entity_id", how="left", suffixes=("", "_2"))
        feats = _feature_block(merged)
        feats["s1_id"] = chunk["s1_id"].to_numpy()
        feats["cand_id"] = chunk["cand_id"].to_numpy()
        if "label" in chunk.columns:
            feats["label"] = chunk["label"].to_numpy()
        parts.append(feats)
    out = pd.concat(parts, ignore_index=True)
    return out


def _pairs_has_label(path) -> bool:
    return "label" in pq.ParquetFile(path).schema_arrow.names


def _phase1_merged(cfg, split, n_parts=16, meta_split=None):
    meta_split = meta_split or split
    data = Path(cfg.data_dir)
    pairs_path = data / "pairs" / f"{split}_pairs.parquet"
    s1_path = data / "processed" / f"{meta_split}_source1.parquet"
    cand_paths = [data / "processed" / f"{meta_split}_source{source}.parquet" for source in (2, 3)]
    out_dir = data / "tmp" / f"{split}_merged"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    country_files = [
        Path(cfg.dataset_dir) / meta_split / f"{meta_split}_source{source}.tsv"
        for source in (1, 2, 3)
    ]
    country_files = [f.as_posix() for f in country_files if f.exists()]

    con = duckdb.connect()
    con.execute("SET memory_limit='12GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(data / 'tmp').as_posix()}'")
    con.execute("PRAGMA max_temp_directory_size='80GiB'")

    if country_files:
        files_sql = "[" + ",".join(f"'{f}'" for f in country_files) + "]"
        con.execute(
            "CREATE TEMP TABLE country AS SELECT entity_id, lower(trim(country)) AS country "
            f"FROM read_csv({files_sql}, delim='\t', header=true, union_by_name=true)"
        )
    else:
        con.execute(
            "CREATE TEMP TABLE country AS SELECT CAST(NULL AS VARCHAR) AS entity_id, "
            "CAST(NULL AS VARCHAR) AS country WHERE false"
        )

    con.execute(f"CREATE TEMP TABLE pairs AS SELECT * FROM read_parquet('{pairs_path.as_posix()}')")
    con.execute(
        "CREATE TEMP TABLE s1 AS SELECT x.*, coalesce(c.country, '') AS country "
        f"FROM read_parquet('{s1_path.as_posix()}') x LEFT JOIN country c ON c.entity_id = x.entity_id"
    )
    cand_sql = "[" + ",".join(f"'{p.as_posix()}'" for p in cand_paths) + "]"
    con.execute(
        "CREATE TEMP TABLE cand AS SELECT x.*, coalesce(c.country, '') AS country "
        f"FROM read_parquet({cand_sql}) x LEFT JOIN country c ON c.entity_id = x.entity_id"
    )
    con.execute("CREATE TEMP TABLE sdeg AS SELECT s1_id, count(*) AS n FROM pairs GROUP BY s1_id")
    con.execute("CREATE TEMP TABLE cdeg AS SELECT cand_id, count(*) AS n FROM pairs GROUP BY cand_id")

    label_sel = "p.label," if _pairs_has_label(pairs_path) else ""
    sql = f"""
    SELECT
        p.s1_id, p.cand_id, p.is_s2, p.pass_id, p.block_score, {label_sel}
        s.name_norm, s.name_fold, s.name_roman, s.name_tokens, s.name_script,
        s.addr_norm, s.addr_raw_missing, s.house_no, s.street_tokens, s.postal,
        s.state_key, s.landmark_flag, coalesce(s.country, '') AS country,
        c.name_norm AS name_norm_2, c.name_fold AS name_fold_2, c.name_roman AS name_roman_2,
        c.name_tokens AS name_tokens_2, c.name_script AS name_script_2,
        c.addr_norm AS addr_norm_2, c.house_no AS house_no_2,
        c.street_tokens AS street_tokens_2, c.postal AS postal_2, c.state_key AS state_key_2,
        coalesce(c.country, '') AS country_2,
        sd.n AS s1_degree, cd.n AS cand_degree,
        hash(p.s1_id) % {int(n_parts)} AS __part
    FROM pairs p
    LEFT JOIN s1 s ON s.entity_id = p.s1_id
    LEFT JOIN cand c ON c.entity_id = p.cand_id
    LEFT JOIN sdeg sd ON sd.s1_id = p.s1_id
    LEFT JOIN cdeg cd ON cd.cand_id = p.cand_id
    """
    tmp_out = (data / "tmp" / f"{split}_merged_copy").as_posix()
    if Path(tmp_out).exists():
        shutil.rmtree(tmp_out)
    con.execute(
        f"COPY ({sql}) TO '{tmp_out}' (FORMAT PARQUET, PARTITION_BY (__part), "
        "WRITE_PARTITION_COLUMNS false, FILENAME_PATTERN 'part_{i}')"
    )
    con.close()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.move(tmp_out, out_dir)
    parts = sorted(out_dir.rglob("*.parquet"))
    return parts


def _process_part(args):
    part_path, out_path = args
    writer = None
    rows = 0
    for batch in pq.ParquetFile(part_path).iter_batches(batch_size=500_000):
        merged = batch.to_pandas()
        feats = _feature_block(merged)
        feats["s1_id"] = merged["s1_id"].to_numpy()
        feats["cand_id"] = merged["cand_id"].to_numpy()
        if "label" in merged.columns:
            feats["label"] = merged["label"].to_numpy()
        table = pa.Table.from_pandas(feats, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        rows += len(feats)
    if writer is not None:
        writer.close()
    return {"part": Path(out_path).name, "rows": rows}


def _combine_feature_parts(part_dir, out_path):
    writer = None
    rows = 0
    for part in sorted(Path(part_dir).glob("*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(batch_size=1_000_000):
            if writer is None:
                writer = pq.ParquetWriter(out_path, batch.schema)
            writer.write_table(pa.Table.from_batches([batch]))
            rows += batch.num_rows
    if writer is not None:
        writer.close()
    return rows


def run_features(cfg, split="train", workers=1, combine=False, n_parts=16, keep_merged=False, meta_split=None):
    data = Path(cfg.data_dir)
    merged_parts = _phase1_merged(cfg, split, n_parts=n_parts, meta_split=meta_split)
    print(f"[features] phase1 merged parts: {len(merged_parts)}", flush=True)

    feat_dir = data / "tmp" / f"{split}_features"
    if feat_dir.exists():
        shutil.rmtree(feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(part), str(feat_dir / f"part_{i:04d}.parquet"))
        for i, part in enumerate(merged_parts)
    ]

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_process_part, tasks))
    else:
        results = [_process_part(task) for task in tasks]
    rows = sum(r["rows"] for r in results)
    print(f"[features] phase2 done: {rows:,} feature rows in {len(tasks)} parts", flush=True)

    if not keep_merged:
        shutil.rmtree(data / "tmp" / f"{split}_merged", ignore_errors=True)

    out_path = data / "features" / f"{split}.parquet"
    if combine:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined = _combine_feature_parts(feat_dir, out_path)
        return {
            "rows": combined,
            "columns": len(FEATURE_ORDER) + (3 if _pairs_has_label(data / "pairs" / f"{split}_pairs.parquet") else 2),
            "path": str(out_path),
            "parts": len(tasks),
        }
    return {"rows": rows, "parts": len(tasks), "parts_dir": str(feat_dir)}
