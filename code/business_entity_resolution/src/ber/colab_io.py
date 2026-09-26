import os
from pathlib import Path

import duckdb
import pandas as pd

ENTITY_COLUMNS = ["entity_id", "name_norm", "name_roman", "addr_norm"]
SPLITS = ("train", "valfull", "test")
COSINE_COLUMNS = ("emb_name_cos", "emb_addr_cos")
EMB_FEATURES = ("emb_name_cos", "emb_addr_cos")
EMB_MISSING = -1.0


def _configure(con, data):
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(Path(data) / 'tmp').as_posix()}'")
    con.execute("PRAGMA max_temp_directory_size='50GiB'")


def export_entities(cfg):
    data = Path(cfg.data_dir)
    out = data / "colab_in" / "entities.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [
        (data / "processed" / f"{split}_source{source}.parquet").as_posix()
        for split in ("train", "test")
        for source in (1, 2, 3)
        if (data / "processed" / f"{split}_source{source}.parquet").exists()
    ]
    if not files:
        raise FileNotFoundError("no processed source parquet files to export")
    file_sql = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    _configure(con, data)
    con.execute(
        f"""
        COPY (
            SELECT entity_id, name_norm, name_roman, addr_norm FROM (
                SELECT entity_id, name_norm, name_roman, addr_norm,
                       row_number() OVER (PARTITION BY entity_id
                           ORDER BY name_norm, name_roman, addr_norm) AS rn
                FROM read_parquet({file_sql})
                WHERE entity_id IS NOT NULL
            ) WHERE rn = 1
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    con.close()
    return out, int(rows)


def export_pairs(cfg, split):
    data = Path(cfg.data_dir)
    src = data / "pairs" / f"{split}_pairs.parquet"
    if not src.exists():
        raise FileNotFoundError(src)
    out = data / "colab_in" / f"pairs_{split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    _configure(con, data)
    con.execute(
        f"COPY (SELECT s1_id, cand_id FROM read_parquet('{src.as_posix()}')) "
        f"TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    con.close()
    return out, int(rows)


def export_for_colab(cfg, splits=SPLITS):
    entities_path, entity_rows = export_entities(cfg)
    pairs = {}
    for split in splits:
        pairs_path, _ = export_pairs(cfg, split)
        pairs[split] = str(pairs_path)
    return {"entities": str(entities_path), "entity_rows": entity_rows, "pairs": pairs}


def import_cosine(cfg, split, columns=COSINE_COLUMNS):
    path = Path(cfg.data_dir) / "colab_out" / f"cosine_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return frame[["s1_id", "cand_id", *columns]]


def _feature_targets(cfg, split):
    data = Path(cfg.data_dir)
    combined = data / "features" / f"{split}.parquet"
    if combined.exists():
        return [combined]
    parts_dir = data / "tmp" / f"{split}_features"
    if parts_dir.exists():
        parts = sorted(parts_dir.glob("*.parquet"))
        if parts:
            return parts
    raise FileNotFoundError(f"no feature parts or combined features for split={split}")


def merge_cosine(cfg, split, columns=EMB_FEATURES):
    data = Path(cfg.data_dir)
    cos_path = data / "colab_out" / f"cosine_{split}.parquet"
    if not cos_path.exists():
        raise FileNotFoundError(cos_path)
    targets = _feature_targets(cfg, split)
    con = duckdb.connect()
    _configure(con, data)
    con.execute(
        f"CREATE TEMP TABLE cos AS SELECT s1_id, cand_id, "
        f"{', '.join(columns)} FROM read_parquet('{cos_path.as_posix()}')"
    )
    existing = set()
    import pyarrow.parquet as pq

    for target in targets:
        existing.update(pq.ParquetFile(target).schema_arrow.names)
    drop = [c for c in columns if c in existing]
    select_features = f"f.* EXCLUDE ({', '.join(drop)})" if drop else "f.*"
    new_cols = ", ".join(f"COALESCE(c.{c}, {EMB_MISSING}) AS {c}" for c in columns)
    rows = 0
    for target in targets:
        tmp = target.with_suffix(".merged.parquet")
        con.execute(
            f"""
            COPY (
                SELECT {select_features}, {new_cols}
                FROM read_parquet('{target.as_posix()}') f
                LEFT JOIN cos c ON c.s1_id = f.s1_id AND c.cand_id = f.cand_id
            ) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        rows += con.execute(f"SELECT COUNT(*) FROM read_parquet('{tmp.as_posix()}')").fetchone()[0]
        os.replace(tmp, target)
    con.close()
    return {"split": split, "rows": int(rows), "parts": len(targets), "columns": list(columns)}
