"""E1 runner — full 384-dim multilingual-e5 embeddings (Colab T4 or Kaggle T4x2).

Field-at-a-time streaming encode + per-pair cosine so the ~19 GB fp16 memmaps are never
gathered randomly (the D16/F15 failure mode).

Per-pair cosine avoids the known Colab landmines:
- no column named ``row`` (reserved word -> 0-byte parquet, F16-A); the index uses ``row_idx``
- no DuckDB ``ORDER BY`` external sort (temp OOM, F16-B); pairs are argsorted in numpy, so
  candidate memmap rows are read in increasing ``r2`` order (sequential).

Environment auto-detection:
  Kaggle  : raw TSVs in /kaggle/input/amz-er-2026-raw, pairs in /kaggle/input/amz-er-2026-e1-pairs,
            outputs to /kaggle/working, scratch in /kaggle/temp/e1.
  Colab   : entities.parquet + pairs in /content/colab_in, outputs to /content/colab_out,
            scratch in /content/tmp.

Env knobs: E1_FIELDS=name,addr,entity  E1_DROP_EMB=1  E1_MAX_ENTITIES=<n> (smoke test)
"""
import json
import os
import subprocess
import sys
import time
import zipfile


def log(*a):
    print("[e1]", *a, flush=True)


def _pip(pkg):
    try:
        __import__(pkg)
    except Exception:
        log("pip install", pkg)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


_pip("sentence_transformers")

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
import torch  # noqa: E402

ON_KAGGLE = os.path.isdir("/kaggle")


def _find_file(name, base="/kaggle/input"):
    if not os.path.isdir(base):
        return None
    for root, _dirs, files in os.walk(base):
        if name in files:
            return os.path.join(root, name)
    return None


if ON_KAGGLE:
    _pairs = _find_file("pairs_train.parquet")
    PAIRS_DIR = os.path.dirname(_pairs) if _pairs else "/kaggle/input/amz-er-2026-e1-pairs"
    OUT = "/kaggle/working"
    TMP = "/kaggle/temp/e1"
else:
    PAIRS_DIR = "/content/colab_in"
    OUT = "/content/colab_out"
    TMP = "/content/tmp"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIM = 384
PREFIX = "passage: "
MAXLEN = 64
ENCODE_BATCH = 1024
READ_BATCH = 131072
COS_CHUNK = 2_000_000
FIELDS = os.environ.get("E1_FIELDS", "name,addr,entity").split(",")
MAX_ENTITIES = int(os.environ.get("E1_MAX_ENTITIES", "0"))
SPLITS = [s for s in ("train", "valfull") if os.path.exists(os.path.join(PAIRS_DIR, "pairs_%s.parquet" % s))]
EXPECT_PAIRS = {"train": 30494378, "valfull": 60912676}
EXPECT_ENTITIES = 24229173


# --- entities ---------------------------------------------------------------
def _raw_tsvs():
    found = []
    for split in ("train", "test"):
        for source in (1, 2, 3):
            path = _find_file("%s_source%d.tsv" % (split, source))
            if path:
                found.append(path)
    return found


def build_entities_from_tsv():
    ent_path = os.path.join(TMP, "entities.parquet")
    if os.path.exists(ent_path):
        return ent_path
    raw = _raw_tsvs()
    if not raw:
        raise SystemExit("no raw TSVs found under /kaggle/input")
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory='" + TMP + "'")
    con.execute("PRAGMA max_temp_directory_size='40GiB'")
    files = "[" + ",".join("'" + f + "'" for f in raw) + "]"
    con.execute(
        "COPY (SELECT entity_id, any_value(business_name) AS name_raw, "
        "any_value(business_address) AS addr_raw FROM read_csv(" + files + ", delim='\\t', "
        "header=true, union_by_name=true, all_varchar=true) "
        "WHERE entity_id IS NOT NULL GROUP BY entity_id) TO '" + ent_path + "' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    rows = con.execute("SELECT COUNT(*) FROM read_parquet('" + ent_path + "')").fetchone()[0]
    con.close()
    log("built entities from TSVs:", rows)
    return ent_path


def resolve_entities():
    if ON_KAGGLE:
        try:
            log("kaggle/input:", sorted(os.listdir("/kaggle/input")))
        except Exception:
            pass
    log("PAIRS_DIR", PAIRS_DIR, "raw_tsvs", len(_raw_tsvs()) if ON_KAGGLE else 0)
    for cand in (os.path.join(TMP, "entities.parquet"), os.path.join(PAIRS_DIR, "entities.parquet")):
        if os.path.exists(cand):
            return cand
    if ON_KAGGLE and _raw_tsvs():
        return build_entities_from_tsv()
    raise SystemExit("no entities source found (PAIRS_DIR=%s)" % PAIRS_DIR)


ENTITIES = resolve_entities()
ENT = pq.read_table(ENTITIES)
if MAX_ENTITIES:
    ENT = ENT.slice(0, MAX_ENTITIES)
N = len(ENT)
COLS = set(ENT.schema.names)
EID_COL = ENT.column("entity_id")
NAME_RAW = ENT.column("name_raw") if "name_raw" in COLS else ENT.column("name_norm")
ADDR_RAW = ENT.column("addr_raw") if "addr_raw" in COLS else ENT.column("addr_norm")
NAME_NORM = ENT.column("name_norm") if "name_norm" in COLS else NAME_RAW
ADDR_NORM = ENT.column("addr_norm") if "addr_norm" in COLS else ADDR_RAW
log("entities", N, "expected", EXPECT_ENTITIES, "pairs dir", PAIRS_DIR, "splits", SPLITS, "fields", FIELDS)


# --- entity index (row_idx, never "row") ------------------------------------
INDEX = os.path.join(TMP, "entity_index.parquet")
if not os.path.exists(INDEX):
    tbl = pa.table({"entity_id": EID_COL, "row_idx": pa.array(np.arange(N, dtype=np.int64))})
    pq.write_table(tbl, INDEX, compression="zstd")
    log("index built rows", N)
_s1 = pc.starts_with(EID_COL, pattern="S1-").to_numpy(zero_copy_only=False)
N_S1 = int(_s1.sum())
s1_pos = np.full(N, -1, dtype=np.int32)
s1_pos[np.nonzero(_s1)[0]] = np.arange(N_S1, dtype=np.int32)
log("N_S1", N_S1)

for split in SPLITS:
    p = os.path.join(PAIRS_DIR, "pairs_%s.parquet" % split)
    got = pq.ParquetFile(p).metadata.num_rows
    log("pairs", split, got, "OK" if got == EXPECT_PAIRS[split] else "MISMATCH %s" % EXPECT_PAIRS[split])


# --- encode / cosine --------------------------------------------------------
t_start = time.time()
model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
model.max_seq_length = MAXLEN
if torch.cuda.is_available():
    model = model.to("cuda").half()
log("model ready cuda", torch.cuda.is_available(), "device", str(model.device))


def field_texts(field):
    for start in range(0, N, READ_BATCH):
        end = min(start + READ_BATCH, N)
        n_raw = NAME_RAW.slice(start, end - start).to_pylist()
        a_raw = ADDR_RAW.slice(start, end - start).to_pylist()
        n_norm = NAME_NORM.slice(start, end - start).to_pylist()
        a_norm = ADDR_NORM.slice(start, end - start).to_pylist()
        if field == "name":
            yield [((n_raw[i] or n_norm[i]) or "") for i in range(len(n_raw))]
        elif field == "addr":
            yield [((a_raw[i] or a_norm[i]) or "") for i in range(len(a_raw))]
        elif field == "entity":
            yield [(((n_raw[i] or n_norm[i]) or "") + " | " + ((a_raw[i] or a_norm[i]) or ""))
                   for i in range(len(n_raw))]
        else:
            raise ValueError(field)


def encode_field(field):
    path = os.path.join(TMP, "emb_%s.f16.npy" % field)
    if os.path.exists(path + ".done"):
        log("encode skip (done)", field)
        return path
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(N, DIM))
    off = 0
    t0 = time.time()
    for texts in field_texts(field):
        vecs = model.encode([PREFIX + t for t in texts], batch_size=ENCODE_BATCH,
                            normalize_embeddings=True, convert_to_numpy=True)
        k = len(texts)
        mm[off:off + k] = vecs.astype(np.float16)
        off += k
        if off % (READ_BATCH * 20) < READ_BATCH:
            mm.flush()
            log(field, off, "/", N, "%.0f/s" % (off / max(time.time() - t0, 1e-9)))
    mm.flush()
    json.dump({"field": field, "rows": off, "dim": DIM, "prefix": PREFIX,
               "model": MODEL_ID, "revision": MODEL_REVISION}, open(path + ".done", "w"))
    log("encode done", field, off, "elapsed_s", round(time.time() - t0, 1))
    return path


def load_s1_embeddings(emb_path):
    mm = np.load(emb_path, mmap_mode="r")
    out = np.empty((N_S1, DIM), dtype=np.float16)
    k = 0
    CH = 2_000_000
    for start in range(0, N, CH):
        block = np.asarray(mm[start:start + CH])
        m = _s1[start:start + CH]
        kk = int(m.sum())
        if kk:
            out[k:k + kk] = block[m]
            k += kk
        del block
    assert k == N_S1, (k, N_S1)
    return out


def _configure(con):
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory='" + TMP + "'")
    con.execute("PRAGMA max_temp_directory_size='40GiB'")


def cosine_split(split, emb_path, s1_emb, field, out_dir):
    out = os.path.join(out_dir, "cos_%s_%s.parquet" % (split, field))
    if os.path.exists(out):
        log("cosine skip (done)", split, field)
        return out
    pairs_path = os.path.join(PAIRS_DIR, "pairs_%s.parquet" % split)
    con = duckdb.connect()
    _configure(con)
    con.execute("CREATE VIEW ei AS SELECT * FROM read_parquet('" + INDEX + "')")
    con.execute("CREATE VIEW pr AS SELECT * FROM read_parquet('" + pairs_path + "')")
    con.execute(
        "CREATE TEMP TABLE pmap AS "
        "SELECT row_number() OVER () AS pair_idx, p.s1_id, p.cand_id, "
        "e1.row_idx AS r1, e2.row_idx AS r2 "
        "FROM pr p JOIN ei e1 ON e1.entity_id = p.s1_id "
        "JOIN ei e2 ON e2.entity_id = p.cand_id"
    )
    n_pairs = con.execute("SELECT COUNT(*) FROM pmap").fetchone()[0]
    log(split, field, "pairs", n_pairs)
    arr = con.execute("SELECT r1, r2 FROM pmap").fetchnumpy()
    r1 = arr["r1"].astype(np.int32)
    r2 = arr["r2"].astype(np.int32)
    del arr
    order = np.argsort(r2, kind="stable").astype(np.int64)

    mm = np.load(emb_path, mmap_mode="r")
    cos = np.empty(len(r2), dtype=np.float16)
    t0 = time.time()
    for start in range(0, len(order), COS_CHUNK):
        sel = order[start:start + COS_CHUNK]
        a = s1_emb[s1_pos[r1[sel]]].astype(np.float32)
        b = np.asarray(mm[r2[sel]]).astype(np.float32)
        cos[sel] = (a * b).sum(1).astype(np.float16)
        del a, b
        if start % (COS_CHUNK * 10) == 0:
            done = min(start + COS_CHUNK, len(order))
            log(split, field, done, "/", len(order), "%.0f/s" % (done / max(time.time() - t0, 1e-9)))
    del order, r1, r2

    con.register("cosdf", pd.DataFrame({"pair_idx": np.arange(len(cos), dtype=np.int64),
                                        field + "_e5_cos": cos}))
    tmp = out + ".tmp"
    col = field + "_e5_cos"
    con.execute(
        "COPY (SELECT p.s1_id, p.cand_id, c." + col + " AS " + col + " "
        "FROM pmap p JOIN cosdf c ON c.pair_idx = p.pair_idx) "
        "TO '" + tmp + "' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    os.replace(tmp, out)
    con.close()
    log("DONE cosine", split, field, len(cos))
    return out


COS_DIR = os.path.join(TMP, "cos_parts")
os.makedirs(COS_DIR, exist_ok=True)
for field in FIELDS:
    emb_path = encode_field(field)
    s1_emb = load_s1_embeddings(emb_path)
    for split in SPLITS:
        cosine_split(split, emb_path, s1_emb, field, COS_DIR)
    del s1_emb
    if os.environ.get("E1_DROP_EMB") == "1":
        os.remove(emb_path)
    log("field complete", field, "elapsed_s", round(time.time() - t_start, 1))

_ALIAS = {"name": "n", "addr": "a", "entity": "e"}
for split in SPLITS:
    avail = [f for f in ("name", "addr", "entity")
             if os.path.exists(os.path.join(COS_DIR, "cos_%s_%s.parquet" % (split, f)))]
    if not avail:
        log(split, "no cosine parts, skipping merge")
        continue
    base = avail[0]
    con = duckdb.connect()
    _configure(con)
    for f in avail:
        con.execute("CREATE TEMP TABLE " + f + " AS SELECT * FROM read_parquet('" +
                    os.path.join(COS_DIR, "cos_%s_%s.parquet" % (split, f)) + "')")
    select = ["%s.s1_id AS s1_id, %s.cand_id AS cand_id" % (_ALIAS[base], _ALIAS[base])]
    for f in avail:
        select.append("%s.%s_e5_cos AS %s_e5_cos" % (_ALIAS[f], f, f))
    joins = "".join(
        " LEFT JOIN " + f + " " + _ALIAS[f] + " ON " + _ALIAS[f] + ".s1_id = " + _ALIAS[base] +
        ".s1_id AND " + _ALIAS[f] + ".cand_id = " + _ALIAS[base] + ".cand_id"
        for f in avail[1:]
    )
    out = os.path.join(OUT, "cosine_%s.parquet" % split)
    con.execute(
        "COPY (SELECT " + ", ".join(select) + " FROM " + base + " " + _ALIAS[base] + joins + ") "
        "TO '" + out + "' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    rows = con.execute("SELECT COUNT(*) FROM read_parquet('" + out + "')").fetchone()[0]
    log(split, "merged rows", rows, "fields", avail)
    con.close()

zip_path = os.path.join(OUT, "cosine_e5_out.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for split in SPLITS:
        zf.write(os.path.join(OUT, "cosine_%s.parquet" % split), "cosine_%s.parquet" % split)
log("E1 COMPLETE elapsed_s", round(time.time() - t_start, 1),
    "zip_MB", round(os.path.getsize(zip_path) / 1e6, 1))
