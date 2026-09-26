"""E1 Colab runner — full 384-dim multilingual-e5 embeddings.

Runs a field-at-a-time streaming encode + per-pair cosine on a Colab T4 so the
18 GB fp16 memmaps are never gathered randomly (the D16 failure mode).

Inputs  : /content/colab_in/{entities.parquet,pairs_train.parquet,pairs_valfull.parquet}
Outputs : /content/colab_out/cosine_{split}.parquet and /content/cosine_e5_out.zip
State   : /content/tmp/entity_index.parquet, /content/tmp/emb_<field>.f16.npy(.done)
          /content/tmp/cos_parts/...  (all steps resumable via .done / existing files)

Env knobs: E1_FIELDS=name,addr,entity  E1_DROP_EMB=1 (delete memmap after each field)
"""
import json
import os
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch

IN = "/content/colab_in"
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
FIELDS = os.environ.get("E1_FIELDS", "name,addr,entity").split(",")
SPLITS = [s for s in ("train", "valfull") if os.path.exists(os.path.join(IN, "pairs_%s.parquet" % s))]

EXPECT = {"entities.parquet": 24229173, "pairs_train.parquet": 30494378, "pairs_valfull.parquet": 60912676}
for f, n in EXPECT.items():
    p = os.path.join(IN, f)
    if not os.path.exists(p):
        raise SystemExit("missing " + p)
    got = pq.ParquetFile(p).metadata.num_rows
    print(f, got, "OK" if got == n else "MISMATCH expected %s" % n, flush=True)
    if got != n:
        raise SystemExit("row mismatch " + f)


def log(*a):
    print("[e1]", *a, flush=True)


t_start = time.time()
model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
model.max_seq_length = MAXLEN
if torch.cuda.is_available():
    model = model.to("cuda").half()
log("model ready cuda", torch.cuda.is_available(), "fields", FIELDS, "splits", SPLITS)

# --- entity index -----------------------------------------------------------
INDEX = os.path.join(TMP, "entity_index.parquet")
if not os.path.exists(INDEX):
    pf = pq.ParquetFile(os.path.join(IN, "entities.parquet"))
    w = None
    rows = 0
    for batch in pf.iter_batches(batch_size=1_000_000, columns=["entity_id"]):
        eids = batch.column("entity_id").to_pylist()
        tbl = pa.table({"entity_id": eids, "row": pa.array(range(rows, rows + len(eids)), pa.int64())})
        if w is None:
            w = pq.ParquetWriter(INDEX, tbl.schema, compression="zstd")
        w.write_table(tbl)
        rows += len(eids)
    w.close()
    log("index built rows", rows)

_idx = pq.read_table(INDEX, columns=["entity_id"])
N = len(_idx)
_eids = _idx.column("entity_id")
_s1mask = pc.starts_with(_eids, pattern="S1-").to_numpy(zero_copy_only=False)
_s1_rows = _idx.column("row").to_numpy()[_s1mask]
N_S1 = int(_s1mask.sum())
s1_pos = np.full(N, -1, dtype=np.int32)
s1_pos[_s1_rows] = np.arange(N_S1, dtype=np.int32)
log("N", N, "N_S1", N_S1)
del _idx, _eids


# --- helpers ----------------------------------------------------------------
def field_texts(field):
    pf = pq.ParquetFile(os.path.join(IN, "entities.parquet"))
    cols = ["name_raw", "addr_raw", "name_norm", "addr_norm"]
    for batch in pf.iter_batches(batch_size=READ_BATCH, columns=cols):
        d = batch.to_pydict()
        nr, ar, nn, an = d["name_raw"], d["addr_raw"], d["name_norm"], d["addr_norm"]
        if field == "name":
            yield [(nr[i] or nn[i]) for i in range(len(nr))]
        elif field == "addr":
            yield [(ar[i] or an[i]) for i in range(len(ar))]
        elif field == "entity":
            yield [((nr[i] or nn[i]) + " | " + (ar[i] or an[i])) for i in range(len(nr))]
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
    log("encode done", field, off)
    return path


def load_s1_embeddings(emb_path):
    mm = np.load(emb_path, mmap_mode="r")
    out = np.empty((N_S1, DIM), dtype=np.float16)
    k = 0
    CH = 2_000_000
    for start in range(0, N, CH):
        block = np.asarray(mm[start:start + CH])
        m = _s1mask[start:start + CH]
        kk = int(m.sum())
        if kk:
            out[k:k + kk] = block[m]
            k += kk
        del block
    assert k == N_S1, (k, N_S1)
    return out


def _configure(con):
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory='" + TMP + "'")
    con.execute("PRAGMA max_temp_directory_size='40GiB'")


def cosine_split(split, emb_path, s1_emb, field, out_dir):
    out = os.path.join(out_dir, "cos_%s_%s.parquet" % (split, field))
    if os.path.exists(out):
        log("cosine skip (done)", split, field)
        return out
    pairs_path = os.path.join(IN, "pairs_%s.parquet" % split)
    con = duckdb.connect()
    _configure(con)
    con.execute("CREATE VIEW ei AS SELECT * FROM read_parquet('" + INDEX + "')")
    con.execute("CREATE VIEW pr AS SELECT * FROM read_parquet('" + pairs_path + "')")
    reader = con.execute(
        "SELECT p.s1_id, p.cand_id, e1.row AS r1, e2.row AS r2 "
        "FROM pr p JOIN ei e1 ON e1.entity_id = p.s1_id "
        "JOIN ei e2 ON e2.entity_id = p.cand_id ORDER BY e2.row"
    ).fetch_record_batch(500_000)
    mm = np.load(emb_path, mmap_mode="r")
    tmp = out + ".tmp"
    writer = None
    total = 0
    t0 = time.time()
    for rb in reader:
        r1 = rb.column("r1").to_numpy()
        r2 = rb.column("r2").to_numpy()
        a = s1_emb[s1_pos[r1]].astype(np.float32)
        b = np.asarray(mm[r2]).astype(np.float32)
        cos = (a * b).sum(1).astype(np.float16)
        tbl = pa.table({"s1_id": rb.column("s1_id"), "cand_id": rb.column("cand_id"),
                        field + "_e5_cos": pa.array(cos)})
        if writer is None:
            writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        total += len(tbl)
        if total % 5_000_000 < 500_000:
            log(split, field, total, "%.0f/s" % (total / max(time.time() - t0, 1e-9)))
    if writer is not None:
        writer.close()
    os.replace(tmp, out)
    con.close()
    log("DONE cosine", split, field, total)
    return out


# --- main loop --------------------------------------------------------------
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

# --- merge + package --------------------------------------------------------
for split in SPLITS:
    con = duckdb.connect()
    _configure(con)
    con.execute("CREATE TEMP TABLE name AS SELECT * FROM read_parquet('" + os.path.join(COS_DIR, "cos_%s_name.parquet" % split) + "')")
    con.execute("CREATE TEMP TABLE addr AS SELECT * FROM read_parquet('" + os.path.join(COS_DIR, "cos_%s_addr.parquet" % split) + "')")
    con.execute("CREATE TEMP TABLE ent AS SELECT * FROM read_parquet('" + os.path.join(COS_DIR, "cos_%s_entity.parquet" % split) + "')")
    out = os.path.join(OUT, "cosine_%s.parquet" % split)
    con.execute(
        "COPY (SELECT n.s1_id, n.cand_id, n.name_e5_cos, a.addr_e5_cos, e.entity_e5_cos "
        "FROM name n JOIN addr a ON a.s1_id = n.s1_id AND a.cand_id = n.cand_id "
        "JOIN ent e ON e.s1_id = n.s1_id AND e.cand_id = n.cand_id) "
        "TO '" + out + "' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    rows = con.execute("SELECT COUNT(*) FROM read_parquet('" + out + "')").fetchone()[0]
    log(split, "merged rows", rows)
    con.close()

import shutil
shutil.make_archive("/content/cosine_e5_out", "zip", OUT)
log("E1 COMPLETE elapsed_s", round(time.time() - t_start, 1),
    "zip_MB", round(os.path.getsize("/content/cosine_e5_out.zip") / 1e6, 1))
