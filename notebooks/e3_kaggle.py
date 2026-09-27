"""E3 runner — measure the e5 ANN candidate-recall ceiling (Kaggle T4).

Re-encodes only the TRAIN entities (S1+S2+S3, ~12.5M) for three fields (name/addr/entity),
builds an IVF-PQ index over Source-2/3 embeddings, queries the held-out Source-1 rows at
K_MAX and writes, per field, the minimum ANN rank at which each held-out truth pair is
recovered. Recall/oracle/cost curves and channel attribution are then computed locally,
where the lexical candidate pairs and ground truth already live.

Outputs (in /kaggle/working):
  e3_minrank_<field>.parquet   s1_id, cand_id, min_rank  (held-out truth pairs recovered by ANN)
  e3_summary.json              K_MAX, n_queries, index size, timings, ann rows per K

Env: E3_FIELDS=name,addr,entity  E3_KMAX=2000  E3_NLIST=4096  E3_M=48  E3_NBITS=8
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def log(*a):
    print("[e3]", *a, flush=True)


def _pip(pkg, module=None):
    module = module or pkg
    try:
        __import__(module)
    except Exception:
        log("pip install", pkg)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


_pip("sentence-transformers")
_pip("faiss-cpu", "faiss")

import duckdb  # noqa: E402
import faiss  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
import torch  # noqa: E402

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIM = 384
PREFIX = "passage: "
MAXLEN = 64
ENCODE_BATCH = 1024
READ_BATCH = 131072
FIELDS = os.environ.get("E3_FIELDS", "name,addr,entity").split(",")
K_MAX = int(os.environ.get("E3_KMAX", "2000"))
NLIST = int(os.environ.get("E3_NLIST", "4096"))
PQ_M = int(os.environ.get("E3_M", "48"))
PQ_NBITS = int(os.environ.get("E3_NBITS", "8"))
NPROBE = int(os.environ.get("E3_NPROBE", "48"))
OUT = "/kaggle/working"
TMP = "/kaggle/temp/e3"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)


def _find_file(name, base="/kaggle/input"):
    for root, _dirs, files in os.walk(base):
        if name in files:
            return os.path.join(root, name)
    return None


TSV = _find_file("train_source1.tsv")
RAW_TRAIN = os.path.dirname(os.path.dirname(TSV)) + "/train" if TSV else None
VALUES = os.path.dirname(_find_file("valfull_s1_ids.parquet") or "")
GT = _find_file("train_ground_truth.tsv")
log("RAW_TRAIN", RAW_TRAIN, "VALUES", VALUES, "GT", GT)


# --- entities (train only) --------------------------------------------------
def build_entities():
    ent_path = os.path.join(TMP, "entities.parquet")
    if os.path.exists(ent_path):
        return ent_path
    raw = [os.path.join(RAW_TRAIN, "train_source%d.tsv" % s) for s in (1, 2, 3)
           if os.path.exists(os.path.join(RAW_TRAIN, "train_source%d.tsv" % s))]
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
        "header=true, union_by_name=true, all_varchar=true) WHERE entity_id IS NOT NULL "
        "GROUP BY entity_id) TO '" + ent_path + "' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    rows = con.execute("SELECT COUNT(*) FROM read_parquet('" + ent_path + "')").fetchone()[0]
    con.close()
    log("built train entities", rows)
    return ent_path


ENT = pq.read_table(build_entities())
N = len(ENT)
COLS = set(ENT.schema.names)
EID = ENT.column("entity_id")
NAME_RAW = ENT.column("name_raw") if "name_raw" in COLS else ENT.column("name_norm")
ADDR_RAW = ENT.column("addr_raw") if "addr_raw" in COLS else ENT.column("addr_norm")
NAME_NORM = ENT.column("name_norm") if "name_norm" in COLS else NAME_RAW
ADDR_NORM = ENT.column("addr_norm") if "addr_norm" in COLS else ADDR_RAW
IS_S1 = pc.starts_with(EID, pattern="S1-").to_numpy(zero_copy_only=False)
N_S1 = int(IS_S1.sum())
log("N", N, "N_S1", N_S1, "N_cand", N - N_S1)


# --- held-out S1 rows + truth rows ------------------------------------------
_vals = pq.read_table(_find_file("valfull_s1_ids.parquet"), columns=["s1_id"])
val_ids = set(_vals.column("s1_id").to_pylist())
eid_list = EID.to_pylist()
row_of = {e: i for i, e in enumerate(eid_list)}
H = np.array([row_of[e] for e in val_ids if e in row_of], dtype=np.int64)
log("held-out S1 with rows", len(H), "of", len(val_ids))

_gt = pd.read_csv(GT, sep="\t", dtype=str, keep_default_na=False,
                  usecols=["source1_entity_id", "matched_entity_ids"])
truth_rows = []
for s1, mids in zip(_gt["source1_entity_id"], _gt["matched_entity_ids"]):
    if not mids or s1 not in val_ids:
        continue
    rs = row_of.get(s1)
    if rs is None:
        continue
    for c in mids.split(","):
        rc = row_of.get(c)
        if rc is not None:
            truth_rows.append((rs, rc, s1, c))
truth = pd.DataFrame(truth_rows, columns=["s1_row", "cand_row", "s1_id", "cand_id"])
del _gt
log("held-out truth pairs (rows mapped)", len(truth))

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")
con.execute("SET temp_directory='" + TMP + "'")
con.register("truth_df", truth)
con.execute("CREATE TEMP TABLE truth AS SELECT s1_row, cand_row, s1_id, cand_id FROM truth_df")


# --- encode -----------------------------------------------------------------
model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
model.max_seq_length = MAXLEN
if torch.cuda.is_available():
    model = model.to("cuda").half()
log("model ready cuda", torch.cuda.is_available())


def field_texts(field):
    for start in range(0, N, READ_BATCH):
        end = min(start + READ_BATCH, N)
        nr = NAME_RAW.slice(start, end - start).to_pylist()
        ar = ADDR_RAW.slice(start, end - start).to_pylist()
        nn = NAME_NORM.slice(start, end - start).to_pylist()
        an = ADDR_NORM.slice(start, end - start).to_pylist()
        if field == "name":
            yield [((nr[i] or nn[i]) or "") for i in range(len(nr))]
        elif field == "addr":
            yield [((ar[i] or an[i]) or "") for i in range(len(ar))]
        elif field == "entity":
            yield [(((nr[i] or nn[i]) or "") + " | " + ((ar[i] or an[i]) or "")) for i in range(len(nr))]
        else:
            raise ValueError(field)


def encode_field(field):
    path = os.path.join(TMP, "emb_%s.f16.npy" % field)
    if os.path.exists(path + ".done"):
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
    json.dump({"field": field, "rows": off}, open(path + ".done", "w"))
    log("encode done", field, off, "elapsed_s", round(time.time() - t0, 1))
    return path


def field_min_rank(field):
    out = os.path.join(OUT, "e3_minrank_%s.parquet" % field)
    if os.path.exists(out):
        log("skip (done)", field)
        return out
    emb_path = encode_field(field)
    emb = np.load(emb_path, mmap_mode="r")
    cand_rows = np.nonzero(~IS_S1)[0]
    queries = np.asarray(emb[H]).astype(np.float32)
    index = faiss.IndexIVFPQ(faiss.IndexFlatIP(DIM), DIM, NLIST, PQ_M, PQ_NBITS,
                             faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    train_rows = cand_rows[::max(1, len(cand_rows) // 1_000_000)][:1_000_000]
    index.train(np.asarray(emb[train_rows]).astype(np.float32))
    log(field, "index trained", round(time.time() - t0, 1))
    for start in range(0, len(cand_rows), 2_000_000):
        block = cand_rows[start:start + 2_000_000]
        index.add(np.asarray(emb[block]).astype(np.float32))
    index.nprobe = NPROBE
    log(field, "index built ntotal", index.ntotal, "elapsed_s", round(time.time() - t0, 1))

    ann_path = os.path.join(TMP, "ann_%s.parquet" % field)
    writer = None
    total = 0
    t1 = time.time()
    QB = 20_000
    for start in range(0, len(H), QB):
        q = queries[start:start + QB]
        _scores, nbrs = index.search(q, K_MAX)
        qrow = H[start:start + QB]
        s1_rep = np.repeat(qrow, K_MAX)
        rank_rep = np.tile(np.arange(K_MAX, dtype=np.int16), len(qrow))
        tbl = pa.table({
            "s1_row": pa.array(s1_rep, pa.int32()),
            "cand_row": pa.array(cand_rows[nbrs].ravel().astype(np.int32)),
            "rank": pa.array(rank_rep),
        })
        if writer is None:
            writer = pq.ParquetWriter(ann_path, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        total += len(tbl)
        if start % (QB * 20) == 0:
            log(field, "ann", min(start + QB, len(H)), "/", len(H),
                "%.0f q/s" % (min(start + QB, len(H)) / max(time.time() - t1, 1e-9)))
    writer.close()
    log(field, "ann rows", total)
    con.execute("CREATE OR REPLACE TEMP TABLE ann AS SELECT s1_row, cand_row, rank "
                "FROM read_parquet('" + ann_path + "')")
    con.execute(
        "COPY (SELECT t.s1_id, t.cand_id, MIN(a.rank) AS min_rank FROM truth t "
        "LEFT JOIN ann a ON a.s1_row = t.s1_row AND a.cand_row = t.cand_row "
        "GROUP BY t.s1_id, t.cand_id) TO '" + out + "' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute("DROP TABLE ann")
    os.remove(ann_path)
    log("DONE", field, "minrank rows", con.execute(
        "SELECT COUNT(*) FROM read_parquet('" + out + "')").fetchone()[0])
    return out


summary = {"K_MAX": K_MAX, "nlist": NLIST, "m": PQ_M, "nbits": PQ_NBITS, "nprobe": NPROBE,
           "n_entities": N, "n_s1": N_S1, "n_candidates": N - N_S1,
           "n_heldout_s1": int(len(H)), "n_truth_pairs": int(len(truth)), "fields": FIELDS}
t_all = time.time()
for field in FIELDS:
    field_min_rank(field)
summary["wall_s"] = round(time.time() - t_all, 1)
json.dump(summary, open(os.path.join(OUT, "e3_summary.json"), "w"), indent=2)
log("E3 COMPLETE", json.dumps(summary))
