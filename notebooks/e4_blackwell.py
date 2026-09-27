"""E4 Blackwell runner — production multi-channel candidates + e5 cosines (memory-safe).

Streams ANN results to parquet per (field, split) instead of concatenating ~2.6B rows in RAM.
Embeddings stay resident in VRAM (~56 GB of 96 GB) for the cosine pass.

Outputs (/kaggle/working): candidates_{train,test}.parquet, cosine_{train,test}.parquet, e4_summary.json
"""
import glob
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
    print("[e4]", *a, flush=True)


_wheels = [p for p in glob.glob("/kaggle/input/**/*.whl", recursive=True)
           if "faiss" in os.path.basename(p).lower()]
WHEEL_DIR = os.path.dirname(_wheels[0]) if _wheels else None
_safet = [p for p in glob.glob("/kaggle/input/**/model.safetensors", recursive=True)
          if "multilingual-e5-small" in p]
MODEL_LOCAL = os.path.dirname(_safet[0]) if _safet else None
log("wheel_dir", WHEEL_DIR, "model", MODEL_LOCAL)
try:
    import faiss  # noqa: F401
except Exception:
    if WHEEL_DIR:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                        "--find-links", WHEEL_DIR, "faiss-cpu"], check=False)

import duckdb  # noqa: E402
import faiss  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
import torch  # noqa: E402

OUT = "/kaggle/working"
TMP = "/kaggle/temp/e4"
ANN_DIR = os.path.join(TMP, "ann")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIM = 384
PREFIX = "passage: "
MAXLEN = 64
K = int(os.environ.get("E4_K", "100"))
CAP = int(os.environ.get("E4_CAP", "250"))
NLIST = 4096
PQ_M = 48
PQ_NBITS = 8
NPROBE = 32
FIELDS = ["name", "addr", "entity"]
PASS = {"name": 11, "addr": 12, "entity": 13}
EXPECT_N = 24229173

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
if "RTX PRO 6000" not in gpu.upper():
    raise SystemExit("needs interactive RTX Pro 6000, got %s" % gpu)
log("gpu", gpu, "vram_GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "K", K, "CAP", CAP)


def _find_file(name, base="/kaggle/input"):
    for root, _dirs, files in os.walk(base):
        if name in files:
            return os.path.join(root, name)
    return None


TSVS = []
for split in ("train", "test"):
    for source in (1, 2, 3):
        p = _find_file("%s_source%d.tsv" % (split, source))
        if p:
            TSVS.append((split, p))
CAND_TRAIN = _find_file("train_candidates.parquet")
CAND_TEST = _find_file("test_candidates.parquet")
log("tsvs", len(TSVS), "cand_train", CAND_TRAIN, "cand_test", CAND_TEST)
if not TSVS or CAND_TRAIN is None or CAND_TEST is None or MODEL_LOCAL is None:
    raise SystemExit("missing inputs: attach amz-er-2026-all AND amz-er-2026-candidates")

ENT_PATH = os.path.join(TMP, "entities.parquet")
if not os.path.exists(ENT_PATH) or pq.ParquetFile(ENT_PATH).metadata.num_rows != EXPECT_N:
    if os.path.exists(ENT_PATH):
        os.remove(ENT_PATH)
    con = duckdb.connect()
    con.execute("SET memory_limit='48GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory='" + TMP + "'")
    selects = []
    for split, path in TSVS:
        selects.append(
            "SELECT entity_id, any_value(business_name) AS name_raw, "
            "any_value(business_address) AS addr_raw, '%s' AS split "
            "FROM read_csv('%s', delim='\\t', header=true, union_by_name=true, all_varchar=true) "
            "WHERE entity_id IS NOT NULL GROUP BY entity_id" % (split, path)
        )
    con.execute("COPY (" + " UNION ALL ".join(selects) + ") TO '" + ENT_PATH + "' (FORMAT PARQUET, COMPRESSION ZSTD)")
    n = con.execute("SELECT COUNT(*) FROM read_parquet('" + ENT_PATH + "')").fetchone()[0]
    con.close()
    log("entities built", n)
    if n != EXPECT_N:
        raise SystemExit("entities %d != %d" % (n, EXPECT_N))

ENT = pq.read_table(ENT_PATH)
N = len(ENT)
COLS = set(ENT.schema.names)
EID = ENT.column("entity_id")
EID_LIST = EID.to_pylist()
NAME_RAW = ENT.column("name_raw") if "name_raw" in COLS else ENT.column("name_norm")
ADDR_RAW = ENT.column("addr_raw") if "addr_raw" in COLS else ENT.column("addr_norm")
NAME_NORM = ENT.column("name_norm") if "name_norm" in COLS else NAME_RAW
ADDR_NORM = ENT.column("addr_norm") if "addr_norm" in COLS else ADDR_RAW
IS_S1 = pc.starts_with(EID, pattern="S1-").to_numpy(zero_copy_only=False)
SPLIT = np.array(ENT.column("split").to_pylist())
row_of = {e: i for i, e in enumerate(EID_LIST)}
eid_arr = np.array(EID_LIST, dtype=object)
log("N", N, "N_s1", int(IS_S1.sum()))

model = SentenceTransformer(MODEL_LOCAL)
model.max_seq_length = MAXLEN
model = model.to("cuda").half()
log("model ready")


def field_texts(field):
    CH = 131072
    for start in range(0, N, CH):
        end = min(start + CH, N)
        nr = NAME_RAW.slice(start, end - start).to_pylist()
        ar = ADDR_RAW.slice(start, end - start).to_pylist()
        nn = NAME_NORM.slice(start, end - start).to_pylist()
        an = ADDR_NORM.slice(start, end - start).to_pylist()
        if field == "name":
            yield [((nr[i] or nn[i]) or "") for i in range(len(nr))]
        elif field == "addr":
            yield [((ar[i] or an[i]) or "") for i in range(len(ar))]
        else:
            yield [(((nr[i] or nn[i]) or "") + " | " + ((ar[i] or an[i]) or "")) for i in range(len(nr))]


def encode_field_gpu(field):
    buf = np.empty((N, DIM), dtype=np.float16)
    off = 0
    t0 = time.time()
    for texts in field_texts(field):
        vecs = model.encode([PREFIX + t for t in texts], batch_size=2048,
                            normalize_embeddings=True, convert_to_numpy=True)
        k = len(texts)
        buf[off:off + k] = vecs.astype(np.float16)
        off += k
        if off % (131072 * 20) < 131072:
            log(field, off, "/", N, "%.0f/s" % (off / max(time.time() - t0, 1e-9)))
    log("encode done", field, off, "elapsed_s", round(time.time() - t0, 1))
    return torch.from_numpy(buf).cuda()


embs = {}
for field in FIELDS:
    embs[field] = encode_field_gpu(field)
    t0 = time.time()
    for split in ("train", "test"):
        ann_path = os.path.join(ANN_DIR, "ann_%s_%s.parquet" % (field, split))
        if os.path.exists(ann_path):
            continue
        cand_rows = np.nonzero((SPLIT == split) & ~IS_S1)[0]
        q_rows = np.nonzero((SPLIT == split) & IS_S1)[0]
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(DIM), DIM, NLIST, PQ_M, PQ_NBITS,
                                 faiss.METRIC_INNER_PRODUCT)
        sample = cand_rows[::max(1, len(cand_rows) // 1_000_000)][:1_000_000]
        index.train(embs[field][sample].float().cpu().numpy())
        for start in range(0, len(cand_rows), 2_000_000):
            block = cand_rows[start:start + 2_000_000]
            index.add(embs[field][block].float().cpu().numpy())
        index.nprobe = NPROBE
        log(field, split, "index ntotal", index.ntotal, "elapsed_s", round(time.time() - t0, 1))
        writer = None
        for start in range(0, len(q_rows), 50_000):
            qr = q_rows[start:start + 50_000]
            _s, nbrs = index.search(embs[field][qr].float().cpu().numpy(), K)
            valid = (nbrs >= 0).ravel()
            s1 = eid_arr[np.repeat(qr, K)[valid]]
            cand = eid_arr[cand_rows[np.where(nbrs >= 0, nbrs, 0)].ravel()[valid]]
            tbl = pa.table({
                "s1_id": pa.array(s1.tolist()),
                "cand_id": pa.array(cand.tolist()),
                "split": pa.array([split] * int(valid.sum())),
                "pass_id": pa.array(np.full(int(valid.sum()), PASS[field], dtype=np.int16)),
                "rank": pa.array(np.tile(np.arange(K, dtype=np.int16), len(qr))[valid]),
                "score": pa.array(_s.ravel()[valid].astype(np.float16)),
            })
            if writer is None:
                writer = pq.ParquetWriter(ann_path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            del s1, cand, tbl
        if writer is not None:
            writer.close()
        del index
        torch.cuda.empty_cache()
    log("field ANN done", field)

ANN_GLOB = (ANN_DIR + "/*.parquet")
con = duckdb.connect()
con.execute("SET memory_limit='48GB'")
con.execute("SET threads=8")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='" + TMP + "'")

summary = {"K": K, "CAP": CAP, "fields": FIELDS}
for split, cand_path in (("train", CAND_TRAIN), ("test", CAND_TEST)):
    con.execute("CREATE OR REPLACE TEMP TABLE lex AS SELECT s1_id, cand_id, pass_id, block_score, is_s2 "
                "FROM read_parquet('" + cand_path + "')")
    con.execute(
        "CREATE OR REPLACE TEMP TABLE annc AS SELECT s1_id, cand_id, MIN(pass_id) AS ann_pass, "
        "MIN(rank) AS ret_rank, MAX(score) AS ret_score, COUNT(DISTINCT pass_id) AS ret_channel_count "
        "FROM read_parquet('" + ANN_GLOB + "') WHERE split = '" + split + "' GROUP BY s1_id, cand_id")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE merged AS
        SELECT COALESCE(l.s1_id, a.s1_id) AS s1_id, COALESCE(l.cand_id, a.cand_id) AS cand_id,
               COALESCE(l.pass_id, a.ann_pass) AS pass_id,
               COALESCE(l.block_score, a.ret_score) AS block_score,
               COALESCE(l.is_s2, starts_with(COALESCE(l.cand_id, a.cand_id), 'S2-')) AS is_s2,
               COALESCE(a.ret_rank, -1) AS ret_rank,
               COALESCE(a.ret_score, 0) AS ret_score,
               COALESCE(a.ret_channel_count, 0) AS ret_channel_count,
               CASE WHEN l.s1_id IS NOT NULL THEN 1 ELSE 0 END AS has_lex
        FROM lex l FULL OUTER JOIN annc a ON a.s1_id = l.s1_id AND a.cand_id = l.cand_id
        """)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE capped AS SELECT * FROM ("
        "SELECT *, ROW_NUMBER() OVER (PARTITION BY s1_id "
        "ORDER BY has_lex DESC, ret_score DESC, pass_id ASC, cand_id ASC) AS rn FROM merged) "
        "WHERE rn <= " + str(CAP))
    n = con.execute("SELECT COUNT(*) FROM capped").fetchone()[0]
    out = os.path.join(OUT, "candidates_%s.parquet" % split)
    con.execute(
        "COPY (SELECT s1_id, cand_id, pass_id, block_score, is_s2, ret_rank, ret_score, ret_channel_count "
        "FROM capped) TO '" + out + "' (FORMAT PARQUET, COMPRESSION ZSTD)")
    summary[split + "_candidates"] = int(n)
    log(split, "candidates", n)
    # free the ANN parquets for this split's field data? keep for other split
con.close()

for split in ("train", "test"):
    out = os.path.join(OUT, "cosine_%s.parquet" % split)
    if os.path.exists(out):
        continue
    cands = pd.read_parquet(os.path.join(OUT, "candidates_%s.parquet" % split), columns=["s1_id", "cand_id"])
    r1 = np.fromiter((row_of.get(x, -1) for x in cands["s1_id"]), dtype=np.int64, count=len(cands))
    r2 = np.fromiter((row_of.get(x, -1) for x in cands["cand_id"]), dtype=np.int64, count=len(cands))
    cols = {}
    for field in FIELDS:
        cos = np.empty(len(cands), dtype=np.float16)
        B = 4_000_000
        for start in range(0, len(cands), B):
            a = embs[field][torch.from_numpy(r1[start:start + B]).cuda()].float()
            b = embs[field][torch.from_numpy(r2[start:start + B]).cuda()].float()
            cos[start:start + B] = (a * b).sum(1).to(torch.float16).cpu().numpy()
            del a, b
        cols[field + "_e5_cos"] = cos
        log(split, "cosine", field, len(cands))
    tbl = pa.table({"s1_id": pa.array(cands["s1_id"].tolist()),
                    "cand_id": pa.array(cands["cand_id"].tolist()),
                    **{k: pa.array(v) for k, v in cols.items()}})
    pq.write_table(tbl, out, compression="zstd")
    log("wrote", out)

json.dump(summary, open(os.path.join(OUT, "e4_summary.json"), "w"), indent=2)
log("E4 COMPLETE", json.dumps(summary))
