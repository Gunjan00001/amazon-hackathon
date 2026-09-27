"""E1 + E3 on the interactive RTX Pro 6000 (Blackwell).

One Run All does both GPU stages of the plan:

  E1  encode name/addr/entity for all 24.2M entities; per-pair cosine for train + valfull
      -> /kaggle/working/cosine_{train,valfull}.parquet  (and cosine_e5_out.zip)
  E3  train-only ANN ceiling: IVF-PQ over train S2/S3, held-out S1 query at K=2000,
      per-truth-pair min ANN rank per field
      -> /kaggle/working/e3_minrank_{name,addr,entity}.parquet + e3_summary.json

E1 outputs are written before E3 starts, so a later failure cannot lose them.
Everything is resumable via .done markers in /kaggle/temp/e1e3.

This notebook is interactive-only: API/CLI runs get a T4x2 and exit via the guard.
"""
import glob
import json
import os
import subprocess
import sys
import time
import zipfile


def log(*a):
    print("[e1e3]", *a, flush=True)


def _pip(pkg, module=None):
    module = module or pkg
    try:
        __import__(module)
    except Exception:
        log("pip install", pkg)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


# Offline assets: the Blackwell requires Internet OFF, so faiss + the e5 model come from the
# attached `amz-er-2026-offline` dataset instead of pip / Hugging Face.
_wheel = glob.glob("/kaggle/input/**/*.whl", recursive=True)
WHEEL_DIR = os.path.dirname(_wheel[0]) if _wheel else None
_safetensors = glob.glob("/kaggle/input/**/model.safetensors", recursive=True)
_model_dir = glob.glob("/kaggle/input/**/multilingual-e5-small", recursive=True)
_model_zip = glob.glob("/kaggle/input/**/multilingual-e5-small.zip", recursive=True)
if _safetensors:
    MODEL_LOCAL = os.path.dirname(_safetensors[0])
elif _model_dir:
    MODEL_LOCAL = _model_dir[0]
elif _model_zip:
    dest = "/kaggle/temp/e1e3/model"
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(_model_zip[0]) as _zf:
        _zf.extractall(dest)
    _found = glob.glob(os.path.join(dest, "**", "model.safetensors"), recursive=True)
    MODEL_LOCAL = os.path.dirname(_found[0]) if _found else os.path.join(dest, "multilingual-e5-small")
else:
    MODEL_LOCAL = None
log("offline model", MODEL_LOCAL, "wheels", WHEEL_DIR)

try:
    import faiss  # noqa: F401
except Exception:
    if WHEEL_DIR:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                        "--find-links", WHEEL_DIR, "faiss-cpu"], check=False)
    else:
        _pip("faiss-cpu", "faiss")

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
import torch  # noqa: E402

OUT = "/kaggle/working"
TMP = "/kaggle/temp/e1e3"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIM = 384
PREFIX = "passage: "
MAXLEN = 64
FIELDS = os.environ.get("E1E3_FIELDS", "name,addr,entity").split(",")
K_MAX = int(os.environ.get("E1E3_KMAX", "2000"))
NLIST = 4096
PQ_M = 48
PQ_NBITS = 8
NPROBE = 48
DO_E3 = os.environ.get("E1E3_DO_E3", "1") == "1"

# --- guard: interactive RTX Pro 6000 only ------------------------------------
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
if "RTX PRO 6000" not in gpu.upper():
    raise SystemExit("needs interactive RTX Pro 6000, got %s" % gpu)
log("gpu", gpu, "vram_GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))


def _find_file(name, base="/kaggle/input"):
    for root, _dirs, files in os.walk(base):
        if name in files:
            return os.path.join(root, name)
    return None


def _find_dir(name, base="/kaggle/input"):
    for root, dirs, _files in os.walk(base):
        if name in dirs:
            return os.path.join(root, name)
    return None


PAIRS_DIR = os.path.dirname(_find_file("pairs_train.parquet") or "")
VAL_IDS_PATH = _find_file("valfull_s1_ids.parquet")
GT_PATH = _find_file("train_ground_truth.tsv")
TSVS = []
for split in ("train", "test"):
    for source in (1, 2, 3):
        p = _find_file("%s_source%d.tsv" % (split, source))
        if p:
            TSVS.append((split, p))
log("pairs_dir", PAIRS_DIR, "val_ids", VAL_IDS_PATH, "gt", GT_PATH, "tsvs", len(TSVS))

if MODEL_LOCAL is None and WHEEL_DIR is None:
    log("WARNING: offline dataset not mounted (model/wheels not found under /kaggle/input)")
if not TSVS:
    try:
        _inp = sorted(os.listdir("/kaggle/input"))
    except Exception:
        _inp = "?"
    raise SystemExit(
        "no input datasets mounted (tsvs=0, /kaggle/input=%s). In the notebook: Add Input -> "
        "attach gunjanpal/amz-er-2026-{raw,e1-pairs,e1-valfull,offline}, then Restart the session "
        "and Run All again." % _inp
    )


# --- entities (all splits, with a split column) ------------------------------
ENT_PATH = os.path.join(TMP, "entities.parquet")
INDEX = os.path.join(TMP, "entity_index.parquet")
EXPECT_N = 24229173
_stale = True
if os.path.exists(ENT_PATH):
    try:
        _stale = pq.ParquetFile(ENT_PATH).metadata.num_rows != EXPECT_N
    except Exception:
        _stale = True
if _stale:
    for _p in (ENT_PATH, INDEX):
        if os.path.exists(_p):
            os.remove(_p)
    con = duckdb.connect()
    con.execute("SET memory_limit='24GB'")
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
    _n = con.execute("SELECT COUNT(*) FROM read_parquet('" + ENT_PATH + "')").fetchone()[0]
    con.close()
    log("entities rebuilt", _n, "(expected %d)" % EXPECT_N)
    if _n != EXPECT_N:
        raise SystemExit("entities count %d != expected %d; are all 7 TSVs mounted?" % (_n, EXPECT_N))
else:
    log("entities cache OK", EXPECT_N)

ENT = pq.read_table(ENT_PATH)
N = len(ENT)
COLS = set(ENT.schema.names)
EID = ENT.column("entity_id")
NAME_RAW = ENT.column("name_raw") if "name_raw" in COLS else ENT.column("name_norm")
ADDR_RAW = ENT.column("addr_raw") if "addr_raw" in COLS else ENT.column("addr_norm")
NAME_NORM = ENT.column("name_norm") if "name_norm" in COLS else NAME_RAW
ADDR_NORM = ENT.column("addr_norm") if "addr_norm" in COLS else ADDR_RAW
IS_S1 = pc.starts_with(EID, pattern="S1-").to_numpy(zero_copy_only=False)
IS_TRAIN = np.array([s == "train" for s in ENT.column("split").to_pylist()], dtype=bool)
log("N", N, "N_s1", int(IS_S1.sum()), "N_train", int(IS_TRAIN.sum()))

INDEX = os.path.join(TMP, "entity_index.parquet")
if not os.path.exists(INDEX):
    pq.write_table(pa.table({"entity_id": EID, "row_idx": pa.array(np.arange(N, dtype=np.int64))}),
                   INDEX, compression="zstd")
eid_list = EID.to_pylist()
row_of = {e: i for i, e in enumerate(eid_list)}

# --- E3 setup: candidates (train S2/S3), held-out S1 rows, truth rows --------
CAND_ROWS = np.nonzero(IS_TRAIN & ~IS_S1)[0]
val_ids = set()
if VAL_IDS_PATH:
    val_ids = set(pq.read_table(VAL_IDS_PATH, columns=["s1_id"]).column("s1_id").to_pylist())
H = np.array([row_of[e] for e in val_ids if e in row_of], dtype=np.int64)
truth = None
if GT_PATH and H.size:
    gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False,
                     usecols=["source1_entity_id", "matched_entity_ids"])
    rows = []
    for s1, mids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if not mids or s1 not in val_ids:
            continue
        rs = row_of.get(s1)
        if rs is None:
            continue
        for c in mids.split(","):
            rc = row_of.get(c)
            if rc is not None:
                rows.append((rs, rc, s1, c))
    truth = pd.DataFrame(rows, columns=["s1_row", "cand_row", "s1_id", "cand_id"])
    del gt
    log("E3 candidates", len(CAND_ROWS), "heldout_s1", len(H), "truth_pairs", len(truth))

model = SentenceTransformer(MODEL_LOCAL) if MODEL_LOCAL else SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
model.max_seq_length = MAXLEN
model = model.to("cuda").half()
log("model ready", "local" if MODEL_LOCAL else "hf")


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


def encode_field(field):
    path = os.path.join(TMP, "emb_%s.f16.npy" % field)
    if os.path.exists(path + ".done"):
        try:
            if np.load(path, mmap_mode="r").shape[0] == N:
                return path
        except Exception:
            pass
        log("stale emb cache for", field, "-> re-encode")
        for _p in (path, path + ".done"):
            if os.path.exists(_p):
                os.remove(_p)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(N, DIM))
    off = 0
    t0 = time.time()
    for texts in field_texts(field):
        vecs = model.encode([PREFIX + t for t in texts], batch_size=2048,
                            normalize_embeddings=True, convert_to_numpy=True)
        k = len(texts)
        mm[off:off + k] = vecs.astype(np.float16)
        off += k
        if off % (131072 * 20) < 131072:
            mm.flush()
            log(field, off, "/", N, "%.0f/s" % (off / max(time.time() - t0, 1e-9)))
    mm.flush()
    json.dump({"field": field, "rows": off}, open(path + ".done", "w"))
    log("encode done", field, off, "elapsed_s", round(time.time() - t0, 1))
    return path


# --- E1: per-pair cosine -----------------------------------------------------
COS_DIR = os.path.join(TMP, "cos_parts")
os.makedirs(COS_DIR, exist_ok=True)


def e1_cosine(split, emb_gpu, field):
    out = os.path.join(COS_DIR, "cos_%s_%s.parquet" % (split, field))
    if os.path.exists(out):
        return out
    pairs_path = os.path.join(PAIRS_DIR, "pairs_%s.parquet" % split)
    con = duckdb.connect()
    con.execute("SET memory_limit='24GB'")
    con.execute("SET threads=8")
    con.execute("SET temp_directory='" + TMP + "'")
    con.execute("CREATE VIEW ei AS SELECT * FROM read_parquet('" + INDEX + "')")
    reader = con.execute(
        "SELECT p.s1_id, p.cand_id, e1.row_idx AS r1, e2.row_idx AS r2 "
        "FROM read_parquet('" + pairs_path + "') p "
        "JOIN ei e1 ON e1.entity_id = p.s1_id JOIN ei e2 ON e2.entity_id = p.cand_id"
    ).fetch_record_batch(1_000_000)
    tmp = out + ".tmp"
    writer = None
    total = 0
    t0 = time.time()
    for rb in reader:
        r1 = torch.from_numpy(rb.column("r1").to_numpy()).cuda()
        r2 = torch.from_numpy(rb.column("r2").to_numpy()).cuda()
        cos = (emb_gpu[r1].float() * emb_gpu[r2].float()).sum(1).to(torch.float16).cpu().numpy()
        tbl = pa.table({"s1_id": rb.column("s1_id"), "cand_id": rb.column("cand_id"),
                        field + "_e5_cos": pa.array(cos)})
        if writer is None:
            writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        total += len(tbl)
        if total % 5_000_000 < 1_000_000:
            log("E1", split, field, total, "%.0f/s" % (total / max(time.time() - t0, 1e-9)))
    if writer is None:
        con.close()
        raise RuntimeError("no pairs joined for %s/%s (stale/empty entity index?)" % (split, field))
    writer.close()
    os.replace(tmp, out)
    con.close()
    log("E1 DONE", split, field, total)
    return out


# --- E3: ANN ceiling per field ----------------------------------------------
def e3_minrank(field, emb_path):
    import faiss

    out = os.path.join(OUT, "e3_minrank_%s.parquet" % field)
    if os.path.exists(out):
        return out
    if truth is None or not len(H):
        log("E3 skipped (no val ids/truth)")
        return None
    emb = np.load(emb_path, mmap_mode="r")
    queries = np.asarray(emb[H]).astype(np.float32)
    index = faiss.IndexIVFPQ(faiss.IndexFlatIP(DIM), DIM, NLIST, PQ_M, PQ_NBITS,
                             faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    sample = CAND_ROWS[::max(1, len(CAND_ROWS) // 1_000_000)][:1_000_000]
    index.train(np.asarray(emb[sample]).astype(np.float32))
    for start in range(0, len(CAND_ROWS), 2_000_000):
        block = CAND_ROWS[start:start + 2_000_000]
        index.add(np.asarray(emb[block]).astype(np.float32))
    index.nprobe = NPROBE
    log("E3", field, "index ntotal", index.ntotal, "elapsed_s", round(time.time() - t0, 1))

    ann_path = os.path.join(TMP, "ann_%s.parquet" % field)
    writer = None
    t1 = time.time()
    QB = 20_000
    for start in range(0, len(H), QB):
        q = queries[start:start + QB]
        _scores, nbrs = index.search(q, K_MAX)
        qrow = H[start:start + QB]
        valid = (nbrs >= 0).ravel()
        s1_rep = np.repeat(qrow, K_MAX)[valid]
        rank_rep = np.tile(np.arange(K_MAX, dtype=np.int16), len(qrow))[valid]
        cand_rep = CAND_ROWS[np.where(nbrs >= 0, nbrs, 0)].ravel()[valid]
        tbl = pa.table({"s1_row": pa.array(s1_rep.astype(np.int32)),
                        "cand_row": pa.array(cand_rep.astype(np.int32)),
                        "rank": pa.array(rank_rep)})
        if writer is None:
            writer = pq.ParquetWriter(ann_path, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        if start % (QB * 20) == 0:
            log("E3", field, min(start + QB, len(H)), "/", len(H),
                "%.0f q/s" % (min(start + QB, len(H)) / max(time.time() - t1, 1e-9)))
    if writer is not None:
        writer.close()

    con = duckdb.connect()
    con.execute("SET memory_limit='24GB'")
    con.execute("SET threads=8")
    con.execute("SET temp_directory='" + TMP + "'")
    con.register("truth_df", truth)
    con.execute("CREATE TEMP TABLE truth AS SELECT s1_row, cand_row, s1_id, cand_id FROM truth_df")
    con.execute("CREATE TEMP TABLE ann AS SELECT s1_row, cand_row, rank FROM read_parquet('" + ann_path + "')")
    con.execute(
        "COPY (SELECT t.s1_id, t.cand_id, MIN(a.rank) AS min_rank FROM truth t "
        "LEFT JOIN ann a ON a.s1_row = t.s1_row AND a.cand_row = t.cand_row "
        "GROUP BY t.s1_id, t.cand_id) TO '" + out + "' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()
    os.remove(ann_path)
    log("E3 DONE", field, "minrank rows",
        pq.ParquetFile(out).metadata.num_rows)
    return out


# --- run ---------------------------------------------------------------------
t_all = time.time()
SPLITS = [s for s in ("train", "valfull") if os.path.exists(os.path.join(PAIRS_DIR, "pairs_%s.parquet" % s))]
for field in FIELDS:
    emb_path = encode_field(field)
    emb_gpu = torch.from_numpy(np.asarray(np.load(emb_path, mmap_mode="r"))).cuda()
    for split in SPLITS:
        e1_cosine(split, emb_gpu, field)
    del emb_gpu
    torch.cuda.empty_cache()
    if DO_E3:
        try:
            e3_minrank(field, emb_path)
        except Exception as exc:  # noqa: BLE001
            log("E3 FAILED for", field, ":", repr(exc))
    log("field complete", field, "elapsed_s", round(time.time() - t_all, 1))

# --- merge E1 cosines + zip (E1 outputs saved before E3 dependence) ----------
_ALIAS = {"name": "n", "addr": "a", "entity": "e"}
for split in SPLITS:
    avail = [f for f in ("name", "addr", "entity") if os.path.exists(os.path.join(COS_DIR, "cos_%s_%s.parquet" % (split, f)))]
    con = duckdb.connect()
    con.execute("SET memory_limit='24GB'")
    con.execute("SET threads=8")
    for f in avail:
        con.execute("CREATE TEMP TABLE " + f + " AS SELECT * FROM read_parquet('" + os.path.join(COS_DIR, "cos_%s_%s.parquet" % (split, f)) + "')")
    base = avail[0]
    select = ["%s.s1_id AS s1_id, %s.cand_id AS cand_id" % (_ALIAS[base], _ALIAS[base])]
    for f in avail:
        select.append("%s.%s_e5_cos AS %s_e5_cos" % (_ALIAS[f], f, f))
    joins = "".join(" LEFT JOIN " + f + " " + _ALIAS[f] + " ON " + _ALIAS[f] + ".s1_id = " + _ALIAS[base] +
                    ".s1_id AND " + _ALIAS[f] + ".cand_id = " + _ALIAS[base] + ".cand_id" for f in avail[1:])
    out = os.path.join(OUT, "cosine_%s.parquet" % split)
    con.execute("COPY (SELECT " + ", ".join(select) + " FROM " + base + " " + _ALIAS[base] + joins + ") "
                "TO '" + out + "' (FORMAT PARQUET, COMPRESSION ZSTD)")
    log("E1 merged", split, con.execute("SELECT COUNT(*) FROM read_parquet('" + out + "')").fetchone()[0])
    con.close()

with zipfile.ZipFile(os.path.join(OUT, "cosine_e5_out.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
    for split in SPLITS:
        zf.write(os.path.join(OUT, "cosine_%s.parquet" % split), "cosine_%s.parquet" % split)

summary = {"fields": FIELDS, "N": N, "N_train": int(IS_TRAIN.sum()), "n_candidates": int(len(CAND_ROWS)),
           "n_heldout_s1": int(len(H)), "n_truth_pairs": int(len(truth)) if truth is not None else 0,
           "K_MAX": K_MAX, "nlist": NLIST, "m": PQ_M, "nbits": PQ_NBITS, "nprobe": NPROBE,
           "wall_s": round(time.time() - t_all, 1)}
json.dump(summary, open(os.path.join(OUT, "e3_summary.json"), "w"), indent=2)
log("ALL COMPLETE", json.dumps(summary))
