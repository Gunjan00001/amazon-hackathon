"""E1 Blackwell runner — full 384-dim multilingual-e5 cosine on an RTX Pro 6000.

Interactive-only: this notebook is meant to be run with "Run All" in a browser session that has
the RTX Pro 6000 accelerator (97,887 MiB). API/CLI runs allocate a T4x2 and will exit immediately
via the guard below, so a pushed version does not waste time.

With ~96 GB VRAM we hold one field's embeddings resident and do batched GPU gather + dot, instead
of the streaming/argsort machinery required on a T4.

Outputs (in /kaggle/working): cosine_train.parquet, cosine_valfull.parquet, cosine_e5_out.zip
"""
import json
import os
import subprocess
import sys
import time
import zipfile


def log(*a):
    print("[e1b]", *a, flush=True)


def _pip(pkg, module=None):
    module = module or pkg
    try:
        __import__(module)
    except Exception:
        log("pip install", pkg)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


_pip("sentence-transformers")

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
import torch  # noqa: E402

OUT = "/kaggle/working"
TMP = "/kaggle/temp/e1b"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIM = 384
PREFIX = "passage: "
MAXLEN = 64
FIELDS = os.environ.get("E1B_FIELDS", "name,addr,entity").split(",")
COS_BATCH = 4_000_000

# --- guard: only proceed on the RTX Pro 6000 (interactive) -------------------
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
log("gpu:", gpu, "vram_GB:", vram)
if "RTX PRO 6000" not in gpu.upper():
    raise SystemExit(
        "This notebook needs the interactive RTX Pro 6000 (got %s). "
        "Open it in the browser with the RTX Pro 6000 accelerator and Run All." % gpu
    )


def _find_file(name, base="/kaggle/input"):
    for root, _dirs, files in os.walk(base):
        if name in files:
            return os.path.join(root, name)
    return None


def _raw_tsvs():
    found = []
    for split in ("train", "test"):
        for source in (1, 2, 3):
            path = _find_file("%s_source%d.tsv" % (split, source))
            if path:
                found.append(path)
    return found


PAIRS_DIR = os.path.dirname(_find_file("pairs_train.parquet") or "")
ENT = None


def build_entities():
    ent_path = os.path.join(TMP, "entities.parquet")
    if os.path.exists(ent_path):
        return ent_path
    raw = _raw_tsvs()
    if not raw:
        raise SystemExit("no raw TSVs found")
    con = duckdb.connect()
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory='" + TMP + "'")
    files = "[" + ",".join("'" + f + "'" for f in raw) + "]"
    con.execute(
        "COPY (SELECT entity_id, any_value(business_name) AS name_raw, "
        "any_value(business_address) AS addr_raw FROM read_csv(" + files + ", delim='\\t', "
        "header=true, union_by_name=true, all_varchar=true) WHERE entity_id IS NOT NULL "
        "GROUP BY entity_id) TO '" + ent_path + "' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    rows = con.execute("SELECT COUNT(*) FROM read_parquet('" + ent_path + "')").fetchone()[0]
    con.close()
    log("entities", rows)
    return ent_path


ENT = pq.read_table(build_entities())
N = len(ENT)
COLS = set(ENT.schema.names)
EID = ENT.column("entity_id")
NAME_RAW = ENT.column("name_raw") if "name_raw" in COLS else ENT.column("name_norm")
ADDR_RAW = ENT.column("addr_raw") if "addr_raw" in COLS else ENT.column("addr_norm")
NAME_NORM = ENT.column("name_norm") if "name_norm" in COLS else NAME_RAW
ADDR_NORM = ENT.column("addr_norm") if "addr_norm" in COLS else ADDR_RAW
log("N", N, "pairs_dir", PAIRS_DIR)

INDEX = os.path.join(TMP, "entity_index.parquet")
if not os.path.exists(INDEX):
    pq.write_table(pa.table({"entity_id": EID, "row_idx": pa.array(np.arange(N, dtype=np.int64))}),
                   INDEX, compression="zstd")

model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
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


def encode_field(field):
    """Return a (N, DIM) fp16 CUDA tensor for one field."""
    path = os.path.join(TMP, "emb_%s.f16.npy" % field)
    if os.path.exists(path + ".done"):
        log("load cached emb", field)
        arr = np.load(path, mmap_mode="r")
        return torch.from_numpy(np.asarray(arr)).cuda()
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
    return torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"))).cuda()


def cosine_split(split, emb_gpu, field, out_dir):
    out = os.path.join(out_dir, "cos_%s_%s.parquet" % (split, field))
    if os.path.exists(out):
        log("cosine skip", split, field)
        return out
    pairs_path = os.path.join(PAIRS_DIR, "pairs_%s.parquet" % split)
    con = duckdb.connect()
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory='" + TMP + "'")
    con.execute("CREATE VIEW ei AS SELECT * FROM read_parquet('" + INDEX + "')")
    con.execute("CREATE VIEW pr AS SELECT * FROM read_parquet('" + pairs_path + "')")
    reader = con.execute(
        "SELECT p.s1_id, p.cand_id, e1.row_idx AS r1, e2.row_idx AS r2 "
        "FROM pr p JOIN ei e1 ON e1.entity_id = p.s1_id "
        "JOIN ei e2 ON e2.entity_id = p.cand_id"
    ).fetch_record_batch(1_000_000)
    tmp = out + ".tmp"
    writer = None
    total = 0
    t0 = time.time()
    for rb in reader:
        r1 = torch.from_numpy(rb.column("r1").to_numpy()).cuda()
        r2 = torch.from_numpy(rb.column("r2").to_numpy()).cuda()
        a = emb_gpu[r1].float()
        b = emb_gpu[r2].float()
        cos = (a * b).sum(1).to(torch.float16).cpu().numpy()
        tbl = pa.table({"s1_id": rb.column("s1_id"), "cand_id": rb.column("cand_id"),
                        field + "_e5_cos": pa.array(cos)})
        if writer is None:
            writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        total += len(tbl)
        if total % 5_000_000 < 1_000_000:
            log(split, field, total, "%.0f/s" % (total / max(time.time() - t0, 1e-9)))
    if writer is not None:
        writer.close()
    os.replace(tmp, out)
    con.close()
    log("DONE cosine", split, field, total)
    return out


COS_DIR = os.path.join(TMP, "cos_parts")
os.makedirs(COS_DIR, exist_ok=True)
SPLITS = [s for s in ("train", "valfull") if os.path.exists(os.path.join(PAIRS_DIR, "pairs_%s.parquet" % s))]
t_all = time.time()
for field in FIELDS:
    emb_gpu = encode_field(field)
    for split in SPLITS:
        cosine_split(split, emb_gpu, field, COS_DIR)
    del emb_gpu
    torch.cuda.empty_cache()
    log("field complete", field, "elapsed_s", round(time.time() - t_all, 1))

_ALIAS = {"name": "n", "addr": "a", "entity": "e"}
for split in SPLITS:
    avail = [f for f in ("name", "addr", "entity") if os.path.exists(os.path.join(COS_DIR, "cos_%s_%s.parquet" % (split, f)))]
    con = duckdb.connect()
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")
    for f in avail:
        con.execute("CREATE TEMP TABLE " + f + " AS SELECT * FROM read_parquet('" + os.path.join(COS_DIR, "cos_%s_%s.parquet" % (split, f)) + "')")
    base = avail[0]
    select = ["%s.s1_id AS s1_id, %s.cand_id AS cand_id" % (_ALIAS[base], _ALIAS[base])]
    for f in avail:
        select.append("%s.%s_e5_cos AS %s_e5_cos" % (_ALIAS[f], f, f))
    joins = "".join(
        " LEFT JOIN " + f + " " + _ALIAS[f] + " ON " + _ALIAS[f] + ".s1_id = " + _ALIAS[base] +
        ".s1_id AND " + _ALIAS[f] + ".cand_id = " + _ALIAS[base] + ".cand_id" for f in avail[1:])
    out = os.path.join(OUT, "cosine_%s.parquet" % split)
    con.execute("COPY (SELECT " + ", ".join(select) + " FROM " + base + " " + _ALIAS[base] + joins + ") "
                "TO '" + out + "' (FORMAT PARQUET, COMPRESSION ZSTD)")
    rows = con.execute("SELECT COUNT(*) FROM read_parquet('" + out + "')").fetchone()[0]
    log(split, "merged rows", rows, "fields", avail)
    con.close()

with zipfile.ZipFile(os.path.join(OUT, "cosine_e5_out.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
    for split in SPLITS:
        zf.write(os.path.join(OUT, "cosine_%s.parquet" % split), "cosine_%s.parquet" % split)
log("E1B COMPLETE wall_s", round(time.time() - t_all, 1))
