import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import parquet as pq

from .. import config
from ..vector_features import MODEL_ID, MODEL_REVISION, VEC_DIM, quantize, record_text


def load_model(model_id, revision):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, revision=revision)


def encode_source(parquet_path, out_path, model, prefix, batch_size, read_batch, limit=0):
    pf = pq.ParquetFile(parquet_path)
    n = pf.metadata.num_rows
    if limit:
        n = min(n, limit)
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype="int8", shape=(n, VEC_DIM))
    pos = 0
    for batch in pf.iter_batches(batch_size=read_batch, columns=["business_name", "business_address"]):
        if pos >= n:
            break
        names = batch.column("business_name").to_pylist()
        addrs = batch.column("business_address").to_pylist()
        if pos + len(names) > n:
            names = names[: n - pos]
            addrs = addrs[: n - pos]
        texts = [prefix + record_text(x, y) for x, y in zip(names, addrs)]
        vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        arr[pos:pos + len(texts)] = quantize(vecs)
        pos += len(texts)
    arr.flush()
    del arr
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default=str(config.ARTIFACT_DIR / "clean"))
    ap.add_argument("--out-dir", default=str(config.ARTIFACT_DIR / "embed"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--read-batch", type=int, default=50_000)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    clean_dir, out_dir = Path(args.clean_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_id, args.revision)

    stats = {"model_id": args.model_id, "revision": args.revision, "dim": VEC_DIM}
    for split in ("train", "test"):
        ids = []
        for n, prefix in ((1, "query: "), (2, "passage: "), (3, "passage: ")):
            path = clean_dir / f"{split}_s{n}.parquet"
            out = out_dir / f"{split}_s{n}.npy"
            t0 = time.time()
            rows = encode_source(path, out, model, prefix, args.batch_size, args.read_batch, args.sample)
            ids.append(pd.read_parquet(path, columns=["entity_id"]).head(rows))
            stats[f"{split}_s{n}"] = {"rows": int(rows), "seconds": round(time.time() - t0, 1)}
            print(f"{split}_s{n}: {rows:,} rows in {time.time()-t0:.0f}s", flush=True)
        pd.concat(ids, ignore_index=True).to_parquet(out_dir / f"{split}_ids.parquet", index=False)

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
