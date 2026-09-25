import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..vector_features import MODEL_ID, MODEL_REVISION, encode_texts, quantize, record_text


def encode_source(df, model_id, revision, batch_size, prefix):
    texts = [record_text(n, a) for n, a in zip(df["business_name"], df["business_address"])]
    vecs = encode_texts(texts, model_id=model_id, revision=revision, batch_size=batch_size, prefix=prefix)
    return quantize(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default=str(config.ARTIFACT_DIR / "clean"))
    ap.add_argument("--out-dir", default=str(config.ARTIFACT_DIR / "embed"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    clean_dir, out_dir = Path(args.clean_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"model_id": args.model_id, "revision": args.revision}
    for split in ("train", "test"):
        ids = []
        t0 = time.time()
        for n, prefix in ((1, "query: "), (2, "passage: "), (3, "passage: ")):
            df = pd.read_parquet(clean_dir / f"{split}_s{n}.parquet",
                                 columns=["entity_id", "business_name", "business_address"])
            if args.sample:
                df = df.head(args.sample)
            vecs = encode_source(df, args.model_id, args.revision, args.batch_size, prefix)
            np.save(out_dir / f"{split}_s{n}.npy", vecs)
            ids.append(pd.DataFrame({"source": n, "entity_id": df["entity_id"].to_numpy()}))
            stats[f"{split}_s{n}"] = {"rows": int(len(df)), "dim": int(vecs.shape[1])}
            print(f"{split}_s{n}: {len(df):,} rows encoded in {time.time()-t0:.0f}s", flush=True)
        pd.concat(ids, ignore_index=True).to_parquet(out_dir / f"{split}_ids.parquet", index=False)

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
