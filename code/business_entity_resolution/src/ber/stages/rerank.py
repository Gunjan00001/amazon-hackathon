import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..labels import positive_pairs
from ..rerank import MODEL_ID, MODEL_REVISION, score_texts, serialize_pair
from ..score import best_threshold


def load_mid(clean_dir, split):
    return pd.concat(
        [pd.read_parquet(clean_dir / f"{split}_s2.parquet"), pd.read_parquet(clean_dir / f"{split}_s3.parquet")],
        ignore_index=True,
    )


def serialize_pairs(s1, mid, pairs):
    n1 = s1["business_name"].to_numpy(dtype=object)
    a1 = s1["business_address"].to_numpy(dtype=object)
    c1 = s1["country"].to_numpy(dtype=object)
    n2 = mid["business_name"].to_numpy(dtype=object)
    a2 = mid["business_address"].to_numpy(dtype=object)
    c2 = mid["country"].to_numpy(dtype=object)
    i = pairs["s1_idx"].to_numpy()
    j = pairs["mid_idx"].to_numpy()
    return [serialize_pair(n1[a], a1[a], c1[a], n2[b], a2[b], c2[b]) for a, b in zip(i, j)]


def score_frame(s1, mid, pairs, model_id, revision, batch_size, chunk=500_000):
    parts = []
    n = len(pairs)
    for start in range(0, n, chunk):
        sub = pairs.iloc[start:start + chunk]
        texts = serialize_pairs(s1, mid, sub)
        scores = score_texts(texts, model_id=model_id, revision=revision, batch_size=batch_size)
        parts.append(pd.DataFrame({
            "s1_idx": sub["s1_idx"].to_numpy(),
            "mid_idx": sub["mid_idx"].to_numpy(),
            "ce_score": scores.astype("float32"),
        }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["s1_idx", "mid_idx", "ce_score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default=str(config.ARTIFACT_DIR / "clean"))
    ap.add_argument("--block-dir", default=str(config.ARTIFACT_DIR / "block"))
    ap.add_argument("--gbdt-dir", default=str(config.ARTIFACT_DIR / "gbdt"))
    ap.add_argument("--out-dir", default=str(config.ARTIFACT_DIR / "rerank"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tune-sample", type=int, default=20000)
    args = ap.parse_args()

    clean_dir, block_dir, out_dir = Path(args.clean_dir), Path(args.block_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_s1 = pd.read_parquet(clean_dir / "test_s1.parquet")
    test_mid = load_mid(clean_dir, "test")
    pruned = pd.read_parquet(Path(args.gbdt_dir) / "pruned_test.parquet")
    print(f"reranking {len(pruned):,} pairs", flush=True)

    t0 = time.time()
    scored = score_frame(test_s1, test_mid, pruned, args.model_id, args.revision, args.batch_size)
    scored.to_parquet(out_dir / "test_ce_scores.parquet", index=False)
    print(f"scored in {time.time()-t0:.0f}s", flush=True)

    metrics = {"model_id": args.model_id, "revision": args.revision, "test_pairs": int(len(scored))}
    if args.tune_sample:
        train_s1 = pd.read_parquet(clean_dir / "train_s1.parquet")
        train_mid = load_mid(clean_dir, "train")
        labels = pd.read_parquet(clean_dir / "labels.parquet")
        cand = pd.read_parquet(block_dir / "train_candidates.parquet")
        rng = np.random.default_rng(config.SEED)
        sample_ids = rng.choice(len(train_s1), size=min(args.tune_sample, len(train_s1)), replace=False)
        sub = cand[cand["s1_idx"].isin(set(sample_ids.tolist()))].reset_index(drop=True)
        pos = set(positive_pairs(train_s1["entity_id"].tolist(), train_mid["entity_id"].tolist(), labels))
        y = np.fromiter((1 if (int(a), int(b)) in pos else 0 for a, b in
                         zip(sub["s1_idx"], sub["mid_idx"])), dtype="int8", count=len(sub))
        val = score_frame(train_s1, train_mid, sub, args.model_id, args.revision, args.batch_size)
        val["y"] = y
        val.to_parquet(out_dir / "val_ce_scores.parquet", index=False)
        f, th = best_threshold(sub["s1_idx"].to_numpy(), y, val["ce_score"].to_numpy())
        metrics.update({"tune_pairs": int(len(sub)), "tune_positives": int(y.sum()),
                        "best_macro_f05": float(f), "best_threshold": float(th)})
        print(json.dumps({k: v for k, v in metrics.items() if k != "model_id"}, indent=2), flush=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
