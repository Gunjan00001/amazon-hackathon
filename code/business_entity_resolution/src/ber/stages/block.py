import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from .. import config
from ..blocking import blocking_recall, generate_candidates
from ..labels import positive_pairs

READ_COLS = ["entity_id", "business_name", "business_address", "country"]


def load_mid(clean_dir, split):
    frames = [pd.read_parquet(clean_dir / f"{split}_s2.parquet", columns=READ_COLS),
              pd.read_parquet(clean_dir / f"{split}_s3.parquet", columns=READ_COLS)]
    return pd.concat(frames, ignore_index=True)


def process(split, clean_dir, out_dir, max_candidates, max_postings, labels, s1_limit=0, mid_limit=0):
    s1 = pd.read_parquet(clean_dir / f"{split}_s1.parquet", columns=READ_COLS)
    mid = load_mid(clean_dir, split)
    if s1_limit:
        s1 = s1.head(s1_limit)
    if mid_limit:
        mid = mid.head(mid_limit)
    pairs = generate_candidates(s1, mid, max_candidates, max_postings)
    pairs.to_parquet(out_dir / f"{split}_candidates.parquet", index=False)

    per_s1 = pairs.groupby("s1_idx").size().reindex(range(len(s1)), fill_value=0)
    st = {
        "pairs": int(len(pairs)),
        "mid_records": int(len(mid)),
        "pairs_per_s1_mean": float(per_s1.mean()),
        "pairs_per_s1_p95": float(per_s1.quantile(0.95)),
        "by_pass": {str(k): int(v) for k, v in pairs["pass"].value_counts().items()},
    }
    if split == "train" and labels is not None:
        lp = set(positive_pairs(s1["entity_id"].tolist(), mid["entity_id"].tolist(), labels))
        st["labels_evaluated"] = len(lp)
        st["recall_ceiling"] = blocking_recall(pairs, lp)
    del s1, mid, pairs
    gc.collect()
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default=str(config.ARTIFACT_DIR / "clean"))
    ap.add_argument("--out-dir", default=str(config.ARTIFACT_DIR / "block"))
    ap.add_argument("--max-candidates", type=int, default=60)
    ap.add_argument("--max-postings", type=int, default=10000)
    ap.add_argument("--s1-limit", type=int, default=0)
    ap.add_argument("--mid-limit", type=int, default=0)
    args = ap.parse_args()

    clean_dir, out_dir = Path(args.clean_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = clean_dir / "labels.parquet"
    labels = pd.read_parquet(labels_path) if labels_path.exists() else None

    all_stats = {}
    for split in ("train", "test"):
        st = process(split, clean_dir, out_dir, args.max_candidates, args.max_postings, labels,
                     args.s1_limit, args.mid_limit)
        all_stats[split] = st
        print(split, json.dumps(st))

    (out_dir / "stats.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
