"""Local blocking probe: measures recall ceiling and candidate count vs cap.

Samples S1 entities, keeps their true mids plus a background sample of mids,
then runs the production blocking code at several caps. CPU-only validation
(RULES.md §6.1): no training, no GPU.

Usage:
    .venv\\Scripts\\python.exe tools\\probe_blocking.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))

from ber.blocking import generate_candidates  # noqa: E402
from ber.io_tsv import load_ground_truth, load_records, split_id_list  # noqa: E402

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", r"D:\Amazon project\DATA\student_resource\dataset"))
N_S1 = int(os.environ.get("PROBE_N_S1", "30000"))
MID_SAMPLE = float(os.environ.get("PROBE_MID_SAMPLE", "0.1"))
CAPS = [int(x) for x in os.environ.get("PROBE_CAPS", "30,60,120").split(",")]
SEED = 42
CHUNK = 1_000_000

rng = np.random.default_rng(SEED)


def main():
    print(f"data_dir={DATA_DIR} n_s1={N_S1} mid_sample={MID_SAMPLE} caps={CAPS}")
    t0 = time.time()
    s1 = load_records(DATA_DIR / "train" / "train_source1.tsv")
    idx = rng.choice(len(s1), size=min(N_S1, len(s1)), replace=False)
    s1 = s1.iloc[np.sort(idx)].reset_index(drop=True)
    s1_map = {e: i for i, e in enumerate(s1["entity_id"])}
    print(f"sampled S1: {len(s1):,} in {time.time()-t0:.0f}s")

    gt = load_ground_truth(DATA_DIR / "train" / "train_ground_truth.tsv")
    gt = gt[gt["source1_entity_id"].isin(s1_map)]
    true_mid_ids = set()
    for mids in gt["matched_entity_ids"]:
        true_mid_ids.update(split_id_list(mids))
    print(f"S1 with labels in sample: {len(gt):,}; true mid ids: {len(true_mid_ids):,}")

    kept = []
    for name in ("train_source2.tsv", "train_source3.tsv"):
        t1 = time.time()
        n_kept = 0
        for chunk in pd.read_csv(DATA_DIR / "train" / name, sep="\t", dtype=str,
                                 keep_default_na=False, chunksize=CHUNK):
            keep = chunk["entity_id"].isin(true_mid_ids) | (rng.random(len(chunk)) < MID_SAMPLE)
            if keep.any():
                kept.append(chunk[keep])
                n_kept += int(keep.sum())
        print(f"streamed {name}: kept {n_kept:,} in {time.time()-t1:.0f}s")

    mid = pd.concat(kept, ignore_index=True)
    mid_map = {e: i for i, e in enumerate(mid["entity_id"])}
    print(f"mid pool: {len(mid):,}")

    true_pairs = set()
    missing = 0
    for sid, mids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        i = s1_map[sid]
        for m in split_id_list(mids):
            j = mid_map.get(m)
            if j is None:
                missing += 1
            else:
                true_pairs.add((i, j))
    print(f"true pairs evaluable: {len(true_pairs):,} (missing mids: {missing:,})")

    results = {}
    for cap in CAPS:
        t1 = time.time()
        pairs = generate_candidates(s1, mid, max_candidates_per_s1=cap, max_postings=10000)
        got = {(int(a), int(b)) for a, b in zip(pairs["s1_idx"], pairs["mid_idx"])}
        recall = len(got & true_pairs) / max(1, len(true_pairs))
        counts = pairs.groupby("s1_idx").size().reindex(range(len(s1)), fill_value=0)
        results[cap] = {
            "pairs": int(len(pairs)),
            "recall_ceiling": round(recall, 4),
            "pairs_per_s1_mean": round(float(counts.mean()), 2),
            "pairs_per_s1_p95": float(counts.quantile(0.95)),
            "by_pass": {str(k): int(v) for k, v in pairs["pass"].value_counts().items()},
            "seconds": round(time.time() - t1, 1),
        }
        print(cap, results[cap], flush=True)

    pd.DataFrame(results).T.to_json(REPO / "tools" / "probe_blocking.json", indent=2, orient="index")
    print("wrote tools/probe_blocking.json")


if __name__ == "__main__":
    main()
