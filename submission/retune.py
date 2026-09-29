import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))

from ber.io_tsv import write_matching_results  # noqa: E402


def load_ids(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id"])["entity_id"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruned", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.7, 0.75])
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    s1_ids = load_ids(test_dir / "test_source1.tsv")
    mid_ids = load_ids(test_dir / "test_source2.tsv") + load_ids(test_dir / "test_source3.tsv")
    pruned = pd.read_parquet(args.pruned, columns=["s1_idx", "mid_idx", "score"])

    for th in args.thresholds:
        kept = pruned[pruned["score"] >= th]
        matches = {}
        for s1, mid in zip(kept["s1_idx"], kept["mid_idx"]):
            matches.setdefault(int(s1), []).append(int(mid))
        out = Path(args.out) / f"matching_results_t{int(round(th*1000)):03d}.tsv"
        match_ids = {s1_ids[i]: [mid_ids[m] for m in v] for i, v in matches.items()}
        write_matching_results(out, s1_ids, match_ids)
        n = sum(len(v) for v in match_ids.values())
        print(f"threshold {th}: rows {len(s1_ids)} matches {n} non_empty {len(s1_ids)-sum(1 for v in match_ids.values() if not v)} -> {out}")


if __name__ == "__main__":
    main()
