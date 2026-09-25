import argparse
import json
from pathlib import Path

import pandas as pd

from .. import config
from ..decision import matches_from_scores, one_to_one_assign, to_entity_ids
from ..io_tsv import write_candidate_pairs, write_matching_results


def load_mid_ids(clean_dir, split):
    return pd.concat(
        [pd.read_parquet(clean_dir / f"{split}_s2.parquet"), pd.read_parquet(clean_dir / f"{split}_s3.parquet")],
        ignore_index=True,
    )["entity_id"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default=str(config.ARTIFACT_DIR / "clean"))
    ap.add_argument("--block-dir", default=str(config.ARTIFACT_DIR / "block"))
    ap.add_argument("--gbdt-dir", default=str(config.ARTIFACT_DIR / "gbdt"))
    ap.add_argument("--out-dir", default=str(config.ARTIFACT_DIR / "out"))
    ap.add_argument("--ce-dir", default="")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--one-to-one", action="store_true")
    args = ap.parse_args()

    clean_dir, out_dir = Path(args.clean_dir), Path(args.out_dir)
    gbdt_dir = Path(args.gbdt_dir)
    s1_ids = pd.read_parquet(clean_dir / "test_s1.parquet")["entity_id"].tolist()
    mid_ids = load_mid_ids(clean_dir, "test")
    candidates = pd.read_parquet(gbdt_dir / "pruned_test.parquet")

    if args.ce_dir:
        ce = pd.read_parquet(Path(args.ce_dir) / "test_ce_scores.parquet")
        score_col = "ce_score" if "ce_score" in ce.columns else "score"
        df = candidates.merge(ce[["s1_idx", "mid_idx", score_col]], on=["s1_idx", "mid_idx"], how="left")
        metrics_path = Path(args.ce_dir) / "metrics.json"
        if not metrics_path.exists():
            metrics_path = gbdt_dir / "metrics.json"
    else:
        if "score" in candidates.columns:
            df = candidates
        else:
            gb = pd.read_parquet(gbdt_dir / "test_scores.parquet")
            df = candidates.merge(gb[["s1_idx", "mid_idx", "score"]], on=["s1_idx", "mid_idx"], how="left")
        score_col = "score"
        metrics_path = gbdt_dir / "metrics.json"

    threshold = args.threshold
    if threshold is None:
        threshold = float(json.loads(metrics_path.read_text(encoding="utf-8"))["best_threshold"])

    df = df.dropna(subset=[score_col]).reset_index(drop=True)
    if args.one_to_one:
        matches_idx = one_to_one_assign(df, threshold, score_col)
    else:
        matches_idx = matches_from_scores(df, threshold, score_col)

    cand_by_s1 = {}
    for s1, mid in zip(candidates["s1_idx"], candidates["mid_idx"]):
        cand_by_s1.setdefault(int(s1), []).append(int(mid))
    cand_ids = to_entity_ids(cand_by_s1, s1_ids, mid_ids)
    match_ids = to_entity_ids(matches_idx, s1_ids, mid_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_pairs(out_dir / "candidate_pairs.tsv", s1_ids, cand_ids)
    write_matching_results(out_dir / "matching_results.tsv", s1_ids, match_ids)

    n_matches = sum(len(v) for v in match_ids.values())
    empty = sum(1 for v in match_ids.values() if not v)
    subset = all(set(v) <= set(cand_ids[k]) for k, v in match_ids.items())
    metrics = {
        "threshold": float(threshold),
        "one_to_one": bool(args.one_to_one),
        "rows": len(s1_ids),
        "candidates": int(len(candidates)),
        "matches": int(n_matches),
        "empty_rows": int(empty),
        "matches_subset_of_candidates": bool(subset),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
