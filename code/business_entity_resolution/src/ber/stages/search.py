import argparse
import json
from pathlib import Path

from .. import config
from ..search import finalize, run_search

SMOKE_SPACE = {
    "max_postings": [2000],
    "max_candidates": [40],
    "num_leaves": [63],
    "learning_rate": [0.1],
    "top_k": [10],
    "one_to_one": [False],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default=str(config.ARTIFACT_DIR / "clean"))
    ap.add_argument("--out-dir", default=str(config.ARTIFACT_DIR / "search"))
    ap.add_argument("--search-dir", default="")
    ap.add_argument("--s1-limit", type=int, default=120000)
    ap.add_argument("--n-block", type=int, default=4)
    ap.add_argument("--n-gbdt", type=int, default=8)
    ap.add_argument("--budget", type=int, default=7200)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--holdout-country", default="")
    ap.add_argument("--train-cap", type=int, default=10_000_000)
    ap.add_argument("--neg-ratio", type=int, default=6)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--finalize-s1-limit", type=int, default=500000)
    ap.add_argument("--smoke-first", action="store_true")
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--no-finalize", action="store_true")
    args = ap.parse_args()

    clean_dir = Path(args.clean_dir)
    out_dir = Path(args.out_dir)

    if args.finalize_only:
        search_dir = Path(args.search_dir or out_dir)
        best = json.loads((search_dir / "best.json").read_text(encoding="utf-8"))
        print("BEST", json.dumps(best, indent=2, default=str), flush=True)
        out = finalize(clean_dir, out_dir, best, val_frac=args.val_frac, train_cap=args.train_cap,
                       neg_ratio=args.neg_ratio, s1_limit=args.finalize_s1_limit,
                       test_limit=args.test_limit, keys_dir=search_dir)
        print("OUT", json.dumps(out, indent=2, default=str), flush=True)
        return

    if args.smoke_first:
        smoke_dir = out_dir / "smoke"
        print("[smoke] start", flush=True)
        best_s = run_search(clean_dir, smoke_dir, space=SMOKE_SPACE, n_block=1, n_gbdt=1,
                            s1_limit=3000, budget_s=1200, keys_dir=smoke_dir)
        out_s = finalize(clean_dir, smoke_dir, best_s, train_cap=200000, neg_ratio=6,
                         s1_limit=3000, test_limit=3000, keys_dir=smoke_dir)
        print("[smoke] OK", json.dumps(out_s, default=str), flush=True)

    best = run_search(
        clean_dir, out_dir,
        n_block=args.n_block, n_gbdt=args.n_gbdt, s1_limit=args.s1_limit,
        val_frac=args.val_frac, holdout_country=(args.holdout_country or None),
        budget_s=args.budget, keys_dir=out_dir,
    )
    print("BEST", json.dumps(best, indent=2, default=str), flush=True)
    if args.search_only or args.no_finalize:
        return
    out = finalize(
        clean_dir, out_dir, best, val_frac=args.val_frac, train_cap=args.train_cap,
        neg_ratio=args.neg_ratio, s1_limit=args.finalize_s1_limit, test_limit=args.test_limit,
        keys_dir=out_dir,
    )
    print("OUT", json.dumps(out, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
