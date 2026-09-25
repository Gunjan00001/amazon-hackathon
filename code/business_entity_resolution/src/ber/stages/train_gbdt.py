import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..decision import prune_top_k
from ..features import CLASSICAL_FEATURES, STRUCTURAL_FEATURES, build_pair_frame_prepared, classical_features, prepare_records, structural_features
from ..gbdt import predict_scores, train_model
from ..labels import positive_pairs


def load_mid(clean_dir, split):
    return pd.concat(
        [pd.read_parquet(clean_dir / f"{split}_s2.parquet"), pd.read_parquet(clean_dir / f"{split}_s3.parquet")],
        ignore_index=True,
    )


def feature_matrix(ps1, pmid, chunk, s1v=None, midv=None):
    frame = build_pair_frame_prepared(ps1, pmid, chunk)
    X = np.column_stack([classical_features(frame), structural_features(frame)])
    if s1v is not None:
        from ..vector_features import pair_vector_features
        i = chunk["s1_idx"].to_numpy()
        j = chunk["mid_idx"].to_numpy()
        X = np.column_stack([X, pair_vector_features(i, j, s1v["vec"], midv["vec"])])
    return X.astype("float32")


def subsample(cand, neg_ratio, train_cap, seed):
    y = cand["y"].to_numpy()
    pos_i = np.flatnonzero(y == 1)
    neg_i = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(neg_i)
    room = max(0, train_cap - len(pos_i))
    keep_neg = min(len(neg_i), neg_ratio * max(1, len(pos_i)), room)
    idx = np.concatenate([pos_i, neg_i[:keep_neg]])
    return cand.iloc[np.sort(idx)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default=str(config.ARTIFACT_DIR / "clean"))
    ap.add_argument("--block-dir", default=str(config.ARTIFACT_DIR / "block"))
    ap.add_argument("--out-dir", default=str(config.ARTIFACT_DIR / "gbdt"))
    ap.add_argument("--embed-dir", default="")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--neg-ratio", type=int, default=6)
    ap.add_argument("--train-cap", type=int, default=30_000_000)
    args = ap.parse_args()

    clean_dir, block_dir, out_dir = Path(args.clean_dir), Path(args.block_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = pd.read_parquet(clean_dir / "train_s1.parquet")
    mid = load_mid(clean_dir, "train")
    ps1, pmid = prepare_records(s1), prepare_records(mid)
    labels = pd.read_parquet(clean_dir / "labels.parquet")

    cand = pd.read_parquet(block_dir / "train_candidates.parquet")
    pos = pd.DataFrame(positive_pairs(s1["entity_id"].tolist(), mid["entity_id"].tolist(), labels),
                       columns=["s1_idx", "mid_idx"])
    pos["y"] = 1
    cand = cand.merge(pos, on=["s1_idx", "mid_idx"], how="left")
    cand["y"] = cand["y"].fillna(0).astype("int8")
    cand = subsample(cand, args.neg_ratio, args.train_cap, config.SEED)
    print(f"train pairs: {len(cand):,} positives {int(cand['y'].sum()):,}", flush=True)

    s1v = midv = None
    if args.embed_dir:
        from ..vector_features import load_vector_store
        store = load_vector_store(args.embed_dir, "train")
        s1v, midv = store["s1"], store["mid"]

    Xs, ys, gs = [], [], []
    t0 = time.time()
    for start in range(0, len(cand), args.chunk):
        ch = cand.iloc[start:start + args.chunk]
        Xs.append(feature_matrix(ps1, pmid, ch, s1v, midv))
        ys.append(ch["y"].to_numpy())
        gs.append(ch["s1_idx"].to_numpy())
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)
    print(f"features {X.shape} built in {time.time()-t0:.0f}s", flush=True)

    model, metrics = train_model(X, y, groups)
    model.booster_.save_model(str(out_dir / "model.txt"))
    metrics["train_pairs"] = int(len(cand))
    metrics["positives"] = int(y.sum())
    metrics["n_features"] = int(X.shape[1])
    metrics["feature_names"] = CLASSICAL_FEATURES + STRUCTURAL_FEATURES + (["vec_*"] * (X.shape[1] - 20))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)

    s1t = pd.read_parquet(clean_dir / "test_s1.parquet")
    midt = load_mid(clean_dir, "test")
    ps1t, pmidt = prepare_records(s1t), prepare_records(midt)
    tcand = pd.read_parquet(block_dir / "test_candidates.parquet")
    s1tv = midtv = None
    if args.embed_dir:
        from ..vector_features import load_vector_store
        store_t = load_vector_store(args.embed_dir, "test")
        s1tv, midtv = store_t["s1"], store_t["mid"]

    parts = []
    t0 = time.time()
    for start in range(0, len(tcand), args.chunk):
        ch = tcand.iloc[start:start + args.chunk]
        Xt = feature_matrix(ps1t, pmidt, ch, s1tv, midtv)
        parts.append(pd.DataFrame({
            "s1_idx": ch["s1_idx"].to_numpy(),
            "mid_idx": ch["mid_idx"].to_numpy(),
            "score": predict_scores(model, Xt).astype("float32"),
        }))
    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(out_dir / "test_scores.parquet", index=False)
    pruned = prune_top_k(scores, args.top_k)
    pruned.to_parquet(out_dir / "pruned_test.parquet", index=False)
    print(f"scored {len(scores):,} test pairs in {time.time()-t0:.0f}s; pruned {len(pruned):,}", flush=True)


if __name__ == "__main__":
    main()
