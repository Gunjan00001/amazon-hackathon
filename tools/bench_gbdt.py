"""Empirical LightGBM vs XGBoost comparison on sampled Business Entity Resolution pairs.

Builds a labeled pair sample from the training data:
  positives = ground-truth matches for a random sample of Source 1 entities
  negatives = blocking-style hard negatives (share country + a name token)

Then trains both GBDT implementations on identical features/splits and reports
pair-level AUC/AP and entity-level macro F0.5 (challenge metric, singleton rule applied).

Usage:
    .venv\\Scripts\\python.exe tools\\bench_gbdt.py
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(r"D:\Amazon project\DATA\student_resource\dataset")
OUT = Path(r"D:\Amazon project\tools\bench_results.json")
SEED = 42
N_S1 = 100_000
MAX_NEG_PER_S1 = int(os.environ.get("BENCH_MAX_NEG", "5"))
NEG_PICK_PROB = float(os.environ.get("BENCH_PICK_PROB", "0.05"))
TAG = os.environ.get("BENCH_TAG", "default")
CHUNK = 1_000_000

RNG = np.random.default_rng(SEED)


def norm_series(s):
    return (
        s.str.normalize("NFKC")
        .str.casefold()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def norm_one(s):
    return norm_series(pd.Series([s])).iloc[0]


def load_s1_sample():
    df = pd.read_csv(BASE / "train" / "train_source1.tsv", sep="\t", dtype=str,
                     keep_default_na=False, usecols=["entity_id", "business_name", "business_address", "country"])
    idx = RNG.choice(len(df), size=N_S1, replace=False)
    s1 = df.iloc[idx].reset_index(drop=True)
    s1["norm_name"] = norm_series(s1["business_name"])
    s1["fold_name"] = s1["norm_name"].str.normalize("NFKD")
    s1["norm_addr"] = norm_series(s1["business_address"])
    s1["fold_addr"] = s1["norm_addr"].str.normalize("NFKD")
    s1["country_n"] = s1["country"].str.strip().str.lower()
    first = s1["norm_name"].str.split(" ").str[0].fillna("")
    tok = np.where(first.str.len() >= 4, first, first.str.slice(0, 3))
    s1["key"] = s1["country_n"] + "|" + pd.Series(tok, index=s1.index)
    return s1


def load_positives(s1_ids):
    gt = pd.read_csv(BASE / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    gt = gt[gt["source1_entity_id"].isin(s1_ids)]
    pos = {}
    for s1, mids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        ids = [x for x in mids.split(",") if x]
        if ids:
            pos[s1] = ids
    return pos


def stream_negatives_and_texts(s1, positive_ids):
    key_index = {}
    for i, k in enumerate(s1["key"]):
        key_index.setdefault(k, []).append(i)

    texts = {}
    negatives = []
    neg_count = np.zeros(len(s1), dtype=np.int32)
    used = set()

    for src in ("train_source2.tsv", "train_source3.tsv"):
        is_s2 = src.endswith("source2.tsv")
        n_rows = 0
        t0 = time.time()
        for chunk in pd.read_csv(BASE / "train" / src, sep="\t", dtype=str,
                                 keep_default_na=False, chunksize=CHUNK):
            n_rows += len(chunk)
            ids = chunk["entity_id"].to_numpy()
            names = chunk["business_name"].to_numpy()
            addrs = chunk["business_address"].to_numpy()
            countries = chunk["country"].to_numpy()

            pos_mask = chunk["entity_id"].isin(positive_ids).to_numpy()
            if pos_mask.any():
                for i in np.nonzero(pos_mask)[0]:
                    texts[ids[i]] = (names[i], addrs[i], countries[i], is_s2)

            norm = norm_series(chunk["business_name"])
            first = norm.str.split(" ").str[0].fillna("")
            tok = np.where(first.str.len() >= 4, first, first.str.slice(0, 3))
            keys = (
                chunk["country"].str.strip().str.lower() + "|" + pd.Series(tok, index=chunk.index)
            ).to_numpy()

            pick = (RNG.random(len(chunk)) < NEG_PICK_PROB) & ~pos_mask
            for i in np.nonzero(pick)[0]:
                k = keys[i]
                bucket = key_index.get(k)
                if not bucket:
                    continue
                j = bucket[RNG.integers(len(bucket))]
                if neg_count[j] >= MAX_NEG_PER_S1:
                    continue
                mid = ids[i]
                if mid in used or mid in positive_ids:
                    continue
                used.add(mid)
                neg_count[j] += 1
                texts[mid] = (names[i], addrs[i], countries[i], is_s2)
                negatives.append((j, mid))
        print(f"  streamed {src}: {n_rows:,} rows in {time.time()-t0:.1f}s, "
              f"negatives so far {len(negatives):,}", flush=True)
    return texts, negatives


def build_pairs(s1, pos, texts, negatives):
    s1_norm_name = np.array(s1["norm_name"].tolist(), dtype=object)
    s1_fold_name = np.array(s1["fold_name"].tolist(), dtype=object)
    s1_norm_addr = np.array(s1["norm_addr"].tolist(), dtype=object)
    s1_fold_addr = np.array(s1["fold_addr"].tolist(), dtype=object)
    s1_country = np.array(s1["country_n"].tolist(), dtype=object)
    id_to_pos = {sid: i for i, sid in enumerate(s1["entity_id"].tolist())}

    rows = {"s1": [], "mid": [], "is_s2": [], "label": []}
    for j, mid in negatives:
        rows["s1"].append(j)
        rows["mid"].append(mid)
        rows["is_s2"].append(texts[mid][3])
        rows["label"].append(0)
    missing_pos = 0
    for sid, mids in pos.items():
        j = id_to_pos[sid]
        for mid in mids:
            t = texts.get(mid)
            if t is None:
                missing_pos += 1
                continue
            rows["s1"].append(j)
            rows["mid"].append(mid)
            rows["is_s2"].append(t[3])
            rows["label"].append(1)
    print(f"  positives missing text: {missing_pos}")

    df = pd.DataFrame(rows)
    uniq = df["mid"].unique()
    raw_n = np.array([texts[m][0] for m in uniq], dtype=object)
    raw_a = np.array([texts[m][1] for m in uniq], dtype=object)
    mdf = pd.DataFrame({"mid": uniq})
    mdf["n2"] = norm_series(pd.Series(raw_n)).to_numpy()
    mdf["a2"] = norm_series(pd.Series(raw_a)).to_numpy()
    mdf["n2_fold"] = mdf["n2"].str.normalize("NFKD").to_numpy()
    mdf["a2_fold"] = mdf["a2"].str.normalize("NFKD").to_numpy()
    mdf["c2"] = np.array([texts[m][2].strip().lower() for m in uniq], dtype=object)
    df = df.merge(mdf, on="mid", how="left")

    j = df["s1"].to_numpy()
    df["n1"] = s1_norm_name[j]
    df["n1_fold"] = s1_fold_name[j]
    df["a1"] = s1_norm_addr[j]
    df["a1_fold"] = s1_fold_addr[j]
    df["c1"] = s1_country[j]
    return df


def token_set(s):
    return set(s.split()) if s else set()


def compute_features(df):
    n1, n2 = df["n1_fold"].tolist(), df["n2_fold"].tolist()
    a1, a2 = df["a1_fold"].tolist(), df["a2_fold"].tolist()
    n1f, n2f = df["n1"].tolist(), df["n2"].tolist()
    feats = {k: np.empty(len(df), dtype=np.float32) for k in (
        "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_jaro",
        "name_exact", "name_jaccard", "name_len_diff",
        "addr_ratio", "addr_token_sort", "addr_jaccard", "addr_missing", "addr_len_diff",
        "same_country", "is_s2")}
    t0 = time.time()
    for i in range(len(df)):
        x1, x2 = n1[i], n2[i]
        y1, y2 = a1[i], a2[i]
        feats["name_ratio"][i] = fuzz.ratio(x1, x2)
        feats["name_partial"][i] = fuzz.partial_ratio(x1, x2)
        feats["name_token_sort"][i] = fuzz.token_sort_ratio(x1, x2)
        feats["name_token_set"][i] = fuzz.token_set_ratio(x1, x2)
        feats["name_jaro"][i] = JaroWinkler.similarity(x1, x2)
        feats["name_exact"][i] = 1.0 if n1f[i] == n2f[i] and n1f[i] else 0.0
        s1t, s2t = token_set(n1f[i]), token_set(n2f[i])
        feats["name_jaccard"][i] = len(s1t & s2t) / len(s1t | s2t) if (s1t or s2t) else 0.0
        feats["name_len_diff"][i] = abs(len(n1f[i]) - len(n2f[i]))
        feats["addr_ratio"][i] = fuzz.ratio(y1, y2) if (y1 and y2) else -1.0
        feats["addr_token_sort"][i] = fuzz.token_sort_ratio(y1, y2) if (y1 and y2) else -1.0
        t1, t2 = token_set(a1[i]), token_set(a2[i])
        feats["addr_jaccard"][i] = len(t1 & t2) / len(t1 | t2) if (t1 and t2) else -1.0
        feats["addr_missing"][i] = 1.0 if (not a1[i] or not a2[i]) else 0.0
        feats["addr_len_diff"][i] = abs(len(a1[i]) - len(a2[i]))
        feats["same_country"][i] = 1.0 if df["c1"].iat[i] == df["c2"].iat[i] else 0.0
        feats["is_s2"][i] = 1.0 if df["is_s2"].iat[i] else 0.0
        if i and i % 200_000 == 0:
            print(f"    features {i:,}/{len(df):,} ({time.time()-t0:.0f}s)", flush=True)
    F = pd.DataFrame(feats)
    F["label"] = df["label"].to_numpy()
    F["group"] = df["s1"].to_numpy()
    return F


def entity_macro_f05(groups, labels, probs, threshold):
    f_scores = []
    for g in np.unique(groups):
        m = groups == g
        y, p = labels[m], probs[m] >= threshold
        tp = int((y & p).sum())
        n_pred = int(p.sum())
        n_true = int(y.sum())
        if n_true == 0:
            f_scores.append(1.0 if n_pred == 0 else 0.0)
            continue
        if n_pred == 0:
            f_scores.append(0.0)
            continue
        precision = tp / n_pred
        recall = tp / n_true
        f_scores.append(0.0 if precision == 0 else (1.25 * precision * recall) / (0.25 * precision + recall))
    return float(np.mean(f_scores))


def best_entity_f05(groups, labels, probs):
    best = (0.0, 0.0)
    for th in np.arange(0.05, 0.951, 0.025):
        f = entity_macro_f05(groups, labels, probs, th)
        if f > best[0]:
            best = (f, float(th))
    return best


def main():
    print("loading S1 sample ...", flush=True)
    s1 = load_s1_sample()
    print("loading positives ...", flush=True)
    pos = load_positives(set(s1["entity_id"]))
    positive_ids = {m for mids in pos.values() for m in mids}
    print(f"  sampled S1: {len(s1):,}; S1 with matches: {len(pos):,}; positives: {len(positive_ids):,}")

    print("streaming sources for negatives + texts ...", flush=True)
    texts, negatives = stream_negatives_and_texts(s1, positive_ids)
    print(f"  negatives: {len(negatives):,}; stored texts: {len(texts):,}")

    flat = [len(v) for v in pos.values()]
    print(f"  positives per matched S1: mean {np.mean(flat):.2f}")

    df = build_pairs(s1, pos, texts, negatives)
    print(f"pairs: {len(df):,} (positives {int(df['label'].sum()):,})")
    F = compute_features(df)

    groups = F["group"].to_numpy()
    unique_groups = np.unique(groups)
    RNG.shuffle(unique_groups)
    n_val = int(0.2 * len(unique_groups))
    val_groups = set(unique_groups[:n_val].tolist())
    val_mask = np.isin(groups, list(val_groups))
    tr_mask = ~val_mask
    X = F.drop(columns=["label", "group"])
    print(f"train pairs {tr_mask.sum():,} | val pairs {val_mask.sum():,} | "
          f"val entities {len(val_groups):,}")

    import lightgbm as lgb
    import xgboost as xgb

    results = {}
    for name in ("lightgbm", "xgboost"):
        print(f"\ntraining {name} ...", flush=True)
        t0 = time.time()
        if name == "lightgbm":
            model = lgb.LGBMClassifier(
                objective="binary", n_estimators=1000, learning_rate=0.05, num_leaves=63,
                min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                n_jobs=8, random_state=SEED, verbosity=-1,
            )
            model.fit(
                X[tr_mask], F["label"][tr_mask],
                eval_set=[(X[val_mask], F["label"][val_mask])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            best_iter = model.best_iteration_
        else:
            model = xgb.XGBClassifier(
                tree_method="hist", n_estimators=1000, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, n_jobs=8, random_state=SEED,
                eval_metric="auc", early_stopping_rounds=50,
            )
            model.fit(X[tr_mask], F["label"][tr_mask],
                      eval_set=[(X[val_mask], F["label"][val_mask])], verbose=False)
            best_iter = model.best_iteration
        fit_time = time.time() - t0

        t0 = time.time()
        probs = model.predict_proba(X[val_mask])[:, 1]
        infer_time = time.time() - t0
        y = F["label"][val_mask].to_numpy()
        g = groups[val_mask]
        auc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
        f05, th = best_entity_f05(g, y, probs)
        results[name] = {
            "best_iteration": int(best_iter), "fit_seconds": round(fit_time, 1),
            "infer_seconds": round(infer_time, 2),
            "pair_auc": round(float(auc), 5), "average_precision": round(float(ap), 5),
            "best_macro_f05": round(f05, 5), "best_threshold": th,
        }
        print(f"  {name}: AUC={auc:.5f} AP={ap:.5f} macroF0.5={f05:.5f} @th={th:.3f} "
              f"fit={fit_time:.1f}s infer={infer_time:.2f}s")

    all_results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    all_results[TAG] = {
        "config": {"n_s1": N_S1, "max_neg_per_s1": MAX_NEG_PER_S1,
                   "neg_pick_prob": NEG_PICK_PROB, "seed": SEED},
        "pairs": int(len(F)),
        "positives": int(F["label"].sum()),
        "val_pairs": int(val_mask.sum()),
        "val_entities": int(len(val_groups)),
        "models": results,
    }
    OUT.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT} (scenario: {TAG})")


if __name__ == "__main__":
    main()
