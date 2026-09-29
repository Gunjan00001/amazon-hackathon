import gc
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .blocking import (blocking_recall, build_index, generate_candidates, generate_candidates_from_keys,
                       load_keys, prepare_keys, save_keys, score_keys)
from .decision import matches_from_scores, one_to_one_assign, prune_top_k, to_entity_ids
from collections import defaultdict

from .evaluate import entity_macro_counts, entity_macro_f05, grouped_split
from .features import CLASSICAL_FEATURES, STRUCTURAL_FEATURES, build_pair_frame_prepared, classical_features, prepare_records, structural_features
from .gbdt import PARAMS, predict_scores, train_model
from .io_tsv import write_candidate_pairs, write_matching_results
from .labels import positive_pairs
from .score import entity_f05

READ_COLS = ["entity_id", "business_name", "business_address", "country"]

DEFAULT_SPACE = {
    "max_postings": [2000, 20000],
    "max_candidates": [40, 100],
    "num_leaves": [63, 127],
    "learning_rate": [0.05, 0.1],
    "top_k": [10, 20],
    "one_to_one": [False, True],
}

BLOCK_KEYS = ["max_postings", "max_candidates"]
MODEL_KEYS = ["num_leaves", "learning_rate", "top_k", "one_to_one"]

DEFAULT_GRID = np.arange(0.05, 0.96, 0.025)


def expand_space(space):
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]


def _dedup(dicts):
    seen = set()
    out = []
    for d in dicts:
        key = tuple(sorted(d.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def sample_configs(configs, n, seed):
    if n is None or n >= len(configs):
        return list(configs)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(configs), size=n, replace=False))
    return [configs[i] for i in idx]


def successive_halving(configs, evaluate, keep_frac=0.5, min_keep=1, max_rounds=4):
    current = list(configs)
    if not current:
        return []
    for _ in range(max_rounds):
        scored = []
        for c in current:
            sc = dict(c)
            sc["score"] = float(evaluate(c))
            scored.append(sc)
        scored.sort(key=lambda d: d["score"], reverse=True)
        keep = max(min_keep, int(math.ceil(len(scored) * keep_frac)))
        current = scored[:keep]
        if len(current) <= min_keep:
            break
    return current


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _load_train(clean_dir):
    clean_dir = Path(clean_dir)
    s1 = pd.read_parquet(clean_dir / "train_s1.parquet", columns=READ_COLS)
    mid = pd.concat([
        pd.read_parquet(clean_dir / "train_s2.parquet", columns=READ_COLS),
        pd.read_parquet(clean_dir / "train_s3.parquet", columns=READ_COLS),
    ], ignore_index=True)
    labels = pd.read_parquet(clean_dir / "labels.parquet")
    return s1, mid, labels


def _load_test(clean_dir):
    clean_dir = Path(clean_dir)
    s1 = pd.read_parquet(clean_dir / "test_s1.parquet", columns=READ_COLS)
    mid = pd.concat([
        pd.read_parquet(clean_dir / "test_s2.parquet", columns=READ_COLS),
        pd.read_parquet(clean_dir / "test_s3.parquet", columns=READ_COLS),
    ], ignore_index=True)
    return s1, mid


def _sample_s1(s1, limit, seed):
    if not limit or limit >= len(s1):
        return s1.reset_index(drop=True)
    idx = np.sort(np.random.default_rng(seed).choice(len(s1), size=limit, replace=False))
    return s1.iloc[idx].reset_index(drop=True)


def _macro_from_pred(groups, labels, pred):
    return float(np.mean([entity_f05(labels[groups == g], pred[groups == g]) for g in np.unique(groups)]))


def _candidate_frame(s1, mid, labels, bcfg, s1_keys=None, mid_keys=None):
    if s1_keys is None:
        s1_keys = prepare_keys(s1)
    if mid_keys is None:
        mid_keys = prepare_keys(mid)
    pairs = generate_candidates_from_keys(s1_keys, mid_keys, bcfg["max_candidates"], bcfg["max_postings"])
    pos = pd.DataFrame(
        positive_pairs(s1["entity_id"].tolist(), mid["entity_id"].tolist(), labels),
        columns=["s1_idx", "mid_idx"],
    )
    pos["y"] = 1
    pairs = pairs.merge(pos, on=["s1_idx", "mid_idx"], how="left")
    pairs["y"] = pairs["y"].fillna(0).astype("int8")
    return pairs


def _subsample(pairs, neg_ratio, cap, seed):
    y = pairs["y"].to_numpy()
    pos_i = np.flatnonzero(y == 1)
    neg_i = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(neg_i)
    room = max(0, cap - len(pos_i))
    keep_neg = min(len(neg_i), neg_ratio * max(1, len(pos_i)), room)
    idx = np.concatenate([pos_i, neg_i[:keep_neg]])
    return pairs.iloc[np.sort(idx)].reset_index(drop=True)


def _features(frame):
    return np.column_stack([classical_features(frame), structural_features(frame)]).astype("float32")


def _build_features(ps1, pmid, pairs, chunk=2_000_000):
    parts = []
    for start in range(0, len(pairs), chunk):
        ch = pairs.iloc[start:start + chunk]
        parts.append(_features(build_pair_frame_prepared(ps1, pmid, ch)))
    if not parts:
        return np.zeros((0, len(CLASSICAL_FEATURES) + len(STRUCTURAL_FEATURES)), dtype="float32")
    return np.vstack(parts)


def _best_decision(vdf, pruned, y, groups, one_to_one, grid):
    s1 = vdf["s1_idx"].to_numpy()
    mid = vdf["mid_idx"].to_numpy()
    best = (0.0, float(grid[0]))
    for th in grid:
        matches = one_to_one_assign(pruned, float(th)) if one_to_one else matches_from_scores(pruned, float(th))
        keys = {(int(a), int(b)) for a, ms in matches.items() for b in ms}
        pred = np.fromiter(((int(a), int(b)) in keys for a, b in zip(s1, mid)), dtype=bool, count=len(s1))
        m = _macro_from_pred(groups, y, pred)
        if m > best[0]:
            best = (m, float(th))
    return best


def search_blocking(s1, mid, labels, block_cfgs, s1_keys=None, mid_keys=None, true_pairs=None):
    if true_pairs is None:
        true_pairs = set(positive_pairs(s1["entity_id"].tolist(), mid["entity_id"].tolist(), labels))
    results = []
    for bcfg in block_cfgs:
        pairs = _candidate_frame(s1, mid, labels, bcfg, s1_keys, mid_keys)
        cand = set(zip(pairs["s1_idx"].astype(int).tolist(), pairs["mid_idx"].astype(int).tolist()))
        hit = len(cand & true_pairs)
        recall = hit / len(true_pairs) if true_pairs else 0.0
        results.append({
            "config": bcfg,
            "recall_ceiling": float(recall),
            "pairs": int(len(pairs)),
            "positives": hit,
            "true_positives": len(true_pairs),
        })
    results.sort(key=lambda r: (r["recall_ceiling"], -r["pairs"]), reverse=True)
    return results


def _tune_entity_decision(vdf, true_pairs, true_len, va_idx, n_s1, top_k, one_to_one, grid):
    if len(vdf) == 0:
        z = np.zeros(len(va_idx), dtype="int64")
        return float(entity_macro_counts(true_len[va_idx], z, z)), float(grid[0]), False
    pruned = prune_top_k(vdf, top_k) if top_k else vdf
    s = pruned["s1_idx"].to_numpy().astype("int64")
    mid = pruned["mid_idx"].to_numpy().astype("int64")
    sc = pruned["score"].to_numpy()
    istrue = np.fromiter(((int(a), int(b)) in true_pairs for a, b in zip(s, mid)), dtype=bool, count=len(s))
    best = (0.0, float(grid[0]))
    for th in grid:
        msk = sc >= float(th)
        n_pred = np.bincount(s[msk], minlength=n_s1)
        n_hit = np.bincount(s[msk & istrue], minlength=n_s1)
        f = entity_macro_counts(true_len[va_idx], n_pred[va_idx], n_hit[va_idx])
        if f > best[0]:
            best = (float(f), float(th))
    m, th = best
    if one_to_one:
        matches = one_to_one_assign(pruned, th)
        n_pred = np.zeros(n_s1, dtype="int64")
        n_hit = np.zeros(n_s1, dtype="int64")
        for a, ms in matches.items():
            n_pred[a] = len(ms)
            n_hit[a] = sum(1 for x in ms if (int(a), int(x)) in true_pairs)
        f = entity_macro_counts(true_len[va_idx], n_pred[va_idx], n_hit[va_idx])
        if f > m:
            return float(f), float(th), True
    return float(m), float(th), False


def run_search(clean_dir, out_dir, space=None, n_block=6, n_gbdt=8, s1_limit=120000,
               val_frac=0.2, seed=config.SEED, holdout_country=None, budget_s=3600,
               shortlist=3, train_cap=3_000_000, neg_ratio=6, keys_dir=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    space = space or DEFAULT_SPACE
    all_cfgs = expand_space(space)
    block_cfgs = sample_configs(_dedup([{k: c[k] for k in BLOCK_KEYS} for c in all_cfgs]), n_block, seed)
    model_cfgs = sample_configs(_dedup([{k: c[k] for k in MODEL_KEYS} for c in all_cfgs]), n_gbdt, seed + 1)
    grid = DEFAULT_GRID

    s1, mid, labels = _load_train(clean_dir)
    s1s = _sample_s1(s1, s1_limit, seed)
    ps1, pmid = prepare_records(s1s), prepare_records(mid)
    s1_keys = prepare_keys(s1s)
    mid_keys = prepare_keys(mid)
    if keys_dir:
        save_keys(mid_keys, Path(keys_dir) / "keys" / "mid_keys.parquet")

    true_pairs = set(positive_pairs(s1s["entity_id"].tolist(), mid["entity_id"].tolist(), labels))
    true_by_s1 = defaultdict(set)
    for a, b in true_pairs:
        true_by_s1[int(a)].add(int(b))
    n_s1 = len(s1s)
    all_idx = np.arange(n_s1)
    tr_mask, va_mask = grouped_split(all_idx, val_frac, seed)
    tr_s1 = set(all_idx[tr_mask].tolist())
    va_idx = all_idx[va_mask]
    va_s1_set = set(va_idx.tolist())
    true_len = np.fromiter((len(true_by_s1.get(i, ())) for i in range(n_s1)), dtype="int64", count=n_s1)

    trials = {"blocking": [], "model": [], "best": None}

    def checkpoint():
        (out_dir / "trials.json").write_text(json.dumps(_jsonable(trials), indent=2), encoding="utf-8")

    block_budget = budget_s * 0.5
    bres = []
    for bcfg in block_cfgs:
        rec = search_blocking(s1s, mid, labels, [bcfg], s1_keys, mid_keys, true_pairs)[0]
        bres.append(rec)
        trials["blocking"].append(rec)
        checkpoint()
        print(f"[block] {json.dumps(_jsonable(rec))} t={time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > block_budget:
            break
    if not bres:
        raise RuntimeError("no blocking configs evaluated")
    bres.sort(key=lambda r: (r["recall_ceiling"], -r["pairs"]), reverse=True)
    shortlisted = bres[:shortlist]
    print(f"[search] shortlist={json.dumps(_jsonable([r['config'] for r in shortlisted]))}", flush=True)

    mres = []
    for brec in shortlisted:
        if mres and time.time() - t0 > budget_s:
            break
        pairs = _candidate_frame(s1s, mid, labels, brec["config"], s1_keys, mid_keys)
        sub = _subsample(pairs[pairs["s1_idx"].isin(tr_s1)], neg_ratio, train_cap, seed)
        if len(sub) == 0 or int(sub["y"].sum()) == 0:
            continue
        X = _build_features(ps1, pmid, sub)
        y = sub["y"].to_numpy()
        groups = sub["s1_idx"].to_numpy()
        va = pairs[pairs["s1_idx"].isin(va_s1_set)].reset_index(drop=True)
        if len(va):
            Xva = _build_features(ps1, pmid, va)
            va_s1 = va["s1_idx"].to_numpy()
            va_mid = va["mid_idx"].to_numpy()
        else:
            Xva = np.zeros((0, len(CLASSICAL_FEATURES) + len(STRUCTURAL_FEATURES)), dtype="float32")
            va_s1 = np.zeros(0, dtype="int64")
            va_mid = np.zeros(0, dtype="int64")
        for gcfg in model_cfgs:
            if mres and time.time() - t0 > budget_s:
                break
            params = dict(PARAMS, num_leaves=int(gcfg["num_leaves"]), learning_rate=float(gcfg["learning_rate"]))
            model, _ = train_model(X, y, groups, params=params)
            scores = predict_scores(model, Xva) if len(Xva) else np.zeros(0, dtype="float32")
            vdf = pd.DataFrame({"s1_idx": va_s1, "mid_idx": va_mid, "score": scores})
            m, th, oo = _tune_entity_decision(vdf, true_pairs, true_len, va_idx, n_s1,
                                              int(gcfg["top_k"]), bool(gcfg["one_to_one"]), grid)
            rec = {
                "blocking": brec["config"],
                "model": {**gcfg, "one_to_one": bool(oo)},
                "macro_f05": float(m),
                "threshold": float(th),
                "recall_ceiling": float(brec["recall_ceiling"]),
            }
            mres.append(rec)
            trials["model"].append(rec)
            checkpoint()
            print(f"[model] {json.dumps(_jsonable(rec))} t={time.time()-t0:.0f}s", flush=True)
    if not mres:
        raise RuntimeError("no model configs evaluated")
    mres.sort(key=lambda r: r["macro_f05"], reverse=True)
    best = mres[0]
    trials["best"] = best
    checkpoint()
    (out_dir / "best.json").write_text(json.dumps(_jsonable(best), indent=2), encoding="utf-8")
    del mid_keys, s1_keys, ps1, pmid
    gc.collect()
    return best


def finalize(clean_dir, out_dir, best, val_frac=0.2, seed=config.SEED,
             train_cap=20_000_000, neg_ratio=6, s1_limit=0, test_limit=0, keys_dir=None,
             test_chunk=200000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bcfg = best["blocking"]
    gcfg = best["model"]
    print(f"[finalize] start blocking={json.dumps(_jsonable(bcfg))} model={json.dumps(_jsonable(gcfg))}", flush=True)

    s1, mid, labels = _load_train(clean_dir)
    if s1_limit:
        s1 = s1.head(s1_limit).reset_index(drop=True)
    mid_key_path = Path(keys_dir) / "keys" / "mid_keys.parquet" if keys_dir else None
    if mid_key_path and mid_key_path.exists():
        mid_keys = load_keys(mid_key_path)
        print(f"[finalize] loaded mid keys from {mid_key_path}", flush=True)
    else:
        mid_keys = prepare_keys(mid)
    pairs = _candidate_frame(s1, mid, labels, bcfg, prepare_keys(s1), mid_keys)
    del mid_keys
    gc.collect()
    sub = _subsample(pairs, neg_ratio, train_cap, seed)
    del pairs
    gc.collect()
    ps1, pmid = prepare_records(s1), prepare_records(mid)
    X = _build_features(ps1, pmid, sub)
    y = sub["y"].to_numpy()
    groups = sub["s1_idx"].to_numpy()
    params = dict(PARAMS, num_leaves=int(gcfg["num_leaves"]), learning_rate=float(gcfg["learning_rate"]))
    model, metrics = train_model(X, y, groups, params=params)
    model.booster_.save_model(str(out_dir / "model.txt"))
    metrics.update({"blocking": bcfg, "model": gcfg, "train_pairs": int(len(sub)), "positives": int(y.sum())})
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    print(f"[finalize] trained model pair_auc={metrics.get('pair_auc')} macro_f05={metrics.get('best_macro_f05')}", flush=True)
    del X, y, groups, sub, ps1, pmid, s1, mid
    gc.collect()

    ts1, tmid = _load_test(clean_dir)
    if test_limit:
        ts1 = ts1.head(test_limit).reset_index(drop=True)
    s1_ids = ts1["entity_id"].tolist()
    mid_ids = tmid["entity_id"].tolist()
    ptmid = prepare_records(tmid)
    cap, postings = bcfg["max_candidates"], bcfg["max_postings"]
    test_index = build_index(prepare_keys(tmid), postings)
    gc.collect()
    print("[finalize] test index built", flush=True)
    score_parts = []
    for start in range(0, len(ts1), test_chunk):
        chunk = ts1.iloc[start:start + test_chunk].reset_index(drop=True)
        cpairs = score_keys(prepare_keys(chunk), test_index, cap)
        if len(cpairs):
            frame = build_pair_frame_prepared(prepare_records(chunk), ptmid, cpairs)
            sc = predict_scores(model, _features(frame)).astype("float32")
            score_parts.append(pd.DataFrame({
                "s1_idx": cpairs["s1_idx"].to_numpy() + start,
                "mid_idx": cpairs["mid_idx"].to_numpy(),
                "score": sc,
            }))
            del frame
        del cpairs
        gc.collect()
        print(f"[finalize] scored test S1 {start + len(chunk)}/{len(ts1)}", flush=True)
    del test_index, ptmid
    gc.collect()
    sdf = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame(
        {"s1_idx": [], "mid_idx": [], "score": []})
    sdf.to_parquet(out_dir / "test_scores.parquet", index=False)
    pruned = prune_top_k(sdf, int(gcfg["top_k"]))
    pruned.to_parquet(out_dir / "pruned_test.parquet", index=False)

    threshold = float(best.get("threshold", 0.5))
    one_to_one = bool(gcfg["one_to_one"])
    matches_idx = one_to_one_assign(pruned, threshold) if one_to_one else matches_from_scores(pruned, threshold)
    cand_by_s1 = {}
    for a, b in zip(pruned["s1_idx"], pruned["mid_idx"]):
        cand_by_s1.setdefault(int(a), []).append(int(b))
    cand_ids = to_entity_ids(cand_by_s1, s1_ids, mid_ids)
    match_ids = to_entity_ids(matches_idx, s1_ids, mid_ids)
    write_candidate_pairs(out_dir / "candidate_pairs.tsv", s1_ids, cand_ids)
    write_matching_results(out_dir / "matching_results.tsv", s1_ids, match_ids)

    out_metrics = {
        "threshold": threshold,
        "one_to_one": one_to_one,
        "rows": len(s1_ids),
        "candidates": int(len(pruned)),
        "matches": int(sum(len(v) for v in match_ids.values())),
        "empty_rows": int(sum(1 for v in match_ids.values() if not v)),
    }
    (out_dir / "out_metrics.json").write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")
    return out_metrics
