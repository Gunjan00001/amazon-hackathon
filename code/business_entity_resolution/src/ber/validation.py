import json
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ber.pairs import grouped_split

SPLIT = "valfull"


def _reports(cfg):
    path = Path(cfg.data_dir) / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def val_s1_ids_path(cfg):
    return _reports(cfg) / f"{SPLIT}_s1_ids.parquet"


def val_s1_ids(cfg, val_frac=0.2):
    path = val_s1_ids_path(cfg)
    if path.exists():
        return pd.read_parquet(path)["s1_id"].to_numpy()
    frame = pd.read_parquet(
        Path(cfg.data_dir) / "features" / "train.parquet", columns=["s1_id"]
    )
    _, val_mask = grouped_split(frame, val_frac, cfg.seed)
    ids = frame.loc[val_mask, "s1_id"].drop_duplicates().sort_values().reset_index(drop=True)
    ids.to_frame("s1_id").to_parquet(path, index=False)
    return ids.to_numpy()


def build_valfull_pairs(cfg, val_frac=0.2):
    data = Path(cfg.data_dir)
    ids = val_s1_ids(cfg, val_frac)
    candidates = (data / "candidates" / "train_candidates.parquet").as_posix()
    out = data / "pairs" / f"{SPLIT}_pairs.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(data / 'tmp').as_posix()}'")
    con.execute("PRAGMA max_temp_directory_size='50GiB'")
    con.execute(f"CREATE TEMP TABLE valids AS SELECT s1_id FROM read_parquet('{val_s1_ids_path(cfg).as_posix()}')")
    con.execute(
        f"COPY (SELECT c.s1_id, c.cand_id, c.is_s2, c.pass_id, c.block_score "
        f"FROM read_parquet('{candidates}') c SEMI JOIN valids v ON v.s1_id = c.s1_id) "
        f"TO '{out.as_posix()}' (FORMAT PARQUET)"
    )
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    con.close()
    return {"pairs": int(rows), "val_s1": int(len(ids)), "path": str(out)}


def _country_of_source1(cfg, split="train"):
    path = Path(cfg.dataset_dir) / split / f"{split}_source1.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id", "country"])
    return dict(zip(frame["entity_id"], frame["country"].str.strip().str.lower()))


def _macro_from_counts(ntrue, npred, tp):
    n = len(ntrue)
    scores = np.ones(n, dtype=np.float64)
    has_true = ntrue > 0
    no_pred = npred == 0
    scores[has_true & no_pred] = 0.0
    valid = has_true & ~no_pred
    if valid.any():
        precision = tp[valid] / npred[valid]
        recall = tp[valid] / ntrue[valid]
        f = np.zeros(valid.sum(), dtype=np.float64)
        nz = precision > 0
        f[nz] = (1.25 * precision[nz] * recall[nz]) / (0.25 * precision[nz] + recall[nz])
        scores[valid] = f
    scores[~has_true] = 0.0
    scores[(~has_true) & no_pred] = 1.0
    return scores


def _bin_counts(con, source_sql, lo, step, n_bins):
    return con.execute(
        f"""
        SELECT s1_id, bin, is_true, COUNT(*) AS cnt FROM (
            SELECT p.s1_id, p.cand_id,
                   CASE WHEN t.cand_id IS NOT NULL THEN 1 ELSE 0 END AS is_true,
                   LEAST(CAST(FLOOR((p.prob - {lo}) / {step} + 1e-9) AS INTEGER), {n_bins - 1}) AS bin
            FROM {source_sql} p
            LEFT JOIN truth t ON t.s1_id = p.s1_id AND t.cand_id = p.cand_id
            WHERE p.prob >= {lo}
        ) GROUP BY s1_id, bin, is_true
        """
    ).fetchall()


def _pivot(counts, ids, n_bins):
    index = {sid: i for i, sid in enumerate(ids)}
    arr = np.zeros((len(ids), n_bins, 2), dtype=np.int64)
    for s1_id, bin_, is_true, cnt in counts:
        i = index.get(s1_id)
        if i is None:
            continue
        arr[i, int(bin_), int(is_true)] += int(cnt)
    return arr


def _macro_at_bins(arr, ntrue, thresholds):
    total = arr[:, :, 0] + arr[:, :, 1]
    tp_cum = np.cumsum(arr[:, ::-1, 1], axis=1)[:, ::-1]
    npred_cum = np.cumsum(total[:, ::-1], axis=1)[:, ::-1]
    results = []
    for k in range(len(thresholds)):
        scores = _macro_from_counts(ntrue, npred_cum[:, k], tp_cum[:, k])
        results.append((float(scores.mean()), scores))
    return results


def _score_preds(cfg, preds_glob, lo=0.2, hi=0.95, step=0.025, valids_path=None, pairs_path=None, bootstrap=500):
    data = Path(cfg.data_dir)
    gt = (data / "processed" / "train_ground_truth.parquet").as_posix()
    s1_parquet = (data / "processed" / "train_source1.parquet").as_posix()
    valids_path = Path(valids_path) if valids_path else val_s1_ids_path(cfg)
    pairs_path = Path(pairs_path) if pairs_path else data / "pairs" / f"{SPLIT}_pairs.parquet"
    thresholds = np.round(np.arange(lo, hi + 1e-9, step), 4).tolist()
    n_bins = len(thresholds)

    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(data / 'tmp').as_posix()}'")
    con.execute("PRAGMA max_temp_directory_size='50GiB'")
    con.execute(f"CREATE TEMP TABLE valids AS SELECT s1_id FROM read_parquet('{valids_path.as_posix()}')")
    con.execute(
        f"""
        CREATE TEMP TABLE truth AS
        SELECT g.source1_entity_id AS s1_id, UNNEST(string_split(g.matched_entity_ids, ',')) AS cand_id
        FROM read_parquet('{gt}') g SEMI JOIN valids v ON v.s1_id = g.source1_entity_id
        WHERE g.matched_entity_ids <> ''
        """
    )
    ids = con.execute("SELECT s1_id FROM valids ORDER BY s1_id").fetchall()
    ids = [r[0] for r in ids]
    ntrue_rows = con.execute("SELECT s1_id, COUNT(*) FROM truth GROUP BY s1_id").fetchall()
    ntrue_map = {s1: int(c) for s1, c in ntrue_rows}
    ntrue = np.array([ntrue_map.get(sid, 0) for sid in ids], dtype=np.int64)

    ceiling = con.execute(
        f"""
        SELECT COUNT(*) AS total, COUNT(c.cand_id) AS hit
        FROM truth t LEFT JOIN (SELECT DISTINCT s1_id, cand_id FROM read_parquet('{pairs_path.as_posix()}')) c
        ON c.s1_id = t.s1_id AND c.cand_id = t.cand_id
        """
    ).fetchone()
    total_truth, hit_truth = int(ceiling[0]), int(ceiling[1])

    preds_sql = f"(SELECT p.s1_id, p.cand_id, p.prob FROM read_parquet('{preds_glob}') p SEMI JOIN valids v ON v.s1_id = p.s1_id)"
    thr_counts = _bin_counts(con, preds_sql, lo, step, n_bins)
    con.execute(
        f"""
        CREATE TEMP TABLE best AS
        SELECT s1_id, cand_id, prob FROM (
            SELECT p.s1_id, p.cand_id, p.prob,
                   ROW_NUMBER() OVER (PARTITION BY p.cand_id ORDER BY p.prob DESC, p.s1_id ASC) AS rn
            FROM {preds_sql} p
        ) WHERE rn = 1
        """
    )
    oto_counts = _bin_counts(con, "best", lo, step, n_bins)
    con.close()

    arr_thr = _pivot(thr_counts, ids, n_bins)
    arr_oto = _pivot(oto_counts, ids, n_bins)
    thr_results = _macro_at_bins(arr_thr, ntrue, thresholds)
    oto_results = _macro_at_bins(arr_oto, ntrue, thresholds)

    best_thr = max(range(n_bins), key=lambda k: thr_results[k][0])
    best_oto = max(range(n_bins), key=lambda k: oto_results[k][0])
    use_oto = oto_results[best_oto][0] > thr_results[best_thr][0]
    chosen_scores = oto_results[best_oto][1] if use_oto else thr_results[best_thr][1]

    rng = np.random.default_rng(cfg.seed)
    n = len(chosen_scores)
    boot = np.empty(bootstrap, dtype=np.float64)
    for b in range(bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = chosen_scores[idx].mean()
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    country_by_s1 = _country_of_source1(cfg, "train")
    per_country = {}
    for country in sorted(set(country_by_s1.get(sid, "") for sid in ids)):
        mask = np.array([country_by_s1.get(sid, "") == country for sid in ids])
        if mask.any():
            per_country[country] = float(chosen_scores[mask].mean())

    return {
        "val_s1": n,
        "candidate_truth_pairs": total_truth,
        "candidate_pairs_found": hit_truth,
        "candidate_recall_ceiling": (hit_truth / total_truth) if total_truth else 0.0,
        "threshold_only": {"macro_f05": thr_results[best_thr][0], "threshold": thresholds[best_thr]},
        "one_to_one": {"macro_f05": oto_results[best_oto][0], "threshold": thresholds[best_oto]},
        "chosen_method": "one_to_one" if use_oto else "threshold_only",
        "chosen_macro_f05": float(chosen_scores.mean()),
        "ci95": ci,
        "per_country": per_country,
        "singletons": int((ntrue == 0).sum()),
    }


def evaluate_full_candidates(cfg, val_frac=0.2, workers=8, reuse=False):
    from ber.features import run_features
    from ber.predict import _load_booster, predict_parts

    data = Path(cfg.data_dir)
    val_s1_ids(cfg, val_frac)
    print(build_valfull_pairs(cfg, val_frac), flush=True)
    feat_dir = data / "tmp" / f"{SPLIT}_features"
    if not feat_dir.exists() or not any(feat_dir.glob("*.parquet")):
        print(run_features(cfg, SPLIT, workers=workers, combine=False, meta_split="train"), flush=True)
    pred_dir = data / "tmp" / f"{SPLIT}_pred"
    if not reuse or not pred_dir.exists() or not any(pred_dir.glob("*.parquet")):
        booster, features, _ = _load_booster(cfg)
        predict_parts(cfg, SPLIT, booster, features, 0.2)
    report = _score_preds(cfg, (data / "tmp" / f"{SPLIT}_pred" / "*.parquet").as_posix())
    report["note"] = "full candidates, held-out S1 groups; test-like conditions, no France labels"
    (_reports(cfg) / "eval_full_candidates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _train_country_model(cfg, frame, train_mask, country, country_by_s1):
    import lightgbm as lgb

    is_country = frame["s1_id"].map(country_by_s1).fillna("").eq(country).to_numpy()
    subset_mask = train_mask & is_country
    sub = frame[subset_mask]
    codes, uniques = pd.factorize(sub["s1_id"].to_numpy())
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(uniques))
    n_val = max(1, int(len(uniques) * 0.1))
    val_groups = set(order[:n_val].tolist())
    inner_val = np.array([c in val_groups for c in codes])
    params = dict(cfg.lgbm_params or {})
    num_boost_round = int(params.pop("n_estimators", 2000))
    params.update(objective="binary", metric="auc", num_threads=8, seed=cfg.seed, verbosity=-1)
    from ber.features import FEATURE_ORDER

    X = sub[FEATURE_ORDER]
    y = sub["label"].to_numpy()
    dtrain = lgb.Dataset(X[~inner_val], label=y[~inner_val])
    dval = lgb.Dataset(X[inner_val], label=y[inner_val], reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def evaluate_loo(cfg, val_frac=0.2):
    import lightgbm as lgb

    from ber.features import FEATURE_ORDER
    from ber.predict import _load_booster, predict_parts

    data = Path(cfg.data_dir)
    frame = pd.read_parquet(
        data / "features" / "train.parquet", columns=FEATURE_ORDER + ["label", "s1_id"]
    )
    train_mask, _ = grouped_split(frame, 0.2, cfg.seed)
    country_by_s1 = _country_of_source1(cfg, "train")
    ids = val_s1_ids(cfg, val_frac)

    reports = {}
    for train_country, val_country in (("us", "india"), ("india", "us")):
        mask_country = np.array([country_by_s1.get(sid, "") == val_country for sid in ids])
        subset_ids = ids[mask_country]
        subset_path = _reports(cfg) / f"{SPLIT}_s1_ids_{val_country}.parquet"
        pd.DataFrame({"s1_id": subset_ids}).to_parquet(subset_path, index=False)
        booster = _train_country_model(cfg, frame, train_mask, train_country, country_by_s1)
        pred_dir = data / "tmp" / f"{SPLIT}_pred_loo_{train_country}"
        if pred_dir.exists():
            shutil.rmtree(pred_dir)
        predict_parts(cfg, SPLIT, booster, FEATURE_ORDER, 0.2, out_dir=pred_dir)
        report = _score_preds(
            cfg,
            (pred_dir / "*.parquet").as_posix(),
            valids_path=subset_path,
            pairs_path=data / "pairs" / f"{SPLIT}_pairs.parquet",
            bootstrap=200,
        )
        report["train_country"] = train_country
        report["val_country"] = val_country
        reports[f"train_{train_country}_val_{val_country}"] = report

    booster, features, _ = _load_booster(cfg)
    for country in ("us", "india"):
        mask_country = np.array([country_by_s1.get(sid, "") == country for sid in ids])
        subset_path = _reports(cfg) / f"{SPLIT}_s1_ids_{country}.parquet"
        pd.DataFrame({"s1_id": ids[mask_country]}).to_parquet(subset_path, index=False)
        report = _score_preds(
            cfg,
            (data / "tmp" / f"{SPLIT}_pred" / "*.parquet").as_posix(),
            valids_path=subset_path,
            pairs_path=data / "pairs" / f"{SPLIT}_pairs.parquet",
            bootstrap=200,
        )
        reports[f"full_model_{country}"] = report

    (_reports(cfg) / "eval_loo.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports
