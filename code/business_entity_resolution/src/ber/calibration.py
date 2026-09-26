import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ber.threshold import entity_f05, sweep_threshold
from ber.validation import _country_of_source1, _macro_from_counts, val_s1_ids_path

TUNE_LOW = 0.05
TUNE_HIGH = 0.95
TUNE_STEP = 0.025
TAU_LOW = 0.3
TAU_HIGH = 0.99
TAU_STEP = 0.01
MIN_COUNTRY_ENTITIES = 5000
CALIB_SAMPLE_MOD = 4


def tune_per_country(
    probs,
    groups,
    labels,
    countries,
    global_default,
    low=TUNE_LOW,
    high=TUNE_HIGH,
    step=TUNE_STEP,
    min_entities=MIN_COUNTRY_ENTITIES,
):
    probs = np.asarray(probs, dtype=np.float64)
    groups = np.asarray(groups)
    labels = np.asarray(labels).astype(bool)
    countries = np.asarray(countries)
    if global_default is None:
        _, global_default = sweep_threshold(probs, groups, labels.astype(np.int8), low, high, step)
    thresholds = {}
    for country in sorted(set(countries.tolist())):
        mask = countries == country
        sub_probs = probs[mask]
        sub_groups = groups[mask]
        sub_labels = labels[mask]
        n_entities = len(pd.unique(sub_groups))
        if n_entities < min_entities or not sub_labels.any() or sub_labels.all():
            thresholds[country] = float(global_default)
            continue
        _, threshold = sweep_threshold(sub_probs, sub_groups, sub_labels.astype(np.int8), low, high, step)
        thresholds[country] = float(threshold)
    return thresholds


def singleton_decision(probs, groups, tau):
    probs = np.asarray(probs, dtype=np.float64)
    groups = np.asarray(groups)
    group_max = (
        pd.DataFrame({"p": probs, "g": groups}).groupby("g")["p"].transform("max").to_numpy()
    )
    return group_max >= tau


def tune_singleton_tau(probs, groups, labels, thresholds, low=TAU_LOW, high=TAU_HIGH, step=TAU_STEP):
    probs = np.asarray(probs, dtype=np.float64)
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    base = probs >= thresholds
    best_score = -1.0
    best_tau = float(low)
    for tau in np.arange(low, high + 1e-9, step):
        mask = base & singleton_decision(probs, groups, tau)
        score = entity_f05(groups, labels, mask)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_score, best_tau


def apply_calibration(probs, groups, thresholds, tau=None):
    probs = np.asarray(probs, dtype=np.float64)
    mask = probs >= np.asarray(thresholds, dtype=np.float64)
    if tau is not None:
        mask &= singleton_decision(probs, groups, tau)
    return mask


def save_calibration(path, global_threshold, by_country, singleton_tau, use_one_to_one):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "global": float(global_threshold),
        "by_country": {str(k): float(v) for k, v in (by_country or {}).items()},
        "singleton_tau": None if singleton_tau is None else float(singleton_tau),
        "use_one_to_one": bool(use_one_to_one),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_calibration(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "global": payload.get("global"),
        "by_country": payload.get("by_country") or {},
        "singleton_tau": payload.get("singleton_tau"),
        "use_one_to_one": bool(payload.get("use_one_to_one", False)),
    }


def _configure(con, data):
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(Path(data) / 'tmp').as_posix()}'")
    con.execute("PRAGMA max_temp_directory_size='50GiB'")


def _pred_glob(cfg):
    return (Path(cfg.data_dir) / "tmp" / "valfull_pred" / "*.parquet").as_posix()


def _truth_table_sql(cfg):
    data = Path(cfg.data_dir)
    gt = (data / "processed" / "train_ground_truth.parquet").as_posix()
    return f"""
        CREATE TEMP TABLE truth AS
        SELECT g.source1_entity_id AS s1_id, UNNEST(string_split(g.matched_entity_ids, ',')) AS cand_id
        FROM read_parquet('{gt}') g SEMI JOIN valids v ON v.s1_id = g.source1_entity_id
        WHERE g.matched_entity_ids <> ''
    """


def _calibration_device(cfg, sample_mod):
    data = Path(cfg.data_dir)
    con = duckdb.connect()
    _configure(con, data)
    con.execute(
        f"CREATE TEMP TABLE valids AS SELECT s1_id FROM read_parquet('{val_s1_ids_path(cfg).as_posix()}')"
    )
    con.execute(_truth_table_sql(cfg))
    frame = con.execute(
        f"""
        SELECT p.s1_id, p.prob, CASE WHEN t.cand_id IS NULL THEN 0 ELSE 1 END AS is_true
        FROM read_parquet('{_pred_glob(cfg)}') p
        SEMI JOIN valids v ON v.s1_id = p.s1_id
        LEFT JOIN truth t ON t.s1_id = p.s1_id AND t.cand_id = p.cand_id
        WHERE hash(p.s1_id) % {int(sample_mod)} = 0
        """
    ).fetchdf()
    con.close()
    return frame


def tune_calibration(cfg, sample_mod=CALIB_SAMPLE_MOD):
    frame = _calibration_device(cfg, sample_mod)
    groups = frame["s1_id"].to_numpy()
    probs = frame["prob"].to_numpy(dtype=np.float64)
    labels = frame["is_true"].to_numpy(dtype=np.int8)
    country_by_s1 = _country_of_source1(cfg, "train")
    countries = np.array([country_by_s1.get(s1, "") for s1 in groups])
    global_score, global_threshold = sweep_threshold(probs, groups, labels)
    by_country = tune_per_country(probs, groups, labels, countries, global_default=global_threshold)
    row_thresholds = np.array([by_country.get(c, global_threshold) for c in countries])
    base_score = entity_f05(groups, labels, probs >= row_thresholds)
    tau_score, tau = tune_singleton_tau(probs, groups, labels, row_thresholds)
    calibration = {
        "global": float(global_threshold),
        "by_country": {k: float(v) for k, v in by_country.items()},
        "singleton_tau": float(tau),
        "use_one_to_one": True,
    }
    report = {
        "sample_pairs": int(len(frame)),
        "sample_entities": int(frame["s1_id"].nunique()),
        "global_threshold": float(global_threshold),
        "global_score": float(global_score),
        "by_country": calibration["by_country"],
        "per_country_score": float(base_score),
        "singleton_tau": float(tau),
        "singleton_score": float(tau_score),
    }
    return calibration, report


def score_calibration(cfg, calibration, bootstrap=500):
    data = Path(cfg.data_dir)
    valids_path = val_s1_ids_path(cfg)
    pairs_path = (data / "pairs" / "valfull_pairs.parquet").as_posix()
    global_threshold = float(calibration["global"])
    tau = calibration["singleton_tau"]
    use_one_to_one = bool(calibration["use_one_to_one"])

    con = duckdb.connect()
    _configure(con, data)
    con.execute(f"CREATE TEMP TABLE valids AS SELECT s1_id FROM read_parquet('{valids_path.as_posix()}')")
    con.execute(_truth_table_sql(cfg))
    ids = [r[0] for r in con.execute("SELECT s1_id FROM valids ORDER BY s1_id").fetchall()]

    country_by_s1 = _country_of_source1(cfg, "train")
    con.register(
        "country_map",
        pd.DataFrame({"s1_id": ids, "country": [country_by_s1.get(s, "") for s in ids]}),
    )
    by_country = calibration["by_country"] or {}
    if by_country:
        thr_frame = pd.DataFrame(
            {"country": list(by_country), "threshold": [float(v) for v in by_country.values()]}
        )
    else:
        thr_frame = pd.DataFrame({"country": pd.Series(dtype="object"), "threshold": pd.Series(dtype="float64")})
    con.register("country_thr", thr_frame)

    con.execute(
        f"""
        CREATE TEMP TABLE gmax AS
        SELECT p.s1_id, MAX(p.prob) AS mx
        FROM read_parquet('{_pred_glob(cfg)}') p
        SEMI JOIN valids v ON v.s1_id = p.s1_id
        GROUP BY p.s1_id
        """
    )
    tau_sql = f"AND g.mx >= {tau}" if tau is not None else ""
    con.execute(
        f"""
        CREATE TEMP TABLE eligible AS
        SELECT p.s1_id, p.cand_id, p.prob
        FROM read_parquet('{_pred_glob(cfg)}') p
        SEMI JOIN valids v ON v.s1_id = p.s1_id
        JOIN gmax g ON g.s1_id = p.s1_id
        LEFT JOIN country_map cm ON cm.s1_id = p.s1_id
        LEFT JOIN country_thr ct ON ct.country = cm.country
        WHERE p.prob >= coalesce(ct.threshold, {global_threshold}) {tau_sql}
        """
    )
    if use_one_to_one:
        con.execute(
            """
            CREATE TEMP TABLE keep AS
            SELECT s1_id, cand_id FROM (
                SELECT s1_id, cand_id,
                       row_number() OVER (PARTITION BY cand_id ORDER BY prob DESC, s1_id ASC) AS rn
                FROM eligible
            ) WHERE rn = 1
            """
        )
    else:
        con.execute("CREATE TEMP TABLE keep AS SELECT DISTINCT s1_id, cand_id FROM eligible")

    ntrue_rows = con.execute("SELECT s1_id, COUNT(*) FROM truth GROUP BY s1_id").fetchall()
    npred_rows = con.execute("SELECT s1_id, COUNT(*) FROM keep GROUP BY s1_id").fetchall()
    tp_rows = con.execute(
        "SELECT k.s1_id, COUNT(*) FROM keep k JOIN truth t ON t.s1_id = k.s1_id AND t.cand_id = k.cand_id GROUP BY k.s1_id"
    ).fetchall()
    ceiling = con.execute(
        f"""
        SELECT COUNT(*) AS total, COUNT(c.cand_id) AS hit
        FROM truth t LEFT JOIN (SELECT DISTINCT s1_id, cand_id FROM read_parquet('{pairs_path}')) c
        ON c.s1_id = t.s1_id AND c.cand_id = t.cand_id
        """
    ).fetchone()
    con.close()

    index = {sid: i for i, sid in enumerate(ids)}
    ntrue = np.zeros(len(ids), dtype=np.int64)
    npred = np.zeros(len(ids), dtype=np.int64)
    tp = np.zeros(len(ids), dtype=np.int64)
    for s1_id, count in ntrue_rows:
        ntrue[index[s1_id]] = int(count)
    for s1_id, count in npred_rows:
        npred[index[s1_id]] = int(count)
    for s1_id, count in tp_rows:
        tp[index[s1_id]] = int(count)

    scores = _macro_from_counts(ntrue, npred, tp)
    rng = np.random.default_rng(cfg.seed)
    n = len(scores)
    boot = np.empty(bootstrap, dtype=np.float64)
    for b in range(bootstrap):
        boot[b] = scores[rng.integers(0, n, size=n)].mean()
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    per_country = {}
    country_arr = np.array([country_by_s1.get(s, "") for s in ids])
    for country in sorted(set(country_arr.tolist())):
        mask = country_arr == country
        if mask.any():
            per_country[country] = float(scores[mask].mean())

    total_truth, hit_truth = int(ceiling[0]), int(ceiling[1])
    return {
        "val_s1": n,
        "chosen_macro_f05": float(scores.mean()),
        "ci95": ci,
        "per_country": per_country,
        "thresholds": {"global": global_threshold, "by_country": {k: float(v) for k, v in by_country.items()}},
        "singleton_tau": tau,
        "use_one_to_one": use_one_to_one,
        "candidate_truth_pairs": total_truth,
        "candidate_pairs_found": hit_truth,
        "candidate_recall_ceiling": (hit_truth / total_truth) if total_truth else 0.0,
        "singletons": int((ntrue == 0).sum()),
    }


def run_calibration(cfg, sample_mod=CALIB_SAMPLE_MOD, bootstrap=500):
    data = Path(cfg.data_dir)
    preds_dir = data / "tmp" / "valfull_pred"
    if not preds_dir.exists() or not any(preds_dir.glob("*.parquet")):
        raise FileNotFoundError("no held-out predictions in DATA/tmp/valfull_pred; run `ber.cli validation` first")
    calibration, tune_report = tune_calibration(cfg, sample_mod)
    save_calibration(
        Path(cfg.models_dir) / "threshold.json",
        calibration["global"],
        calibration["by_country"],
        calibration["singleton_tau"],
        calibration["use_one_to_one"],
    )
    report = score_calibration(cfg, calibration, bootstrap=bootstrap)
    report["tuning"] = tune_report
    report["note"] = (
        "calibrated on held-out full-candidate predictions (1/sample_mod tuning sample), "
        f"scored on all held-out S1; sample_mod={sample_mod}"
    )
    (data / "reports").mkdir(parents=True, exist_ok=True)
    (data / "reports" / "eval_calibrated.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[calibrate] threshold global={calibration['global']:.3f} "
        f"by_country={ {k: round(v, 3) for k, v in calibration['by_country'].items()} } "
        f"tau={calibration['singleton_tau']:.2f} -> "
        f"chosen_macro_f05={report['chosen_macro_f05']:.4f} "
        f"per_country={ {k: round(v, 4) for k, v in report['per_country'].items()} }",
        flush=True,
    )
    return report
