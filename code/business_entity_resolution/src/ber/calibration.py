import json
from pathlib import Path

import numpy as np
import pandas as pd

from ber.features import _country_map
from ber.pairs import grouped_split
from ber.threshold import entity_f05, sweep_threshold

TUNE_LOW = 0.05
TUNE_HIGH = 0.95
TUNE_STEP = 0.025
TAU_LOW = 0.3
TAU_HIGH = 0.99
TAU_STEP = 0.01
MIN_COUNTRY_ENTITIES = 5000


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


def run_calibration(cfg):
    from ber.predict import _load_booster

    data = Path(cfg.data_dir)
    models = Path(cfg.models_dir)
    booster, features, _ = _load_booster(cfg)
    frame = pd.read_parquet(
        data / "features" / "train.parquet", columns=features + ["label", "s1_id"]
    )
    _, val_mask = grouped_split(frame, 0.2, cfg.seed)
    probs = booster.predict(frame.loc[val_mask, features], num_iteration=booster.best_iteration)
    groups = frame.loc[val_mask, "s1_id"].to_numpy()
    labels = frame.loc[val_mask, "label"].to_numpy()
    country_by_s1 = _country_map(cfg, "train")
    countries = np.array([country_by_s1.get(s1, "") for s1 in groups])

    global_score, global_threshold = sweep_threshold(probs, groups, labels)
    by_country = tune_per_country(probs, groups, labels, countries, global_default=global_threshold)
    row_thresholds = np.array([by_country.get(country, global_threshold) for country in countries])
    base_score = entity_f05(groups, labels, probs >= row_thresholds)
    tau_score, tau = tune_singleton_tau(probs, groups, labels, row_thresholds)

    save_calibration(
        models / "threshold.json",
        global_threshold=global_threshold,
        by_country=by_country,
        singleton_tau=tau,
        use_one_to_one=True,
    )
    report = {
        "val_pairs": int(val_mask.sum()),
        "val_entities": int(frame.loc[val_mask, "s1_id"].nunique()),
        "global_score": float(global_score),
        "global_threshold": float(global_threshold),
        "by_country": {k: float(v) for k, v in by_country.items()},
        "per_country_score": float(base_score),
        "singleton_tau": float(tau),
        "singleton_score": float(tau_score),
    }
    (data / "reports").mkdir(parents=True, exist_ok=True)
    (data / "reports" / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[calibrate] global={global_threshold:.3f} ({global_score:.4f}) "
        f"per_country={ {k: round(v, 3) for k, v in by_country.items()} } ({base_score:.4f}) "
        f"tau={tau:.2f} ({tau_score:.4f})",
        flush=True,
    )
    return report
