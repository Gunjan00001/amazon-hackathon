import json
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from ber.features import FEATURE_ORDER
from ber.pairs import grouped_split
from ber.threshold import sweep_threshold


def train_model(cfg, split="train"):
    start = time.perf_counter()

    def phase(message):
        nonlocal start
        now = time.perf_counter()
        print(f"[train] {message} (+{now - start:.1f}s)", flush=True)
        start = now

    data = Path(cfg.data_dir)
    frame = pd.read_parquet(
        data / "features" / f"{split}.parquet",
        columns=FEATURE_ORDER + ["label", "s1_id"],
    )
    phase(f"loaded features: {len(frame):,} pairs")

    X = frame[FEATURE_ORDER]
    y = frame["label"].to_numpy()
    train_mask, val_mask = grouped_split(frame, 0.2, cfg.seed)
    phase(f"grouped split: {int(train_mask.sum()):,} train / {int(val_mask.sum()):,} val")

    params = dict(cfg.lgbm_params or {})
    num_boost_round = int(params.pop("n_estimators", 2000))
    params.update(
        objective="binary",
        metric="auc",
        num_threads=8,
        seed=cfg.seed,
        verbosity=1,
    )
    dtrain = lgb.Dataset(X[train_mask], label=y[train_mask])
    dval = lgb.Dataset(X[val_mask], label=y[val_mask], reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(25)],
    )
    phase(f"trained: best_iteration={booster.best_iteration}")

    probs = booster.predict(X[val_mask], num_iteration=booster.best_iteration)
    groups = frame.loc[val_mask, "s1_id"].to_numpy()
    labels = y[val_mask]
    score, threshold = sweep_threshold(probs, groups, labels)
    phase(f"threshold sweep: f05={score:.4f} @ {threshold}")

    models = Path(cfg.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(models / "lgbm.txt"))
    (models / "feature_list.json").write_text(json.dumps(FEATURE_ORDER, indent=2), encoding="utf-8")
    (models / "threshold.json").write_text(
        json.dumps({"global": threshold, "use_one_to_one": False}, indent=2), encoding="utf-8"
    )
    metrics = {
        "val_macro_f05": score,
        "threshold": threshold,
        "best_iteration": int(booster.best_iteration),
        "train_pairs": int(train_mask.sum()),
        "val_pairs": int(val_mask.sum()),
        "val_entities": int(frame.loc[val_mask, "s1_id"].nunique()),
    }
    (models / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    phase("saved model + metrics")
    return metrics
