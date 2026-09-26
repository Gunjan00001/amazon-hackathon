def macro_f05(truth: dict, pred: dict) -> float:
    keys = set(truth) | set(pred)
    if not keys:
        return 0.0
    scores = []
    for key in keys:
        t = truth.get(key, set())
        p = pred.get(key, set())
        if not t:
            scores.append(1.0 if not p else 0.0)
            continue
        if not p:
            scores.append(0.0)
            continue
        tp = len(t & p)
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / len(p)
        recall = tp / len(t)
        scores.append((1.25 * precision * recall) / (0.25 * precision + recall))
    return sum(scores) / len(scores)


def _one_to_one_mask(groups, cand_ids, probs, mask):
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({"g": groups, "c": cand_ids, "p": probs, "m": mask})
    selected = frame[frame["m"]].sort_values(
        ["c", "p", "g"], ascending=[True, False, True]
    )
    keep_idx = selected.drop_duplicates("c", keep="first").index.to_numpy()
    out = np.zeros(len(frame), dtype=bool)
    out[keep_idx] = True
    return out


def evaluate_marks(cfg, split="train"):
    import json
    from pathlib import Path

    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    from ber.features import FEATURE_ORDER
    from ber.pairs import grouped_split
    from ber.threshold import entity_f05

    data = Path(cfg.data_dir)
    frame = pd.read_parquet(
        data / "features" / f"{split}.parquet",
        columns=FEATURE_ORDER + ["label", "s1_id", "cand_id"],
    )
    train_mask, val_mask = grouped_split(frame, 0.2, cfg.seed)
    booster = lgb.Booster(model_file=str(Path(cfg.models_dir) / "lgbm.txt"))
    probs = booster.predict(frame.loc[val_mask, FEATURE_ORDER], num_iteration=booster.best_iteration)
    groups = frame.loc[val_mask, "s1_id"].to_numpy()
    cands = frame.loc[val_mask, "cand_id"].to_numpy()
    labels = frame.loc[val_mask, "label"].to_numpy()

    best_threshold_only = (0.0, 0.05)
    best_one_to_one = (0.0, 0.05)
    for threshold in np.arange(0.05, 0.951, 0.025):
        mask = probs >= threshold
        score = entity_f05(groups, labels, mask)
        if score > best_threshold_only[0]:
            best_threshold_only = (score, float(threshold))
        oto_mask = _one_to_one_mask(groups, cands, probs, mask)
        oto_score = entity_f05(groups, labels, oto_mask)
        if oto_score > best_one_to_one[0]:
            best_one_to_one = (oto_score, float(threshold))

    _, threshold = best_threshold_only
    mask = probs >= threshold
    tp = int(np.sum((labels.astype(bool)) & mask))
    precision = tp / int(mask.sum()) if mask.sum() else 0.0
    recall = tp / int(labels.sum()) if labels.sum() else 0.0

    unique_groups = pd.factorize(groups)[0]
    n_groups = unique_groups.max() + 1 if len(unique_groups) else 0
    ntrue = np.bincount(unique_groups, weights=labels.astype(float), minlength=n_groups)
    npred = np.bincount(unique_groups, weights=mask.astype(float), minlength=n_groups)
    singletons = int((ntrue == 0).sum())
    false_merges_on_singletons = int(((ntrue == 0) & (npred > 0)).sum())

    result = {
        "split": split,
        "val_pairs": int(val_mask.sum()),
        "val_entities": int(n_groups),
        "threshold_only": {
            "macro_f05": best_threshold_only[0],
            "threshold": best_threshold_only[1],
        },
        "one_to_one": {
            "macro_f05": best_one_to_one[0],
            "threshold": best_one_to_one[1],
        },
        "best_method": "one_to_one" if best_one_to_one[0] > best_threshold_only[0] else "threshold_only",
        "precision": precision,
        "recall": recall,
        "singletons": singletons,
        "false_merges_on_singletons": false_merges_on_singletons,
        "note": "4:1 sampled negatives; optimistic and not leaderboard-comparable",
    }
    reports = data / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "eval_marks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
