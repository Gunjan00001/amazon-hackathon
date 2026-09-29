import numpy as np

from .score import best_threshold, macro_f05


def grouped_split(groups, val_frac=0.2, seed=42):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    val_groups = set(uniq[: max(1, int(val_frac * len(uniq)))].tolist())
    val_mask = np.isin(groups, list(val_groups))
    return ~val_mask, val_mask


def holdout_mask(countries, country):
    return np.asarray(countries) == country


def entity_f05_counts(n_true, n_pred, n_hit):
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or n_hit == 0:
        return 0.0
    precision = n_hit / n_pred
    recall = n_hit / n_true
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def entity_macro_f05(true_by_s1, pred_by_s1, universe=None):
    if universe is None:
        universe = sorted(set(true_by_s1) | set(pred_by_s1))
    scores = []
    for s in universe:
        t = set(true_by_s1.get(s, ()))
        p = set(pred_by_s1.get(s, ()))
        scores.append(entity_f05_counts(len(t), len(p), len(t & p)))
    return float(np.mean(scores)) if scores else 0.0


def entity_macro_counts(n_true, n_pred, n_hit):
    n_true = np.asarray(n_true, dtype="float64")
    n_pred = np.asarray(n_pred, dtype="float64")
    n_hit = np.asarray(n_hit, dtype="float64")
    out = np.zeros_like(n_true)
    out[(n_true == 0) & (n_pred == 0)] = 1.0
    mask = n_true > 0
    precision = np.divide(n_hit, n_pred, out=np.zeros_like(n_hit), where=n_pred > 0)
    recall = np.divide(n_hit, n_true, out=np.zeros_like(n_hit), where=n_true > 0)
    denom = 0.25 * precision + recall
    f = np.divide(1.25 * precision * recall, denom, out=np.zeros_like(precision), where=denom > 0)
    out[mask] = f[mask]
    return float(out.mean())


def evaluate_solution(groups, labels, scores, countries=None, holdout_country=None, grid=None):
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.025)
    f, th = best_threshold(groups, labels, scores, grid)
    out = {"macro_f05": float(f), "threshold": float(th), "n_groups": int(len(np.unique(groups)))}
    if countries is not None and holdout_country is not None:
        m = holdout_mask(countries, holdout_country)
        if m.any():
            out["holdout_country"] = holdout_country
            out["macro_f05_holdout"] = float(macro_f05(groups[m], labels[m], scores[m], th))
    return out
