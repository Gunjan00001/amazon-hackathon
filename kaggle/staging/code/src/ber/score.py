import numpy as np


def entity_f05(labels: np.ndarray, preds: np.ndarray) -> float:
    n_true = int(labels.sum())
    n_pred = int(preds.sum())
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0
    tp = int((labels & preds).sum())
    precision = tp / n_pred
    recall = tp / n_true
    if precision == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(groups: np.ndarray, labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    preds = scores >= threshold
    return float(np.mean([entity_f05(labels[groups == g], preds[groups == g]) for g in np.unique(groups)]))


def best_threshold(groups, labels, scores, grid=np.arange(0.05, 0.96, 0.025)):
    best = (0.0, float(grid[0]))
    for th in grid:
        f = macro_f05(groups, labels, scores, float(th))
        if f > best[0]:
            best = (f, float(th))
    return best
