import numpy as np
import pandas as pd


def entity_f05(groups, labels, predicted, thresholds=None):
    frame = pd.DataFrame({"g": groups, "y": labels})
    pred = np.asarray(predicted) if thresholds is None else (np.asarray(predicted) >= thresholds)
    frame["p"] = pred
    frame["tp"] = (frame["y"].astype(bool) & frame["p"].astype(bool)).astype(int)
    agg = frame.groupby("g").agg(tp=("tp", "sum"), npred=("p", "sum"), ntrue=("y", "sum"))
    scores = np.ones(len(agg), dtype=np.float64)
    has_true = agg["ntrue"] > 0
    no_pred = agg["npred"] == 0
    scores[has_true & no_pred] = 0.0
    valid = has_true & ~no_pred
    tp = agg.loc[valid, "tp"].to_numpy(dtype=np.float64)
    npred = agg.loc[valid, "npred"].to_numpy(dtype=np.float64)
    ntrue = agg.loc[valid, "ntrue"].to_numpy(dtype=np.float64)
    precision = tp / npred
    recall = tp / ntrue
    f = np.zeros_like(precision)
    nz = precision > 0
    f[nz] = (1.25 * precision[nz] * recall[nz]) / (0.25 * precision[nz] + recall[nz])
    scores[valid] = f
    empty_true = ~has_true
    scores[empty_true] = 0.0
    scores[empty_true & no_pred] = 1.0
    return float(scores.mean())


def _entity_f05_from_codes(codes, n_groups, labels, predicted):
    codes = np.asarray(codes)
    y = np.asarray(labels).astype(bool)
    p = np.asarray(predicted).astype(bool)
    ntrue = np.bincount(codes, weights=y.astype(np.float64), minlength=n_groups)
    npred = np.bincount(codes, weights=p.astype(np.float64), minlength=n_groups)
    tp = np.bincount(codes, weights=(y & p).astype(np.float64), minlength=n_groups)
    precision = np.divide(tp, npred, out=np.zeros(n_groups), where=npred > 0)
    recall = np.divide(tp, ntrue, out=np.zeros(n_groups), where=ntrue > 0)
    f = np.zeros(n_groups, dtype=np.float64)
    nz = precision > 0
    f[nz] = (1.25 * precision[nz] * recall[nz]) / (0.25 * precision[nz] + recall[nz])
    has_true = ntrue > 0
    no_pred = npred == 0
    scores = np.ones(n_groups, dtype=np.float64)
    scores[has_true & no_pred] = 0.0
    scores[has_true & ~no_pred] = f[has_true & ~no_pred]
    empty_true = ~has_true
    scores[empty_true] = 0.0
    scores[empty_true & no_pred] = 1.0
    return float(scores.mean())


def sweep_threshold(probs, groups, labels, low=0.05, high=0.95, step=0.025):
    codes, uniques = pd.factorize(groups)
    codes = np.asarray(codes)
    n_groups = len(uniques)
    if n_groups == 0:
        return (0.0, low)
    probs = np.asarray(probs, dtype=np.float64)
    best = (0.0, low)
    for threshold in np.arange(low, high + 1e-9, step):
        score = _entity_f05_from_codes(codes, n_groups, labels, probs >= threshold)
        if score > best[0]:
            best = (score, float(threshold))
    return best
