import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score, roc_auc_score

from .score import best_threshold

PARAMS = dict(objective="binary", n_estimators=1000, learning_rate=0.05, num_leaves=63,
              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              n_jobs=-1, random_state=42, verbosity=-1)


def split_groups(groups, val_frac=0.2, seed=42):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    val_groups = set(uniq[: max(1, int(val_frac * len(uniq)))].tolist())
    val_mask = np.isin(groups, list(val_groups))
    return ~val_mask, val_mask


def _val_usable(y_va, min_pos=5, min_neg=5):
    return int(y_va.sum()) >= min_pos and int((y_va == 0).sum()) >= min_neg


def train_model(X, y, groups, params=None):
    model = LGBMClassifier(**(params or PARAMS))
    tr, va = split_groups(groups)
    if _val_usable(y[va]):
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
                  callbacks=[early_stopping(50, verbose=False)])
        scores = model.predict_proba(X[va])[:, 1]
        f, th = best_threshold(groups[va], y[va], scores)
        eval_y, eval_scores = y[va], scores
    else:
        model.fit(X, y)
        scores = model.predict_proba(X)[:, 1]
        f, th = best_threshold(groups, y, scores)
        eval_y, eval_scores = y, scores
    if len(np.unique(eval_y)) < 2:
        auc = ap = float("nan")
    else:
        auc = float(roc_auc_score(eval_y, eval_scores))
        ap = float(average_precision_score(eval_y, eval_scores))
    metrics = {
        "pair_auc": auc,
        "average_precision": ap,
        "best_macro_f05": float(f),
        "best_threshold": float(th),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    return model, metrics


def predict_scores(model, X):
    return model.predict_proba(X)[:, 1]
