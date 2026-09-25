import numpy as np
from ber.gbdt import predict_scores, split_groups, train_model


def test_split_groups_is_disjoint_and_grouped():
    groups = np.repeat(np.arange(100), 5)
    tr, va = split_groups(groups, val_frac=0.2, seed=0)
    assert not (set(groups[tr]) & set(groups[va]))
    assert len(set(groups[va])) == 20


def test_train_model_separates_easy_signal():
    rng = np.random.default_rng(0)
    n = 4000
    groups = np.repeat(np.arange(400), 10)
    X = rng.normal(size=(n, 4)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    model, metrics = train_model(X, y, groups)
    assert metrics["pair_auc"] > 0.9
    assert 0.0 <= metrics["best_threshold"] <= 1.0
    scores = predict_scores(model, X)
    assert scores.shape == (n,) and scores.min() >= 0 and scores.max() <= 1
