import numpy as np
from sklearn.linear_model import LogisticRegression

from lobpred.config import SEGMENT_GAP_MS
from lobpred.evaluate import fit_predict_proba, walk_forward_splits


def test_walk_forward_trains_only_on_the_past_and_purges_overlap():
    ts = np.arange(0, 600_000, 100)
    h = 5
    folds = list(walk_forward_splits(ts, n_folds=4, horizon_s=h))
    assert len(folds) == 4
    for train, test in folds:
        assert ts[train].max() < ts[test].min()
        # No training label window reaches into the test block.
        assert ts[train].max() + h * 1000 + SEGMENT_GAP_MS < ts[test].min()


def test_probabilities_cover_all_classes_when_one_is_missing_from_training():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = np.where(X[:, 0] > 0, 2, 0)  # class 1 never appears
    proba = fit_predict_proba(LogisticRegression(), X, y, X[:5])
    assert proba.shape == (5, 3)
    assert np.allclose(proba[:, 1], 0) and np.allclose(proba.sum(axis=1), 1)
