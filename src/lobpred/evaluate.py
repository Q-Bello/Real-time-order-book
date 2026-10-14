"""Walk-forward evaluation of every model at every horizon, plus the cost-aware backtest.

Design:
  - Walk-forward: the data is cut into consecutive blocks. Fold k trains on all blocks
    before block k and tests on block k, so the model never sees the future.
  - Purging: training rows whose label window reaches into the test block are dropped.
  - Baselines: majority class, a logistic model on level-1 OFI alone (the classic linear
    benchmark), and logistic regression on all features, so XGBoost's value is measured
    against simple alternatives rather than in isolation.
  - Metrics: accuracy, balanced accuracy, macro F1, log loss, and information
    coefficient (rank correlation between the signal P(UP) - P(DOWN) and the realised
    return), reported as mean and standard deviation across folds.

Writes reports/results.json, out-of-sample predictions to data/oof/, and a model per
horizon trained on all the data to models/ for live use.

Usage:
    python -m lobpred.evaluate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from lobpred.backtest import THRESHOLDS, run_backtest, summarise
from lobpred.config import DATASET_PATH, HORIZONS_S, MODEL_DIR, OOF_DIR, REPORT_DIR, SEGMENT_GAP_MS
from lobpred.features import FEATURES

CLASSES = np.array([0, 1, 2])
MAIN_MODEL = "XGBoost"


class MajorityClass:
    """Predicts the training period's class frequencies for every row."""

    def fit(self, X, y):
        self.p_ = np.bincount(y) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))


def make_models(fast: bool) -> dict:
    return {
        "Majority class": (FEATURES, lambda: MajorityClass()),
        "OFI only (logistic)": (["ofi_l1_10"], lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000))),
        "Logistic (all features)": (FEATURES, lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000))),
        MAIN_MODEL: (FEATURES, lambda: XGBClassifier(
            n_estimators=60 if fast else 300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method="hist", n_jobs=-1, random_state=0)),
    }


def fit_predict_proba(estimator, X_train, y_train, X_test) -> np.ndarray:
    """Fit on whichever classes appear in training; return probabilities for all 3."""
    present = np.unique(y_train)
    mapping = {c: i for i, c in enumerate(present)}
    estimator.fit(X_train, np.vectorize(mapping.get)(y_train))
    out = np.zeros((len(X_test), len(CLASSES)))
    out[:, present] = estimator.predict_proba(X_test)
    return out


def walk_forward_splits(ts: np.ndarray, n_folds: int, horizon_s: float):
    """Expanding-window splits with purging of training labels that overlap the test block."""
    blocks = np.array_split(np.arange(len(ts)), n_folds + 1)
    purge_ms = horizon_s * 1000 + SEGMENT_GAP_MS
    for k in range(1, n_folds + 1):
        test = blocks[k]
        train = np.concatenate(blocks[:k])
        train = train[ts[train] + purge_ms < ts[test[0]]]
        yield train, test


def score(y: np.ndarray, proba: np.ndarray, ret: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    clipped = np.clip(proba, 1e-6, 1)
    signal = proba[:, 2] - proba[:, 1]
    ic = spearmanr(signal, ret).statistic if np.ptp(signal) > 1e-12 and np.ptp(ret) > 0 else 0.0
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
        "log_loss": log_loss(y, clipped / clipped.sum(axis=1, keepdims=True), labels=CLASSES),
        "ic": float(np.nan_to_num(ic)),
    }


def permutation_importance(model, X: pd.DataFrame, y: np.ndarray, repeats: int = 3, seed: int = 0) -> dict:
    """Drop in balanced accuracy when each feature is shuffled."""
    rng = np.random.default_rng(seed)
    base = balanced_accuracy_score(y, model.predict_proba(X).argmax(axis=1))
    out = {}
    for col in X.columns:
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - balanced_accuracy_score(y, model.predict_proba(Xp).argmax(axis=1)))
        out[col] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def evaluate(df: pd.DataFrame, horizons_s, n_folds: int, fast: bool, oof_dir: Path, model_dir: Path) -> dict:
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    hours = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 3.6e6
    models = make_models(fast)
    results = {
        "setup": {
            "rows": int(len(df)), "hours": round(hours, 2), "folds": n_folds,
            "split": "expanding walk-forward, purged", "features": FEATURES,
            "models": list(models),
        },
        "metrics": {}, "per_fold": {}, "importance": {}, "backtest": {},
    }
    oof_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for h in horizons_s:
        t0 = time.time()
        d = df.dropna(subset=[f"y_{h}s"]).reset_index(drop=True)
        y = d[f"y_{h}s"].to_numpy().astype(int)
        ret = d[f"fwd_ret_{h}s_bps"].to_numpy()
        ts = d["ts"].to_numpy()
        fold_scores = {name: [] for name in models}
        oof = np.full((len(d), len(CLASSES)), np.nan)
        last_main = last_test = None

        for train, test in walk_forward_splits(ts, n_folds, h):
            for name, (cols, factory) in models.items():
                est = factory()
                proba = fit_predict_proba(est, d.loc[train, cols], y[train], d.loc[test, cols])
                fold_scores[name].append(score(y[test], proba, ret[test]))
                if name == MAIN_MODEL:
                    oof[test] = proba
                    last_main, last_test = est, test

        key = f"{h}s"
        results["per_fold"][key] = {n: [{m: round(v, 4) for m, v in s.items()} for s in fs]
                                    for n, fs in fold_scores.items()}
        results["metrics"][key] = {
            name: {m: {"mean": round(float(np.mean([s[m] for s in fs])), 4),
                       "std": round(float(np.std([s[m] for s in fs])), 4)} for m in fs[0]}
            for name, fs in fold_scores.items()
        }

        sample = last_test[-20_000:]
        results["importance"][key] = {
            f: round(v, 4) for f, v in permutation_importance(
                last_main, d.loc[sample, FEATURES], y[sample]).items()
        }

        has = ~np.isnan(oof[:, 0])
        out = d.loc[has, ["ts", "segment", "bid", "ask", "mid", f"fwd_ret_{h}s_bps", f"y_{h}s"]].copy()
        out[["p_stable", "p_down", "p_up"]] = oof[has]
        out["score"] = out["p_up"] - out["p_down"]
        out.to_parquet(oof_dir / f"oof_{key}.parquet", index=False)
        oof_hours = (out["ts"].iloc[-1] - out["ts"].iloc[0]) / 3.6e6
        results["backtest"][key] = {
            f"{thr:g}": summarise(run_backtest(out, h, thr), oof_hours) for thr in THRESHOLDS
        }

        final = make_models(fast)[MAIN_MODEL][1]()
        present = np.unique(y)
        final.fit(d[FEATURES], np.vectorize({c: i for i, c in enumerate(present)}.get)(y))
        joblib.dump({"model": final, "features": FEATURES, "classes": present.tolist(), "horizon_s": h},
                    model_dir / f"xgb_{key}.joblib")

        m = results["metrics"][key]
        print(f"{key:>4}: balanced accuracy  XGBoost {m[MAIN_MODEL]['balanced_accuracy']['mean']:.3f}  "
              f"| logistic {m['Logistic (all features)']['balanced_accuracy']['mean']:.3f}  "
              f"| OFI only {m['OFI only (logistic)']['balanced_accuracy']['mean']:.3f}  "
              f"| majority {m['Majority class']['balanced_accuracy']['mean']:.3f}  "
              f"| IC {m[MAIN_MODEL]['ic']['mean']:+.3f}  ({time.time() - t0:.0f}s)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATASET_PATH)
    parser.add_argument("--horizons", type=float, nargs="+", default=list(HORIZONS_S))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="Smaller XGBoost, for quick checks")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--oof-dir", type=Path, default=OOF_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    horizons = [int(h) if float(h).is_integer() else h for h in args.horizons]
    df = pd.read_parquet(args.data)
    results = evaluate(df, horizons, args.folds, args.fast, args.oof_dir, args.model_dir)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.report_dir / 'results.json'}. Next: python -m lobpred.report")


if __name__ == "__main__":
    main()
