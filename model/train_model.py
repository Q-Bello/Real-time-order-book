# Trains an XGBoost classifier on the collected dataset and evaluates it honestly.
#
# Evaluation design:
#   - Chronological split: train on the earlier data, test on the later data. A random
#     split would let the model train on moments seconds away from (and overlapping
#     with) test moments, which inflates scores on time-series data.
#   - Purge gap: each label looks `window` updates ahead, so the last training rows'
#     labels overlap the start of the test period. Those rows are dropped.
#   - Baseline: always predicting the most common class from the training period.
#     Accuracy only means something relative to this.
#
# Results are printed and saved to model/results.json.
#
# Usage (from the project root):
#   python -m model.train_model

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from microstructure.data_labeller import FEATURE_COLUMNS, LABEL_NAMES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, "data_stream", "lob_features.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "xgboost_model.pkl")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")


def chronological_split(df, test_fraction, purge):
    split = int(len(df) * (1 - test_fraction))
    train = df.iloc[: max(split - purge, 0)]
    test = df.iloc[split:]
    return train, test


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the mid-price direction model.")
    parser.add_argument("--data", default=CSV_PATH)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=30, help="Label horizon used when collecting")
    parser.add_argument("--balanced", action="store_true",
                        help="Weight classes equally in training (trades accuracy for balanced accuracy)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = [c for c in FEATURE_COLUMNS + ["label"] if c not in df.columns]
    if missing:
        sys.exit(f"Dataset is missing columns {missing}. Re-collect it with the current generate_dataset.py.")
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    train, test = chronological_split(df, args.test_fraction, purge=args.window)
    if train["label"].nunique() < 2 or len(test) == 0:
        sys.exit("Not enough data or label variety to train. Collect more rows first.")

    # XGBoost needs labels 0..k-1; map whatever classes are present.
    classes = np.sort(df["label"].unique())
    to_index = {c: i for i, c in enumerate(classes)}
    y_train = train["label"].map(to_index).to_numpy()
    y_test = test["label"].map(to_index).to_numpy()
    X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]

    model = XGBClassifier(
        objective="multi:softprob" if len(classes) > 2 else "binary:logistic",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss" if len(classes) > 2 else "logloss",
        random_state=42,
    )
    weights = compute_sample_weight("balanced", y_train) if args.balanced else None
    model.fit(X_train, y_train, sample_weight=weights)
    preds = model.predict(X_test)

    # Baseline: predict the training period's most common class for every test row.
    majority = np.bincount(y_train).argmax()
    baseline_preds = np.full_like(y_test, majority)

    names = [LABEL_NAMES.get(int(c), str(c)) for c in classes]
    results = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "split": "chronological, purged",
        "label_horizon_updates": args.window,
        "median_horizon_seconds": round(float(df["horizon_ms"].median()) / 1000, 2) if "horizon_ms" in df else None,
        "class_balance_test": {n: round(float((y_test == i).mean()), 3) for i, n in enumerate(names)},
        "baseline_class": names[majority],
        "baseline_accuracy": round(float(accuracy_score(y_test, baseline_preds)), 4),
        "model_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "baseline_balanced_accuracy": round(float(balanced_accuracy_score(y_test, baseline_preds)), 4),
        "model_balanced_accuracy": round(float(balanced_accuracy_score(y_test, preds)), 4),
        "model_macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
        "class_weighting": "balanced" if args.balanced else "none",
        "feature_importance": {
            f: round(float(v), 4) for f, v in zip(FEATURE_COLUMNS, model.feature_importances_)
        },
    }

    print(f"\nTrain rows: {results['rows_train']}  Test rows: {results['rows_test']} (chronological, purged)")
    print(f"Test class balance: {results['class_balance_test']}")
    print(classification_report(y_test, preds, labels=list(range(len(classes))), target_names=names, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, preds, labels=list(range(len(classes)))))
    print(f"\nAccuracy:          model {results['model_accuracy']:.1%}  vs  "
          f"baseline {results['baseline_accuracy']:.1%} (always '{results['baseline_class']}')")
    print(f"Balanced accuracy: model {results['model_balanced_accuracy']:.1%}  vs  "
          f"baseline {results['baseline_balanced_accuracy']:.1%}")
    if results["median_horizon_seconds"] is not None:
        print(f"Median label horizon: {results['median_horizon_seconds']}s")

    joblib.dump({"model": model, "classes": classes.tolist(), "features": FEATURE_COLUMNS}, MODEL_PATH)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nModel saved to {MODEL_PATH}\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
