"""End to end on simulated data: record format -> dataset -> evaluation -> report."""

import json

from lobpred.dataset import build_dataset
from lobpred.evaluate import evaluate
from lobpred.simulate import write_simulation


def test_full_pipeline_on_simulated_market(tmp_path):
    raw = write_simulation(tmp_path / "raw" / "sim.jsonl.gz", minutes=20, seed=1, gap_at_minute=10)
    df, meta = build_dataset([raw], horizons_s=(1, 5))
    assert meta["segments"] == 2 and meta["rows"] > 10_000

    results = evaluate(df, [1, 5], n_folds=3, fast=True, oof_dir=tmp_path / "oof", model_dir=tmp_path / "models")
    xgb = results["metrics"]["1s"]["XGBoost"]
    majority = results["metrics"]["1s"]["Majority class"]
    # The simulator has a real order-flow signal, so the model must beat the baseline.
    assert xgb["balanced_accuracy"]["mean"] > majority["balanced_accuracy"]["mean"] + 0.03
    assert xgb["ic"]["mean"] > 0.1
    assert (tmp_path / "models" / "xgb_5s.joblib").exists()
    json.dumps(results)  # serialisable
