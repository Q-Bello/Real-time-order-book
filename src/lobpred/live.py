"""Live predictions in the console, scored against what actually happens.

Uses exactly the same feed, book and feature code as the recorded-data pipeline.
Every prediction is checked once its horizon has passed, so the console shows a
running live hit rate alongside the model's call.

Usage:
    python -m lobpred.live --horizon 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from lobpred.config import MODEL_DIR, PRIMARY_HORIZON_S
from lobpred.features import FeatureEngine
from lobpred.labels import LABEL_NAMES
from lobpred.market import MarketState
from lobpred.stream import live_records


class Predictor:
    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.classes = np.array(bundle["classes"])
        self.horizon_s = bundle["horizon_s"]

    def predict(self, row: dict) -> tuple[int, np.ndarray]:
        """Returns (predicted class, probabilities for STABLE, DOWN, UP)."""
        x = np.array([[row[f] for f in self.features]])
        proba = np.zeros(3)
        proba[self.classes] = self.model.predict_proba(x)[0]
        return int(proba.argmax()), proba


class LiveScorer:
    """Checks each prediction against the realised move once the horizon has passed."""

    def __init__(self, horizon_s: float, threshold_spreads: float = 0.5):
        self.horizon_ms = int(horizon_s * 1000)
        self.threshold_spreads = threshold_spreads
        self.pending: deque = deque()
        self.n = self.correct = 0

    def add(self, ts: int, mid: float, spread: float, pred: int) -> None:
        self.pending.append((ts, mid, spread, pred))

    def update(self, ts: int, mid: float) -> None:
        while self.pending and ts >= self.pending[0][0] + self.horizon_ms:
            _, mid0, spread0, pred = self.pending.popleft()
            thr = self.threshold_spreads * spread0
            actual = 2 if mid - mid0 > thr else (1 if mid - mid0 < -thr else 0)
            self.n += 1
            self.correct += int(actual == pred)

    def reset(self) -> None:
        self.pending.clear()


async def run(predictor: Predictor) -> None:
    state, engine = MarketState(), FeatureEngine()
    scorer = LiveScorer(predictor.horizon_s)
    print(f"Streaming live {predictor.horizon_s}s predictions (warming up for ~5s)...")
    async for record in live_records():
        update = state.on_record(record)
        if update is None:
            continue
        if update.kind == "trade":
            engine.on_trade(update.data)
            continue
        if update.resynced:
            engine.reset()
            scorer.reset()
        row = engine.on_book(state.book, update.ts)
        if row is None:
            continue
        scorer.update(update.ts, row["mid"])
        pred, proba = predictor.predict(row)
        scorer.add(update.ts, row["mid"], row["spread"], pred)
        hit = f"{scorer.correct / scorer.n:.1%} of {scorer.n}" if scorer.n else "n/a"
        print(
            f"\r{LABEL_NAMES[pred]:<6} P(up) {proba[2]:.2f} P(down) {proba[1]:.2f} "
            f"| mid {row['mid']:.2f} | OFI {row['ofi_l1_10']:+.2f} | imbalance {row['imb_l1']:+.2f} "
            f"| live hit rate {hit}   ",
            end="",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=PRIMARY_HORIZON_S)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    path = args.model_dir / f"xgb_{args.horizon}s.joblib"
    if not path.exists():
        raise SystemExit(f"No model at {path}. Run `python -m lobpred.evaluate` first.")
    logging.basicConfig(level=logging.INFO, format="\n%(levelname)s %(message)s")
    try:
        asyncio.run(run(Predictor(path)))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
