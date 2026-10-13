"""Build the modelling dataset by replaying recorded sessions.

Replays every record through the same MarketState and FeatureEngine used live, adds
multi-horizon labels, and writes a Parquet file plus a summary of the data, including
how fast the feature pipeline runs per book update.

Usage:
    python -m lobpred.dataset data/raw/*.jsonl.gz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lobpred.config import DATASET_PATH, HORIZONS_S, REPORT_DIR, SEGMENT_GAP_MS
from lobpred.features import FeatureEngine
from lobpred.labels import add_labels
from lobpred.market import MarketState
from lobpred.replay import read_records


def build_dataset(paths, horizons_s=HORIZONS_S) -> tuple[pd.DataFrame, dict]:
    state = MarketState()
    engine = FeatureEngine()
    rows: list[dict] = []
    latencies_ns: list[int] = []
    book_latencies_ns: list[int] = []
    segment = 0
    last_ts: int | None = None
    n_book = n_trades = 0
    first_ts = None

    for record in read_records(paths):
        t0 = time.perf_counter_ns()
        update = state.on_record(record)
        if record["k"] == "depth":
            book_latencies_ns.append(time.perf_counter_ns() - t0)
        if update is None:
            continue
        if update.kind == "trade":
            engine.on_trade(update.data)
            n_trades += 1
            continue

        n_book += 1
        if update.resynced or (last_ts is not None and update.ts - last_ts > SEGMENT_GAP_MS):
            engine.reset()
            segment += 1
        last_ts = update.ts
        first_ts = first_ts if first_ts is not None else update.ts

        t0 = time.perf_counter_ns()
        row = engine.on_book(state.book, update.ts)
        latencies_ns.append(time.perf_counter_ns() - t0)
        if row is not None:
            row["segment"] = segment
            rows.append(row)

    if not rows:
        raise SystemExit("No feature rows produced. Record more data first.")

    df = add_labels(pd.DataFrame(rows), horizons_s)
    seconds = (last_ts - first_ts) / 1000
    lat_us = np.array(latencies_ns) / 1000
    book_us = np.array(book_latencies_ns) / 1000
    meta = {
        "rows": int(len(df)),
        "hours_of_data": round(seconds / 3600, 2),
        "segments": int(df["segment"].nunique()),
        "book_updates": n_book,
        "trades": n_trades,
        "book_updates_per_second": round(n_book / max(seconds, 1e-9), 2),
        "level_changes_per_second": round(state.n_level_changes / max(seconds, 1e-9), 1),
        "book_update_latency_us": {
            "p50": round(float(np.percentile(book_us, 50)), 1),
            "p99": round(float(np.percentile(book_us, 99)), 1),
        },
        "feature_latency_us": {
            "p50": round(float(np.percentile(lat_us, 50)), 1),
            "p99": round(float(np.percentile(lat_us, 99)), 1),
        },
        "class_balance": {
            f"{h}s": {name: round(float((df[f"y_{h}s"].dropna() == k).mean()), 3)
                      for k, name in {0: "STABLE", 1: "DOWN", 2: "UP"}.items()}
            for h in horizons_s
        },
    }
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=DATASET_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    t0 = time.time()
    df, meta = build_dataset(args.recordings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "dataset_summary.json").write_text(json.dumps(meta, indent=2))

    print(f"Built {meta['rows']:,} rows from {meta['hours_of_data']} hours of data in {time.time() - t0:.0f}s")
    print(f"Book updates/s: {meta['book_updates_per_second']}  |  level changes/s: {meta['level_changes_per_second']}")
    print(f"Book update latency: p50 {meta['book_update_latency_us']['p50']}us, "
          f"p99 {meta['book_update_latency_us']['p99']}us")
    print(f"Feature latency per update: p50 {meta['feature_latency_us']['p50']}us, "
          f"p99 {meta['feature_latency_us']['p99']}us")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
