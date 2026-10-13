"""Record the raw Binance feed to disk for offline research.

Everything (book diffs, snapshots, trades) is written exactly as received, so features
and labels can be rebuilt later with different definitions without re-collecting.

Usage:
    python -m lobpred.record --minutes 60
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from lobpred.config import RAW_DIR, SYMBOL
from lobpred.market import MarketState
from lobpred.stream import live_records


async def record(minutes: float, out_dir: Path, symbol: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{symbol}_{stamp}.jsonl.gz"
    state = MarketState()  # only used to report live statistics
    counts = {"depth": 0, "trade": 0, "snap": 0}
    start = time.monotonic()
    last_print = start
    end = start + minutes * 60

    print(f"Recording {symbol.upper()} for {minutes:g} minutes to {path}")
    with gzip.open(path, "at", compresslevel=6) as f:
        try:
            async for rec in live_records(symbol):
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                counts[rec["k"]] += 1
                state.on_record(rec)
                now = time.monotonic()
                if now - last_print >= 5:
                    elapsed = now - start
                    print(
                        f"\r{elapsed / 60:5.1f} min | book msgs {counts['depth']:>7} "
                        f"| level changes/s {state.n_level_changes / elapsed:6.0f} "
                        f"| trades {counts['trade']:>7} | re-syncs {max(state.n_resyncs - 1, 0)}",
                        end="",
                        flush=True,
                    )
                    last_print = now
                if now >= end:
                    break
        finally:
            elapsed = max(time.monotonic() - start, 1e-9)
            meta = {
                "symbol": symbol.upper(),
                "file": path.name,
                "seconds": round(elapsed, 1),
                "book_messages": counts["depth"],
                "trades": counts["trade"],
                "snapshots": counts["snap"],
                "level_changes_per_second": round(state.n_level_changes / elapsed, 1),
            }
            path.with_suffix("").with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
            print(f"\nSaved {path} ({meta['seconds'] / 60:.1f} min, "
                  f"{meta['level_changes_per_second']:.0f} level changes/s)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=60, help="How long to record (default 60)")
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    parser.add_argument("--symbol", default=SYMBOL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="\n%(levelname)s %(message)s")
    try:
        asyncio.run(record(args.minutes, args.out, args.symbol))
    except KeyboardInterrupt:
        print("\nStopped early; everything recorded so far is saved.")


if __name__ == "__main__":
    main()
