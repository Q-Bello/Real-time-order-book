# Collects live BTC/USDT order book data from Binance, computes features and labels,
# and saves a training dataset to data_stream/lob_features.csv.
#
# Rows are written to disk in batches, so stopping early with Ctrl+C keeps everything
# collected so far. At Binance's 100ms update rate, 20,000 rows takes roughly 35-40 minutes.
#
# Usage (from the project root):
#   python -m data_stream.generate_dataset --rows 20000

import argparse
import asyncio
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.order_book_cache import OrderBook
from utils.binance_feed import synced_order_book
from microstructure.data_labeller import FeatureExtractor

DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lob_features.csv")
SAVE_EVERY = 500


async def collect_data(target_rows, window, output, append):
    order_book = OrderBook(depth=20)
    extractor = FeatureExtractor(window=window)
    if os.path.exists(output) and not append:
        os.remove(output)

    rows_collected = 0
    print("Connecting to Binance and syncing the order book...")
    try:
        async for event_time, resynced in synced_order_book(order_book):
            if resynced:
                extractor.reset()
            row = extractor.update(order_book.bids, order_book.asks, timestamp=event_time)
            if row is None:
                continue
            rows_collected += 1
            if rows_collected % SAVE_EVERY == 0:
                extractor.save_to_csv(output, append=True)
            sys.stdout.write(f"\rCollected {rows_collected}/{target_rows} rows")
            sys.stdout.flush()
            if rows_collected >= target_rows:
                break
    finally:
        extractor.save_to_csv(output, append=True)

    print(f"\n\nSaved to {output}")
    df = pd.read_csv(output)
    print(f"Total rows in file: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts(normalize=True).rename({0: "STABLE", 1: "DOWN", 2: "UP"}).round(3))
    if "horizon_ms" in df:
        print(f"Median label horizon: {df['horizon_ms'].median() / 1000:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Collect a labelled order book dataset.")
    parser.add_argument("--rows", type=int, default=20000, help="Number of labelled rows to collect")
    parser.add_argument("--window", type=int, default=30, help="Label horizon in book updates")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--append", action="store_true", help="Add to an existing dataset")
    args = parser.parse_args()
    try:
        asyncio.run(collect_data(args.rows, args.window, args.output, args.append))
    except KeyboardInterrupt:
        print("\nStopped early. Rows collected so far have been saved.")


if __name__ == "__main__":
    main()
