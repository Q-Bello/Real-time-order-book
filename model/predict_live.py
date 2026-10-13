# Optional console script: streams live data and prints the model's prediction for
# every book update, without launching the dashboard.
#
# Usage (from the project root):
#   python -m model.predict_live

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.order_book_cache import OrderBook
from utils.binance_feed import synced_order_book
from microstructure.data_labeller import FeatureExtractor
from model.predictor import Predictor


async def predict_live(predictor):
    order_book = OrderBook(depth=20)
    extractor = FeatureExtractor()
    print("Syncing order book and streaming live predictions...")
    async for event_time, resynced in synced_order_book(order_book):
        if resynced:
            extractor.reset()
        # Live prediction needs features for *now*, not a labelled row from the past.
        row = extractor.compute_features(order_book.bids, order_book.asks, timestamp=event_time)
        if row is None:
            continue
        label, confidence = predictor.predict(row)
        print(
            f"\rPrediction: {label:<6} (conf {confidence:.0%}) | Mid: {row['mid_price']:.2f} "
            f"| Spread: {row['spread']:.2f} | OFI: {row['ofi']:+.3f} | VOI: {row['voi']:+.3f}   ",
            end="",
        )


if __name__ == "__main__":
    try:
        predictor = Predictor()
    except FileNotFoundError:
        sys.exit("No trained model found. Run generate_dataset and train_model first.")
    try:
        asyncio.run(predict_live(predictor))
    except KeyboardInterrupt:
        print("\nPrediction stopped by user.")
