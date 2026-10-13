# Turns a stream of order book states into model features and labels.
#
# Features are computed from the book at time t. The label for that row describes
# what the mid-price does over the NEXT `window` updates (t+1 ... t+window), so a
# row can only be emitted once those future updates have arrived. Features and
# label therefore always refer to the same moment, with no look-ahead.
#
# Label: compare the average future mid-price with the mid-price at t, using a
# threshold of half the spread at t, so the definition adapts to market conditions.
#   0 = STABLE, 1 = DOWN, 2 = UP

import csv
import os
from collections import deque

import numpy as np

from microstructure.feature_engineering import (
    calculate_mid_price,
    calculate_spread,
    calculate_ofi,
    calculate_weighted_mid_price,
    calculate_voi,
)

# The only columns the model is trained on. Raw prices are deliberately excluded:
# BTC's price level drifts over time, so a model can "learn" the level of the
# training period rather than anything about order flow.
FEATURE_COLUMNS = ["spread", "ofi", "voi", "microprice_offset"]
LABEL_NAMES = {0: "STABLE", 1: "DOWN", 2: "UP"}


class FeatureExtractor:
    def __init__(self, window=30, threshold_spreads=0.5):
        # Forward-looking horizon for labels, in number of book updates.
        self.window = window
        self.threshold_spreads = threshold_spreads
        self.prev_bids = {}
        self.prev_asks = {}
        # Rows waiting for their future window to arrive before they can be labelled.
        self.pending = deque()
        self.feature_rows = []

    def reset(self):
        """Forget previous-book state, e.g. after the feed re-syncs."""
        self.prev_bids = {}
        self.prev_asks = {}
        self.pending.clear()

    def compute_features(self, current_bids, current_asks, timestamp=None):
        """Features for the current book state (no label). Used for live prediction."""
        if not current_bids or not current_asks:
            return None
        if not self.prev_bids:
            self.prev_bids, self.prev_asks = dict(current_bids), dict(current_asks)
            return None

        sorted_bids = sorted(current_bids.items(), key=lambda x: -x[0])
        sorted_asks = sorted(current_asks.items(), key=lambda x: x[0])

        mid_price = calculate_mid_price(sorted_bids, sorted_asks)
        spread = calculate_spread(sorted_bids, sorted_asks)
        wmp = calculate_weighted_mid_price(sorted_bids, sorted_asks)
        ofi = calculate_ofi(current_bids, current_asks, self.prev_bids, self.prev_asks)
        voi = calculate_voi(current_bids, current_asks, self.prev_bids, self.prev_asks)

        # Copy, because the order book dicts are mutated in place by the feed.
        self.prev_bids, self.prev_asks = dict(current_bids), dict(current_asks)

        if mid_price is None or spread is None or spread <= 0 or wmp is None:
            return None

        return {
            "timestamp": timestamp,
            "mid_price": mid_price,
            "weighted_mid_price": wmp,
            "spread": spread,
            "ofi": ofi,
            "voi": voi,
            # Where the volume-weighted price sits inside the spread: -0.5 (at the bid)
            # to +0.5 (at the ask). Scale-free, unlike the raw weighted mid-price.
            "microprice_offset": (wmp - mid_price) / spread,
        }

    def update(self, current_bids, current_asks, timestamp=None):
        """Feed one book state. Returns a labelled row once one is ready, else None."""
        row = self.compute_features(current_bids, current_asks, timestamp)
        if row is None:
            return None

        self.pending.append(row)
        # Need the event itself plus `window` later updates.
        if len(self.pending) < self.window + 1:
            return None

        event = self.pending.popleft()
        future = list(self.pending)[: self.window]
        average_future_price = np.mean([r["mid_price"] for r in future])
        threshold = event["spread"] * self.threshold_spreads

        label = 0
        if average_future_price > event["mid_price"] + threshold:
            label = 2
        elif average_future_price < event["mid_price"] - threshold:
            label = 1

        labelled = dict(event)
        labelled["label"] = label
        if timestamp is not None and event["timestamp"] is not None:
            # How far ahead this label actually looked, in milliseconds.
            labelled["horizon_ms"] = future[-1]["timestamp"] - event["timestamp"]
        self.feature_rows.append(labelled)
        return labelled

    def save_to_csv(self, filename="orderbook_features.csv", append=False):
        """Write accumulated rows to CSV and clear them from memory."""
        if not self.feature_rows:
            return
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        write_header = not (append and os.path.exists(filename))
        with open(filename, "a" if append else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.feature_rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(self.feature_rows)
        self.feature_rows = []
