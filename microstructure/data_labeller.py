# This is the core of the data pipeline.
# It takes live order book data, calculates all features, and creates a label.
# The labelling is dynamic, based on whether the future price moves more than a fraction
# of the current bid-ask spread, making it adaptive to market volatility.

import csv
from collections import deque
import os
import numpy as np

from microstructure.feature_engineering import (
    calculate_mid_price,
    calculate_spread,
    calculate_ofi,
    calculate_weighted_mid_price,
    calculate_voi
)


class FeatureExtractor:
    def __init__(self, depth=20, window=30):
        self.depth = depth
        # The forward-looking window (in number of messages) for creating labels.
        self.window = window
        # Store the previous state of the book for calculating OFI and VOI.
        self.prev_bids = {}
        self.prev_asks = {}
        # A deque to store the recent history of mid-prices for labelling.
        self.mid_prices = deque(maxlen=window + 1)
        # A list to accumulate rows of features and labels before saving.
        self.feature_rows = []

    def update(self, current_bids, current_asks):
        # Do nothing if the order book is empty.
        if not current_bids or not current_asks: return None
        # On the first run, just store the state and wait for the next update.
        if not self.prev_bids:
            self.prev_bids = current_bids
            self.prev_asks = current_asks
            return None

        # Sort bids and asks for accurate feature calculation.
        sorted_bids = sorted(current_bids.items(), key=lambda x: -x[0])
        sorted_asks = sorted(current_asks.items(), key=lambda x: x[0])

        # Calculate all features for the current state.
        mid_price = calculate_mid_price(sorted_bids, sorted_asks)
        spread = calculate_spread(sorted_bids, sorted_asks)

        # A valid spread is required for our dynamic threshold.
        if mid_price is None or spread is None or spread == 0:
            return None

        wmp = calculate_weighted_mid_price(sorted_bids, sorted_asks)
        ofi = calculate_ofi(current_bids, current_asks, self.prev_bids, self.prev_asks)
        voi = calculate_voi(current_bids, current_asks, self.prev_bids, self.prev_asks)

        # Update previous state for the next iteration.
        self.prev_bids = current_bids
        self.prev_asks = current_asks

        if any(v is None for v in [wmp]): return None

        # Add the current mid-price to our historical deque.
        self.mid_prices.append(mid_price)

        # We need a full window of mid-prices to create a label.
        if len(self.mid_prices) < self.window + 1: return None

        # --- Dynamic Labelling Logic ---
        # The price corresponding to our calculated features.
        price_at_event = self.mid_prices[0]
        # The prices that occurred *after* our event.
        future_prices = list(self.mid_prices)[1:]
        average_future_price = np.mean(future_prices)

        # The threshold is a fraction of the spread, making it adaptive to volatility.
        dynamic_threshold = spread * 0.5

        # Default to STABLE (0).
        label = 0
        if average_future_price > price_at_event + dynamic_threshold:
            label = 2  # UP
        elif average_future_price < price_at_event - dynamic_threshold:
            label = 1  # DOWN

        # Assemble the final row with features and the calculated label.
        row = {
            "mid_price": price_at_event,
            "weighted_mid_price": wmp,
            "spread": spread,
            "ofi": ofi,
            "voi": voi,
            "label": label
        }

        self.feature_rows.append(row)
        return row

    def save_to_csv(self, filename="orderbook_features.csv"):
        if not self.feature_rows: return
        # Ensure the directory exists before trying to save the file.
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        # Write the accumulated rows to the specified CSV file.
        keys = self.feature_rows[0].keys()
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.feature_rows)