import csv
from collections import deque
from microstructure.feature_engineering import calculate_mid_price, calculate_spread, calculate_ofi

class FeatureExtractor:
    def __init__(self, depth=20, window=10):
        self.depth = depth
        self.window = window
        self.prev_bids = {}
        self.prev_asks = {}
        self.mid_prices = deque(maxlen=window + 1)
        self.feature_rows = []

    def update(self, current_bids, current_asks):
        if not self.prev_bids:
            self.prev_bids = current_bids
            self.prev_asks = current_asks
            return None

        mid_price = calculate_mid_price(list(current_bids.items()), list(current_asks.items()))
        spread = calculate_spread(list(current_bids.items()), list(current_asks.items()))
        ofi = calculate_ofi(current_bids, current_asks, self.prev_bids, self.prev_asks)

        self.mid_prices.append(mid_price)

        self.prev_bids = current_bids
        self.prev_asks = current_asks

        # Not enough data to label yet
        if len(self.mid_prices) < self.window + 1:
            return None

        current = self.mid_prices[0]
        future = self.mid_prices[-1]

        # Label: 1 if price goes up, 0 if down or same
        label = 1 if future > current else 0

        row = {
            "mid_price": mid_price,
            "spread": spread,
            "ofi": ofi,
            "label": label
        }

        self.feature_rows.append(row)
        return row

    def save_to_csv(self, filename="orderbook_features.csv"):
        if not self.feature_rows:
            return
        keys = self.feature_rows[0].keys()
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.feature_rows)
