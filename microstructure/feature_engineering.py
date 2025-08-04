import numpy as np

def calculate_mid_price(bids, asks):
    if not bids or not asks:
        return None
    best_bid = max(bids, key=lambda x: x[0])[0]
    best_ask = min(asks, key=lambda x: x[0])[0]
    return (best_bid + best_ask) / 2

def calculate_spread(bids, asks):
    if not bids or not asks:
        return None
    best_bid = max(bids, key=lambda x: x[0])[0]
    best_ask = min(asks, key=lambda x: x[0])[0]
    return best_ask - best_bid

def calculate_ofi(current_bids, current_asks, prev_bids, prev_asks):
    best_bid = max(current_bids.keys(), default=0)
    best_ask = min(current_asks.keys(), default=float("inf"))

    bid_change = current_bids.get(best_bid, 0) - prev_bids.get(best_bid, 0)
    ask_change = prev_asks.get(best_ask, 0) - current_asks.get(best_ask, 0)

    return bid_change + ask_change

if __name__ == "__main__":
    prev_bids = {100.0: 1.5, 99.5: 2.0}
    prev_asks = {100.5: 1.0, 101.0: 1.8}
    current_bids = {100.0: 2.5, 99.5: 1.0}
    current_asks = {100.5: 0.8, 101.0: 2.0}

    mid = calculate_mid_price(list(current_bids.items()), list(current_asks.items()))
    spread = calculate_spread(list(current_bids.items()), list(current_asks.items()))
    ofi = calculate_ofi(current_bids, current_asks, prev_bids, prev_asks)

    print(f"Mid-price: {mid}")
    print(f"Spread: {spread}")
    print(f"Order Flow Imbalance (OFI): {ofi}")