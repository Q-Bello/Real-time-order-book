# This file contains functions to calculate financial microstructure features.
# These features are designed to give the model insight into the state of the order book.

import numpy as np

# Calculates the mid-price, the midpoint between the best bid and best ask.
def calculate_mid_price(bids, asks):
    if not bids or not asks:
        return None
    best_bid_price = bids[0][0]
    best_ask_price = asks[0][0]
    return (best_bid_price + best_ask_price) / 2


# Calculates the spread, the difference between the best ask and best bid.
def calculate_spread(bids, asks):
    if not bids or not asks:
        return None
    best_bid_price = bids[0][0]
    best_ask_price = asks[0][0]
    return best_ask_price - best_bid_price


# Calculates the weighted mid-price, which is weighted by the volume at the top of the book.
def calculate_weighted_mid_price(bids, asks):
    if not bids or not asks:
        return None
    best_bid_price, best_bid_qty = bids[0]
    best_ask_price, best_ask_qty = asks[0]

    # Avoid division by zero if both quantities are zero.
    if (best_bid_qty + best_ask_qty) == 0:
        return (best_bid_price + best_ask_price) / 2

    wmp = (best_bid_price * best_ask_qty + best_ask_price * best_bid_qty) / (best_bid_qty + best_ask_qty)
    return wmp


# Calculates Order Flow Imbalance (OFI), a measure of aggressive buying vs. selling.
def calculate_ofi(current_bids, current_asks, prev_bids, prev_asks):
    prev_best_bid = max(prev_bids.keys()) if prev_bids else 0
    prev_best_ask = min(prev_asks.keys()) if prev_asks else float('inf')

    current_best_bid = max(current_bids.keys()) if current_bids else 0
    current_best_ask = min(current_asks.keys()) if current_asks else float('inf')

    # Calculate flow on the bid side based on price changes.
    bid_flow = 0
    if current_best_bid > prev_best_bid:
        bid_flow = current_bids.get(current_best_bid, 0)
    elif current_best_bid == prev_best_bid:
        bid_flow = current_bids.get(current_best_bid, 0) - prev_bids.get(prev_best_bid, 0)
    else:
        bid_flow = -prev_bids.get(prev_best_bid, 0)

    # Calculate flow on the ask side.
    ask_flow = 0
    if current_best_ask < prev_best_ask:
        ask_flow = current_asks.get(current_best_ask, 0)
    elif current_best_ask == prev_best_ask:
        ask_flow = current_asks.get(current_best_ask, 0) - prev_asks.get(prev_best_ask, 0)
    else:
        ask_flow = -prev_asks.get(prev_best_ask, 0)

    return bid_flow - ask_flow


# Calculates Volume Order Imbalance (VOI), a broader measure of sentiment.
def calculate_voi(current_bids, current_asks, prev_bids, prev_asks):
    # Consider all price levels that existed in either the current or previous state.
    all_bid_levels = set(current_bids.keys()) | set(prev_bids.keys())
    all_ask_levels = set(current_asks.keys()) | set(prev_asks.keys())

    # Sum the changes in volume across all bid levels.
    bid_volume_change = 0
    for price in all_bid_levels:
        delta = current_bids.get(price, 0) - prev_bids.get(price, 0)
        bid_volume_change += delta

    # Sum the changes in volume across all ask levels.
    ask_volume_change = 0
    for price in all_ask_levels:
        delta = current_asks.get(price, 0) - prev_asks.get(price, 0)
        ask_volume_change += delta

    # VOI is the net change in bid volume minus the net change in ask volume.
    return bid_volume_change - ask_volume_change
