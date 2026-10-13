from microstructure.data_labeller import FeatureExtractor


def book_at(mid, bid_qty=1.0, ask_qty=1.0):
    return {mid - 0.5: bid_qty}, {mid + 0.5: ask_qty}


def test_label_describes_future_of_the_row_it_is_attached_to():
    window = 3
    ex = FeatureExtractor(window=window)
    mids = [100, 100, 101, 102, 103, 103, 103]  # price rises after t=1
    rows = []
    for t, m in enumerate(mids):
        bids, asks = book_at(m)
        row = ex.update(bids, asks, timestamp=t)
        if row:
            rows.append(row)
    first = rows[0]
    # First feature row is t=1 (t=0 only seeds the previous-book state).
    assert first["timestamp"] == 1
    assert first["mid_price"] == 100
    # Mean of the next 3 mids (101, 102, 103) is above 100 + half the 1.0 spread.
    assert first["label"] == 2
    assert first["horizon_ms"] == 3


def test_ofi_detects_change_when_book_is_mutated_in_place():
    # The live feed mutates the same dicts; previous state must be a copy.
    ex = FeatureExtractor(window=1)
    bids, asks = {99.5: 1.0}, {100.5: 1.0}
    ex.compute_features(bids, asks)
    bids[99.5] = 3.0  # extra bid size arrives at the best bid
    row = ex.compute_features(bids, asks)
    assert row["ofi"] == 2.0


def test_microprice_offset_is_scale_free():
    ex = FeatureExtractor()
    ex.compute_features({99.5: 1.0}, {100.5: 1.0})
    row = ex.compute_features({99.5: 3.0}, {100.5: 1.0})
    # More size on the bid pushes the weighted mid toward the ask.
    assert 0 < row["microprice_offset"] <= 0.5
