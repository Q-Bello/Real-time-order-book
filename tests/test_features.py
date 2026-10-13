import numpy as np
import pytest

from lobpred.book import OrderBook
from lobpred.features import FEATURES, FeatureEngine, level_ofi
from lobpred.labels import add_labels


def test_level_one_ofi_cases():
    prev_b, prev_a = [(100.0, 5.0)], [(101.0, 5.0)]
    # Same prices: change in queue sizes.
    assert level_ofi([(100.0, 7.0)], [(101.0, 4.0)], prev_b, prev_a)[0] == pytest.approx(2.0 - (-1.0))
    # Bid price improves: the whole new bid queue counts as buying pressure.
    assert level_ofi([(100.5, 3.0)], [(101.0, 5.0)], prev_b, prev_a)[0] == pytest.approx(3.0)
    # Ask price drops: the whole new ask queue counts as selling pressure.
    assert level_ofi([(100.0, 5.0)], [(100.5, 2.0)], prev_b, prev_a)[0] == pytest.approx(-2.0)


def _book(mid, bid_qty=1.0, ask_qty=1.0, levels=10):
    b = OrderBook()
    b.load_snapshot({"lastUpdateId": 1,
                     "bids": [[f"{mid - 0.005 - i * 0.01:.3f}", str(bid_qty)] for i in range(levels)],
                     "asks": [[f"{mid + 0.005 + i * 0.01:.3f}", str(ask_qty)] for i in range(levels)]})
    return b


def test_engine_warms_up_then_emits_all_features():
    engine = FeatureEngine()
    rows = [engine.on_book(_book(100.0), ts=i * 100) for i in range(60)]
    assert rows[0] is None
    assert set(FEATURES) <= set(rows[-1])
    assert all(np.isfinite(rows[-1][f]) for f in FEATURES)


def test_trade_flow_imbalance_sign():
    engine = FeatureEngine()
    for i in range(60):
        engine.on_trade({"T": i * 100, "q": "1.0", "m": False})  # aggressive buys
        row = engine.on_book(_book(100.0), ts=i * 100)
    assert row["tfi_1s"] == pytest.approx(1.0)


def test_features_never_share_names_with_labels():
    df = add_labels(__import__("pandas").DataFrame(
        {"ts": [0, 1000], "mid": [1.0, 1.0], "spread": [0.01, 0.01], "segment": [0, 0]}), [1])
    label_cols = {c for c in df.columns if c.startswith(("fwd_ret_", "y_"))}
    assert not label_cols & set(FEATURES)
