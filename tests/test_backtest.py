import pandas as pd
import pytest

from lobpred.backtest import run_backtest, summarise


def test_long_trade_costs_and_holding_period():
    df = pd.DataFrame({
        "ts": [0, 1000, 2000, 3000],
        "segment": 0,
        "bid": [99.99, 100.99, 101.99, 102.99],
        "ask": [100.01, 101.01, 102.01, 103.01],
        "mid": [100.0, 101.0, 102.0, 103.0],
        "score": [0.5, 0.0, 0.0, 0.0],
    })
    trades = run_backtest(df, horizon_s=1, threshold=0.2)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["side"] == 1 and t["exit_ts"] == 1000
    assert t["gross_bps"] == pytest.approx(100.0)
    assert t["after_spread_bps"] == pytest.approx((100.99 / 100.01 - 1) * 1e4)
    summary = summarise(trades, hours=1, fees_bps=(1.0,))
    assert summary["mean_net_bps_fee_1"] == pytest.approx(t["after_spread_bps"] - 2, abs=1e-3)


def test_short_and_no_trade_across_segments():
    df = pd.DataFrame({"ts": [0, 1000], "segment": [0, 1], "bid": [99.0, 98.0], "ask": [99.1, 98.1],
                       "mid": [99.05, 98.05], "score": [-0.9, 0.0]})
    assert run_backtest(df, horizon_s=1, threshold=0.2).empty
