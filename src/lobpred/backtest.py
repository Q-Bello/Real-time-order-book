"""Turns out-of-sample predictions into trades and measures what is left after costs.

Rules, fixed in advance rather than tuned on the results:
  - Signal = P(UP) - P(DOWN). Enter when |signal| >= threshold, long if positive.
  - Hold for exactly the prediction horizon, then exit. One position at a time.
  - Execution crosses the spread both ways (buy at the ask, sell at the bid), plus an
    exchange fee per side. Fee levels are illustrative, not a claim about any account.
  - Shorts are treated symmetrically for simplicity (as on a perpetual future).

The key output is the break-even fee: the per-side cost at which the average trade
makes zero. Comparing it with real fees answers whether a predictive edge is tradeable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
FEES_BPS = (0.0, 1.0, 7.5)  # per side


def run_backtest(df: pd.DataFrame, horizon_s: float, threshold: float) -> pd.DataFrame:
    """df needs ts, segment, bid, ask, mid and score columns, in time order."""
    ts = df["ts"].to_numpy()
    seg = df["segment"].to_numpy()
    bid, ask, mid = (df[c].to_numpy() for c in ("bid", "ask", "mid"))
    score = df["score"].to_numpy()

    exit_idx = np.searchsorted(ts, ts + int(horizon_s * 1000), side="left")
    n = len(df)
    valid = exit_idx < n
    valid[valid] &= seg[exit_idx[valid]] == seg[valid]

    trades = []
    i = 0
    while i < n:
        if valid[i] and abs(score[i]) >= threshold:
            j = exit_idx[i]
            side = 1 if score[i] > 0 else -1
            entry_px = ask[i] if side > 0 else bid[i]
            exit_px = bid[j] if side > 0 else ask[j]
            trades.append((ts[i], ts[j], side, score[i],
                           side * (mid[j] / mid[i] - 1) * 1e4,
                           side * (exit_px / entry_px - 1) * 1e4))
            i = j
        else:
            i += 1
    return pd.DataFrame(trades, columns=["entry_ts", "exit_ts", "side", "score",
                                         "gross_bps", "after_spread_bps"])


def summarise(trades: pd.DataFrame, hours: float, fees_bps=FEES_BPS) -> dict:
    if trades.empty:
        return {"trades": 0}
    after = trades["after_spread_bps"]
    out = {
        "trades": int(len(trades)),
        "trades_per_hour": round(len(trades) / max(hours, 1e-9), 1),
        "hit_rate": round(float((trades["gross_bps"] > 0).mean()), 3),
        "mean_gross_bps": round(float(trades["gross_bps"].mean()), 3),
        "mean_after_spread_bps": round(float(after.mean()), 3),
        "breakeven_fee_bps_per_side": round(float(after.mean() / 2), 3),
    }
    for fee in fees_bps:
        out[f"mean_net_bps_fee_{fee:g}"] = round(float((after - 2 * fee).mean()), 3)
    return out
