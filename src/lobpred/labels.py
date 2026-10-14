"""Forward-looking labels at fixed time horizons.

For each row at time t and horizon h, the future mid is the mid at the first book
update at or after t + h. Labels are only computed within a segment (a stretch of
continuous data), so a future price is never taken from the other side of a gap.

fwd_ret_{h}s_bps   future mid return in basis points
y_{h}s         2 = UP, 1 = DOWN, 0 = STABLE, using a threshold of half the spread at t
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_NAMES = {0: "STABLE", 1: "DOWN", 2: "UP"}


def add_labels(df: pd.DataFrame, horizons_s, threshold_spreads: float = 0.5) -> pd.DataFrame:
    clash = {c for h in horizons_s for c in (f"fwd_ret_{h}s_bps", f"y_{h}s")} & set(df.columns)
    if clash:
        raise ValueError(f"Label columns would overwrite existing columns: {sorted(clash)}")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    future = df[["ts", "mid", "segment"]].rename(columns={"ts": "ts_future", "mid": "mid_future"})
    for h in horizons_s:
        target = df[["ts", "segment"]].copy()
        target["ts_target"] = target["ts"] + int(h * 1000)
        # Both frames are already in time order; merge_asof keeps the left frame's row order.
        merged = pd.merge_asof(
            target,
            future,
            left_on="ts_target",
            right_on="ts_future",
            by="segment",
            direction="forward",
        )
        fut = merged["mid_future"].to_numpy()
        move = fut - df["mid"].to_numpy()
        thr = threshold_spreads * df["spread"].to_numpy()
        df[f"fwd_ret_{h}s_bps"] = (fut / df["mid"].to_numpy() - 1) * 1e4
        y = np.where(move > thr, 2, np.where(move < -thr, 1, 0)).astype(float)
        y[np.isnan(fut)] = np.nan
        df[f"y_{h}s"] = y
    return df
