"""Incremental microstructure feature engine.

Called once per book update (and once per trade), so the same code serves offline
dataset building and live prediction. Every feature uses only information available
at the moment it is computed.

Features
--------
spread_ticks       Bid-ask spread in ticks.
imb_l1/l5/l10      Depth imbalance (bid - ask) / (bid + ask) over the best 1, 5, 10 levels.
                   imb_l1 also carries the microprice signal: the size-weighted mid equals
                   mid + spread / 2 * imb_l1.
ofi_l1_{n}         Order flow imbalance at the best level (Cont, Kukanov & Stoikov 2014),
                   summed over the last n updates and scaled by typical queue size.
ofi_ml_{n}         Multi-level OFI: the same quantity summed over the best 5 levels
                   (Xu, Gould & Howison 2019).
tfi_{w}            Trade flow imbalance: (buy - sell) / (buy + sell) aggressor volume over
                   the last w seconds.
log_trade_vol_5s   log(1 + traded volume over the last 5 seconds).
past_ret_{w}_bps   Mid-price change over the last w seconds, in basis points.
rv_bps             Realised volatility: std of per-update mid changes (bps) over the last
                   50 updates.
log_depth_l10      log of total size on the best 10 levels of both sides.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from lobpred.book import OrderBook
from lobpred.config import TICK_SIZE

OFI_LEVELS = 5
OFI_WINDOWS = (10, 50)
TRADE_WINDOWS_S = (1, 5)
RETURN_WINDOWS_S = (1, 5)
VOL_WINDOW = 50
DEPTH_EMA_ALPHA = 0.01

FEATURES = [
    "spread_ticks",
    "imb_l1", "imb_l5", "imb_l10",
    *[f"ofi_l1_{n}" for n in OFI_WINDOWS],
    *[f"ofi_ml_{n}" for n in OFI_WINDOWS],
    *[f"tfi_{w}s" for w in TRADE_WINDOWS_S],
    "log_trade_vol_5s",
    *[f"past_ret_{w}s_bps" for w in RETURN_WINDOWS_S],
    "rv_bps",
    "log_depth_l10",
]


def _imbalance(bids, asks) -> float:
    b = sum(q for _, q in bids)
    a = sum(q for _, q in asks)
    return (b - a) / (b + a) if b + a > 0 else 0.0


def level_ofi(bids, asks, prev_bids, prev_asks) -> np.ndarray:
    """Per-level order flow imbalance between two consecutive book states.

    For each level k: bid-side flow minus ask-side flow, where a better price counts
    the whole new queue, an unchanged price counts the change in size, and a worse
    price counts the loss of the whole old queue.
    """
    k = min(len(bids), len(asks), len(prev_bids), len(prev_asks))
    out = np.zeros(OFI_LEVELS)
    for i in range(min(k, OFI_LEVELS)):
        (pb, qb), (ppb, pqb) = bids[i], prev_bids[i]
        (pa, qa), (ppa, pqa) = asks[i], prev_asks[i]
        bid_flow = qb if pb > ppb else (qb - pqb if pb == ppb else -pqb)
        ask_flow = qa if pa < ppa else (qa - pqa if pa == ppa else -pqa)
        out[i] = bid_flow - ask_flow
    return out


class FeatureEngine:
    def __init__(self, tick_size: float = TICK_SIZE) -> None:
        self.tick_size = tick_size
        self.reset()

    def reset(self) -> None:
        """Clear all history, e.g. after the book re-syncs."""
        self._prev_bids: list | None = None
        self._prev_asks: list | None = None
        self._ofi: deque[np.ndarray] = deque(maxlen=max(OFI_WINDOWS))
        self._ofi_sums = {n: np.zeros(OFI_LEVELS) for n in OFI_WINDOWS}
        self._mid_changes = deque(maxlen=VOL_WINDOW)
        self._mids: deque[tuple[int, float]] = deque()
        self._trades: deque[tuple[int, float]] = deque()
        self._depth_ema: float | None = None
        self._last_mid: float | None = None
        self.n_updates = 0

    @property
    def warm(self) -> bool:
        return self.n_updates > max(max(OFI_WINDOWS), VOL_WINDOW)

    def on_trade(self, trade: dict) -> None:
        qty = float(trade["q"])
        # "m" = buyer is the maker, i.e. the aggressor sold.
        self._trades.append((int(trade["T"]), -qty if trade["m"] else qty))

    def on_book(self, book: OrderBook, ts: int) -> dict | None:
        """Compute features for the current book. Returns None until warmed up."""
        bids, asks = book.top_bids(10), book.top_asks(10)
        if not bids or not asks:
            return None
        bid, ask = bids[0][0], asks[0][0]
        mid = (bid + ask) / 2
        spread = ask - bid

        top5_avg = (sum(q for _, q in bids[:OFI_LEVELS]) + sum(q for _, q in asks[:OFI_LEVELS])) / (2 * OFI_LEVELS)
        self._depth_ema = top5_avg if self._depth_ema is None else (
            (1 - DEPTH_EMA_ALPHA) * self._depth_ema + DEPTH_EMA_ALPHA * top5_avg)

        if self._prev_bids is not None:
            new = level_ofi(bids, asks, self._prev_bids, self._prev_asks)
            # Running window sums: add the new update, drop the one leaving each window.
            for n in OFI_WINDOWS:
                if len(self._ofi) >= n:
                    self._ofi_sums[n] -= self._ofi[-n]
                self._ofi_sums[n] += new
            self._ofi.append(new)
        self._prev_bids, self._prev_asks = bids, asks

        if self._last_mid is not None:
            self._mid_changes.append((mid / self._last_mid - 1) * 1e4)
        self._last_mid = mid
        self._mids.append((ts, mid))
        horizon_ms = max(RETURN_WINDOWS_S + TRADE_WINDOWS_S) * 1000
        while len(self._mids) > 1 and self._mids[1][0] <= ts - horizon_ms:
            self._mids.popleft()
        while self._trades and self._trades[0][0] <= ts - horizon_ms:
            self._trades.popleft()

        self.n_updates += 1
        if not self.warm:
            return None

        row = {
            "ts": ts, "bid": bid, "ask": ask, "mid": mid, "spread": spread,
            "spread_ticks": spread / self.tick_size,
            "imb_l1": _imbalance(bids[:1], asks[:1]),
            "imb_l5": _imbalance(bids[:5], asks[:5]),
            "imb_l10": _imbalance(bids, asks),
            "log_depth_l10": math.log(sum(q for _, q in bids) + sum(q for _, q in asks)),
        }

        scale = self._depth_ema or 1.0
        for n in OFI_WINDOWS:
            row[f"ofi_l1_{n}"] = float(self._ofi_sums[n][0]) / scale
            row[f"ofi_ml_{n}"] = float(self._ofi_sums[n].sum()) / (scale * OFI_LEVELS)

        for w in TRADE_WINDOWS_S:
            cutoff = ts - w * 1000
            signed = [s for t, s in self._trades if t > cutoff]
            total = sum(abs(s) for s in signed)
            row[f"tfi_{w}s"] = sum(signed) / total if total > 0 else 0.0
            if w == 5:
                row["log_trade_vol_5s"] = math.log1p(total)

        for w in RETURN_WINDOWS_S:
            cutoff = ts - w * 1000
            past_mid = self._mids[0][1]
            for t, m in self._mids:
                if t > cutoff:
                    break
                past_mid = m
            row[f"past_ret_{w}s_bps"] = (mid / past_mid - 1) * 1e4

        row["rv_bps"] = float(np.std(self._mid_changes))
        return row
