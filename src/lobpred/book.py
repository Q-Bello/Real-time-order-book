"""Local Level 2 order book maintained from a REST snapshot plus Binance diff events.

Prices are kept in sorted maps, so the best bid/ask lookup is O(1) and reading the
top k levels is O(k), independent of how deep the book is. Diffs must be applied in
update-ID order; a missing update makes the local book wrong, so it raises GapError.
"""

from __future__ import annotations

from sortedcontainers import SortedDict


class GapError(Exception):
    """An update was missed, so the book must be re-synced from a fresh snapshot."""


class OrderBook:
    def __init__(self) -> None:
        self.bids: SortedDict = SortedDict()  # ascending, best bid is the last key
        self.asks: SortedDict = SortedDict()  # ascending, best ask is the first key
        self.last_update_id: int | None = None

    @property
    def synced(self) -> bool:
        return self.last_update_id is not None

    def load_snapshot(self, snapshot: dict) -> None:
        self.bids = SortedDict({float(p): float(q) for p, q in snapshot["bids"] if float(q) > 0})
        self.asks = SortedDict({float(p): float(q) for p, q in snapshot["asks"] if float(q) > 0})
        self.last_update_id = int(snapshot["lastUpdateId"])

    def apply_diff(self, event: dict) -> bool:
        """Apply one diff event. Returns False if it was already covered by the snapshot."""
        if self.last_update_id is None:
            raise GapError("no snapshot loaded")
        first_id, final_id = int(event["U"]), int(event["u"])
        if final_id <= self.last_update_id:
            return False
        if first_id > self.last_update_id + 1:
            raise GapError(f"expected update {self.last_update_id + 1}, got {first_id}")
        self._apply_side(self.bids, event.get("b", ()))
        self._apply_side(self.asks, event.get("a", ()))
        self.last_update_id = final_id
        return True

    @staticmethod
    def _apply_side(side: SortedDict, levels) -> None:
        for price, qty in levels:
            p, q = float(price), float(qty)
            if q == 0.0:
                side.pop(p, None)
            else:
                side[p] = q

    def best_bid(self) -> tuple[float, float]:
        return self.bids.peekitem(-1)

    def best_ask(self) -> tuple[float, float]:
        return self.asks.peekitem(0)

    def top_bids(self, k: int) -> list[tuple[float, float]]:
        """Best k bids, best first."""
        return self.bids.items()[-k:][::-1]

    def top_asks(self, k: int) -> list[tuple[float, float]]:
        """Best k asks, best first."""
        return self.asks.items()[:k]

    def is_valid(self) -> bool:
        return bool(self.bids) and bool(self.asks) and self.best_bid()[0] < self.best_ask()[0]
