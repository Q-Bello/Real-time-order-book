from typing import Dict, List, Optional, Tuple


class OrderBookGapError(Exception):
    """Raised when a diff event does not follow on from the last applied update."""


# This class manages a local, in-memory copy of the order book.
# Binance's diff-depth stream only sends changes, so the book must be seeded from a
# REST snapshot and each diff applied in update-ID order. Without that, levels that
# never change are never seen and stale levels are never removed, so the best bid
# and ask (and every feature built on them) are wrong.
class OrderBook:
    def __init__(self, depth: int = 20):
        # Key: price (float), Value: quantity (float)
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.depth: int = depth
        # ID of the last update applied; None until a snapshot has been loaded.
        self.last_update_id: Optional[int] = None

    def load_snapshot(self, snapshot: Dict):
        """Seed the book from a REST /api/v3/depth response."""
        self.bids = {float(p): float(q) for p, q in snapshot["bids"] if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in snapshot["asks"] if float(q) > 0}
        self.last_update_id = int(snapshot["lastUpdateId"])

    def apply_diff(self, event: Dict) -> bool:
        """Apply one diff-depth event.

        Returns False if the event is stale (already covered by the snapshot) and was
        skipped. Raises OrderBookGapError if updates were missed, in which case the
        caller must re-sync from a fresh snapshot.
        """
        if self.last_update_id is None:
            raise OrderBookGapError("Snapshot not loaded")
        first_id, final_id = int(event["U"]), int(event["u"])
        if final_id <= self.last_update_id:
            return False
        if first_id > self.last_update_id + 1:
            raise OrderBookGapError(
                f"Gap in updates: expected {self.last_update_id + 1}, got {first_id}"
            )
        self.update(event)
        self.last_update_id = final_id
        return True

    def update(self, data: Dict):
        # Process both bids ('b') and asks ('a') from the incoming message.
        for side, book in [('b', self.bids), ('a', self.asks)]:
            for price_str, qty_str in data.get(side, []):
                price, qty = float(price_str), float(qty_str)
                # A quantity of zero means the level has been removed.
                if qty == 0:
                    book.pop(price, None)
                else:
                    book[price] = qty

    def get_top_levels(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        top_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:self.depth]
        top_asks = sorted(self.asks.items(), key=lambda x: x[0])[:self.depth]
        return top_bids, top_asks
