"""Turns a stream of raw records (live or replayed) into a synchronised market state.

Record format, shared by the recorder, replay and live paths:
    {"r": local receive time in ms, "k": "depth" | "trade" | "snap", "d": Binance payload}

Using the same state machine for live trading and offline replay guarantees the
features a model is trained on are computed exactly as they will be in production.
"""

from __future__ import annotations

from dataclasses import dataclass

from lobpred.book import GapError, OrderBook

MAX_PENDING = 10_000


@dataclass
class Update:
    """What changed after one record."""

    kind: str  # "book" or "trade"
    ts: int  # exchange time in ms
    resynced: bool = False  # first book state after a (re)sync
    data: dict | None = None  # the trade payload for kind == "trade"


class MarketState:
    def __init__(self) -> None:
        self.book = OrderBook()
        self._pending: list[dict] = []
        self._just_synced = False
        self.n_resyncs = 0
        self.n_level_changes = 0

    def on_record(self, record: dict) -> Update | None:
        kind, data = record["k"], record["d"]

        if kind == "trade":
            return Update("trade", int(data["T"]), data=data)

        if kind == "snap":
            self.book.load_snapshot(data)
            self.n_resyncs += 1
            ts = None
            for event in self._pending:
                try:
                    if self._apply(event):
                        ts = int(event["E"])
                except GapError:
                    # The snapshot does not join up with the buffered events; wait for the next.
                    self.book.last_update_id = None
                    break
            self._pending = []
            if self.book.synced:
                self._just_synced = True
                if ts is not None and self.book.is_valid():
                    return self._book_update(ts)
            return None

        if kind == "depth":
            if not self.book.synced:
                self._pending.append(data)
                del self._pending[:-MAX_PENDING]
                return None
            try:
                applied = self._apply(data)
            except GapError:
                self.book.last_update_id = None
                self._pending = [data]
                return None
            if applied and self.book.is_valid():
                return self._book_update(int(data["E"]))
            return None

        raise ValueError(f"unknown record kind {kind!r}")

    def _apply(self, event: dict) -> bool:
        applied = self.book.apply_diff(event)
        if applied:
            self.n_level_changes += len(event.get("b", ())) + len(event.get("a", ()))
        return applied

    def _book_update(self, ts: int) -> Update:
        resynced, self._just_synced = self._just_synced, False
        return Update("book", ts, resynced=resynced)
