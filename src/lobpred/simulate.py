"""Synthetic market generator, used only to test the pipeline end to end.

Produces records in exactly the recorder's format: an initial snapshot, 100ms book
diffs with consecutive update IDs, aggregated trades, and (optionally) a dropped
update followed by a re-sync, so every code path can be exercised offline.

The dynamics are a toy queue model: a persistent latent order-flow signal tilts trade
direction and where limit orders arrive, and the price ticks when a best queue is
exhausted. Results from simulated data say nothing about real markets.

Usage:
    python -m lobpred.simulate --minutes 30 --out data/sim
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

TICK = 0.01
LEVELS = 25


class Simulator:
    def __init__(self, seed: int = 0, start_price: float = 100.0, start_ms: int = 1_700_000_000_000):
        self.rng = np.random.default_rng(seed)
        self.t = start_ms
        self.update_id = 1_000
        self.trade_id = 1
        self.signal = 0.0
        mid_ticks = round(start_price / TICK)
        self.bids = {mid_ticks - i: self._fresh_qty() for i in range(1, LEVELS + 1)}
        self.asks = {mid_ticks + i: self._fresh_qty() for i in range(1, LEVELS + 1)}

    def _fresh_qty(self) -> float:
        return round(float(self.rng.gamma(2.0, 1.0)) + 0.1, 4)

    @staticmethod
    def _px(ticks: int) -> str:
        return f"{ticks * TICK:.2f}"

    def snapshot(self) -> dict:
        return {
            "lastUpdateId": self.update_id,
            "bids": [[self._px(p), str(q)] for p, q in sorted(self.bids.items(), reverse=True)],
            "asks": [[self._px(p), str(q)] for p, q in sorted(self.asks.items())],
        }

    def step(self) -> tuple[dict, list[dict]]:
        """Advance 100ms. Returns (depth diff event, trades)."""
        rng = self.rng
        self.t += 100
        self.signal = 0.95 * self.signal + rng.normal(0, 0.25)
        p_buy = 1 / (1 + np.exp(-2.5 * self.signal))
        changed_b: dict[int, float] = {}
        changed_a: dict[int, float] = {}
        trades = []

        for _ in range(rng.poisson(1.5)):
            buy = rng.random() < p_buy
            book, changed = (self.asks, changed_a) if buy else (self.bids, changed_b)
            best = min(book) if buy else max(book)
            qty = min(round(float(rng.exponential(0.6)) + 0.001, 4), book[best])
            book[best] = round(book[best] - qty, 4)
            if book[best] <= 1e-9:
                del book[best]
                changed[best] = 0.0
            else:
                changed[best] = book[best]
            trades.append({"e": "aggTrade", "E": self.t, "T": self.t, "a": self.trade_id,
                           "p": self._px(best), "q": f"{qty:.4f}", "m": not buy})
            self.trade_id += 1

        # Limit order arrivals lean towards the side the signal favours.
        for _ in range(rng.poisson(3)):
            bid_side = rng.random() < 0.5 + 0.25 * np.tanh(self.signal)
            book, changed = (self.bids, changed_b) if bid_side else (self.asks, changed_a)
            best = max(book) if bid_side else min(book)
            spread_ticks = min(self.asks) - max(self.bids)
            offset = int(rng.integers(0, 5))
            if spread_ticks > 1 and rng.random() < 0.6:
                level = best + 1 if bid_side else best - 1  # improve the price
            else:
                level = best - offset if bid_side else best + offset
            book[level] = round(book.get(level, 0.0) + self._fresh_qty() * 0.5, 4)
            changed[level] = book[level]

        # Random cancellations away from the touch.
        for _ in range(rng.poisson(2)):
            bid_side = rng.random() < 0.5
            book, changed = (self.bids, changed_b) if bid_side else (self.asks, changed_a)
            level = list(book)[int(rng.integers(0, len(book)))]
            book[level] = round(book[level] * 0.5, 4)
            changed[level] = book[level]

        # Keep the book deep enough on both sides.
        for book, changed, sign in ((self.bids, changed_b, -1), (self.asks, changed_a, 1)):
            while len(book) < LEVELS:
                far = (min(book) if sign < 0 else max(book)) + sign
                book[far] = self._fresh_qty()
                changed[far] = book[far]

        n_changes = max(len(changed_b) + len(changed_a), 1)
        first = self.update_id + 1
        self.update_id += n_changes
        event = {
            "e": "depthUpdate", "E": self.t, "s": "SIMUSDT", "U": first, "u": self.update_id,
            "b": [[self._px(p), f"{q:.4f}"] for p, q in changed_b.items()],
            "a": [[self._px(p), f"{q:.4f}"] for p, q in changed_a.items()],
        }
        return event, trades


def simulate_records(minutes: float, seed: int = 0, gap_at_minute: float | None = None):
    """Yield records for `minutes` of simulated market, optionally dropping one update."""
    sim = Simulator(seed=seed)
    steps = int(minutes * 600)
    gap_step = int(gap_at_minute * 600) if gap_at_minute is not None else None
    first, _ = sim.step()
    yield {"r": sim.t, "k": "depth", "d": first}
    yield {"r": sim.t, "k": "snap", "d": sim.snapshot()}
    for i in range(steps):
        event, trades = sim.step()
        for tr in trades:
            yield {"r": sim.t, "k": "trade", "d": tr}
        if i == gap_step:
            continue  # simulate a lost message
        yield {"r": sim.t, "k": "depth", "d": event}
        if gap_step is not None and i == gap_step + 1:
            yield {"r": sim.t, "k": "snap", "d": sim.snapshot()}


def write_simulation(path: Path, minutes: float, seed: int = 0, gap_at_minute: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for rec in simulate_records(minutes, seed, gap_at_minute):
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/sim"))
    args = parser.parse_args()
    path = write_simulation(args.out / f"sim_seed{args.seed}.jsonl.gz", args.minutes, args.seed,
                            gap_at_minute=args.minutes / 2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
