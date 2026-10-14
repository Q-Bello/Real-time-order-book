"""Live Binance feed producing records in the shared format.

Follows Binance's procedure for a correct local book: buffer diff events, fetch a REST
snapshot that is at least as new as the first buffered event, then apply diffs in
order. Each diff's first update ID must follow the previous diff's last ID; if it
does not, an update was lost and a new snapshot is fetched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from collections.abc import AsyncIterator

import websockets

from lobpred.config import SNAPSHOT_URL, SYMBOL, WS_URL

log = logging.getLogger(__name__)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _fetch_snapshot(symbol: str) -> dict:
    with urllib.request.urlopen(SNAPSHOT_URL.format(S=symbol.upper()), timeout=10) as resp:
        return json.loads(resp.read())


async def live_records(symbol: str = SYMBOL) -> AsyncIterator[dict]:
    url = WS_URL.format(s=symbol.lower())
    while True:
        try:
            async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
                last_u: int | None = None
                async for raw in ws:
                    msg = json.loads(raw)
                    data = msg["data"]
                    if msg["stream"].endswith("aggTrade"):
                        yield {"r": _now_ms(), "k": "trade", "d": data}
                        continue

                    yield {"r": _now_ms(), "k": "depth", "d": data}
                    if last_u is not None and int(data["U"]) == last_u + 1:
                        last_u = int(data["u"])
                        continue

                    if last_u is not None:
                        log.warning("Missed book updates (expected %s, got %s); re-syncing", last_u + 1, data["U"])
                    last_u = int(data["u"])
                    snapshot = await asyncio.to_thread(_fetch_snapshot, symbol)
                    while int(snapshot["lastUpdateId"]) < int(data["U"]):
                        await asyncio.sleep(0.25)
                        snapshot = await asyncio.to_thread(_fetch_snapshot, symbol)
                    yield {"r": _now_ms(), "k": "snap", "d": snapshot}
        except (websockets.ConnectionClosed, OSError) as exc:
            log.warning("Connection problem (%s); reconnecting in 1s", exc)
            await asyncio.sleep(1)
