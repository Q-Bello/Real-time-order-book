# Shared, correctly synchronised Binance order book feed.
# Follows Binance's documented procedure for maintaining a local order book:
#   1. Open the diff-depth stream and buffer events.
#   2. Fetch a REST snapshot.
#   3. Drop buffered events already covered by the snapshot, then apply the rest in order.
#   4. If an update is ever missed, start again from step 1.
# Every script (dataset generation, live prediction, dashboard) uses this feed,
# so training and live features are computed from the same book.

import asyncio
import json
import urllib.request

import websockets

from utils.order_book_cache import OrderBook, OrderBookGapError

WS_URI = "wss://stream.binance.com:9443/ws/{symbol}@depth@100ms"
SNAPSHOT_URL = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=1000"


def _fetch_snapshot(symbol: str) -> dict:
    with urllib.request.urlopen(SNAPSHOT_URL.format(symbol=symbol.upper()), timeout=10) as resp:
        return json.loads(resp.read())


async def synced_order_book(order_book: OrderBook, symbol: str = "btcusdt"):
    """Async generator yielding (event_time_ms, resynced) after each applied update.

    The OrderBook passed in is updated in place, so callers read it directly.
    Reconnects and re-syncs automatically on a gap or dropped connection.
    `resynced` is True on the first update after a (re)sync, so callers can reset
    any state that compares the book with its previous state (OFI, VOI).
    """
    while True:
        try:
            async with websockets.connect(WS_URI.format(symbol=symbol.lower())) as ws:
                # Start buffering before the snapshot so nothing is missed in between.
                buffer = [json.loads(await ws.recv())]
                snapshot = await asyncio.to_thread(_fetch_snapshot, symbol)
                # The snapshot must be newer than the first buffered event.
                while int(snapshot["lastUpdateId"]) < int(buffer[0]["U"]):
                    buffer.append(json.loads(await ws.recv()))
                    snapshot = await asyncio.to_thread(_fetch_snapshot, symbol)

                order_book.load_snapshot(snapshot)
                resynced = True
                for event in buffer:
                    if order_book.apply_diff(event):
                        yield int(event["E"]), resynced
                        resynced = False

                while True:
                    event = json.loads(await ws.recv())
                    if order_book.apply_diff(event):
                        yield int(event["E"]), resynced
                        resynced = False
        except OrderBookGapError as e:
            print(f"\n[feed] {e}. Re-syncing from a fresh snapshot...")
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"\n[feed] Connection issue ({e}). Reconnecting...")
            await asyncio.sleep(1)
