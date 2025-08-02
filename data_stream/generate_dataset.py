import asyncio
import websockets
import json
from utils.order_book_cache import OrderBook
from microstructure.data_labeller import FeatureExtractor

order_book = OrderBook(depth=20)
extractor = FeatureExtractor(window=10)

async def collect_data():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
    async with websockets.connect(uri) as ws:
        count = 0
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            order_book.update(data)
            current_bids, current_asks = order_book.bids, order_book.asks

            row = extractor.update(current_bids, current_asks)
            if row:
                print(f"[{count}] {row}")
                count += 1

            if count >= 100:
                break

    extractor.save_to_csv("lob_features.csv")
    print("✅ Saved 100 rows to lob_features.csv")

if __name__ == "__main__":
    asyncio.run(collect_data())
