import asyncio
import json
import joblib
import websockets
from utils.order_book_cache import OrderBook
from microstructure.data_labeller import FeatureExtractor
import numpy as np

# Load trained model
xgb_model = joblib.load("/Users/quddusbello/PycharmProjects/real-time-order-book/model/xgboost_model.pkl")

# Initialise cache + extractor
order_book = OrderBook(depth=20)
extractor = FeatureExtractor(window=10)

async def predict_live():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
    async with websockets.connect(uri) as ws:
        print("📡 Streaming live data for prediction...")
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            order_book.update(data)

            current_bids, current_asks = order_book.bids, order_book.asks
            row = extractor.update(current_bids, current_asks)

            if row:
                features = np.array([[row["mid_price"], row["spread"], row["ofi"]]])
                prediction = xgb_model.predict(features)[0]
                label = "📈 UP" if prediction == 1 else "📉 DOWN"
                print(f"{label} | Mid: {row['mid_price']:.2f} | Spread: {row['spread']:.2f} | OFI: {row['ofi']:.2f}")

if __name__ == "__main__":
    asyncio.run(predict_live())

