# This is an optional script for running live predictions directly in the console.
# It provides a simple, lightweight way to test the model's output without
# launching the full visual dashboard.

import asyncio
import json
import joblib
import websockets
import numpy as np
import pandas as pd
import os

# Add the project root to the system path to allow importing our own modules.
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils.order_book_cache import OrderBook
from microstructure.data_labeller import FeatureExtractor

# --- 1. Setup and Initialization ---
# Build robust file paths to ensure the script can find its files.
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'xgboost_model.pkl')
csv_path = os.path.join(project_root, 'data_stream', 'lob_features.csv')

# Load the trained model and the feature order from the CSV.
try:
    xgb_model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    # It is critical to get the feature order to match the model's training.
    FEATURE_ORDER = df.drop('label', axis=1).columns.tolist()
    print("Model and feature order loaded successfully.")
except FileNotFoundError:
    print("ERROR: Load failed. Run data generation and training scripts first.")
    exit(1)

# Initialise the objects that will manage the data stream.
order_book = OrderBook(depth=20)
extractor = FeatureExtractor(window=30)

# Map numeric labels to display text for the console output.
LABEL_MAP = {
    0: "STABLE",
    1: "DOWN",
    2: "UP"
}


# --- 2. Live Prediction Coroutine ---
async def predict_live():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
    async with websockets.connect(uri) as ws:
        print("\nStreaming live data for prediction...")
        while True:
            # Wait for a new message from the WebSocket.
            msg = await ws.recv()
            data = json.loads(msg)

            # Update the local order book and extract features.
            order_book.update(data)
            row = extractor.update(order_book.bids, order_book.asks)

            # A row is only returned after the initial data buffer is full.
            if row:
                try:
                    # Create the feature array in the exact order the model was trained on.
                    features = np.array([[row[feature] for feature in FEATURE_ORDER]])

                    # Make a prediction and get the probability.
                    prediction = xgb_model.predict(features)[0]
                    proba = xgb_model.predict_proba(features)[0]

                    confidence = proba[prediction]
                    label_text = LABEL_MAP.get(prediction, "N/A")

                    price = row['mid_price']
                    wmp = row['weighted_mid_price']
                    voi = row['voi']

                    # Print the output on a single, updating line.
                    print(
                        f"\rPrediction: {label_text} (Conf: {confidence:.2%}) | Mid: {price:.2f} | WMP: {wmp:.2f} | VOI: {voi:.2f}  ",
                        end="")

                except Exception as e:
                    print(f"Prediction Error: {e}")


# --- 3. Run the Application ---
if __name__ == "__main__":
    try:
        # Start the asynchronous event loop.
        asyncio.run(predict_live())
    except KeyboardInterrupt:
        # Allow the user to stop the script cleanly with Ctrl+C.
        print("\nPrediction stopped by user.")
