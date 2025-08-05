# This script connects to the Binance WebSocket to collect live market data.
# It uses the FeatureExtractor to process this data and saves a labelled dataset (CSV)
# which is then used to train the machine learning model.

import asyncio
import websockets
import json
import sys
import os
import pandas as pd
from utils.order_book_cache import OrderBook
from microstructure.data_labeller import FeatureExtractor


async def collect_data():
    # Initialise the objects that will manage the data.
    order_book = OrderBook(depth=20)
    extractor = FeatureExtractor(window=30)

    # Define how many data points to collect.
    DATASET_TARGET_SIZE = 200
    BUFFER_SIZE = extractor.window + 1

    uri = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"

    print(f"Connecting to WebSocket at {uri}...")
    async with websockets.connect(uri) as ws:
        print("Successfully connected.")
        print(f"Waiting for {BUFFER_SIZE} data points to fill the initial buffer...")

        rows_collected = 0
        messages_received = 0

        # Loop until the target number of rows is collected.
        while rows_collected < DATASET_TARGET_SIZE:
            try:
                # Wait for a new message from the WebSocket.
                msg = await ws.recv()
                data = json.loads(msg)
                messages_received += 1

                # Update the local order book and extract features.
                order_book.update(data)
                current_bids, current_asks = order_book.bids, order_book.asks
                row = extractor.update(current_bids, current_asks)

                # The extractor returns a row only after its buffer is full.
                if row:
                    rows_collected += 1
                    # Print progress on a single line.
                    sys.stdout.write(f"\rCollected row {rows_collected}/{DATASET_TARGET_SIZE}...")
                    sys.stdout.flush()
                else:
                    # Show buffering progress.
                    sys.stdout.write(f"\rBuffering... [Received: {messages_received}, Need: {BUFFER_SIZE}]")
                    sys.stdout.flush()
            except Exception as e:
                print(f"\nError during collection: {e}")
                break

    print(f"\n\nFinished collecting {len(extractor.feature_rows)} data points.")

    # Save the data if any was collected.
    if extractor.feature_rows:
        # Build a robust file path to save the CSV in the correct directory.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = "lob_features.csv"
        output_filename = os.path.join(script_dir, filename)

        extractor.save_to_csv(output_filename)
        print(f"Saved data to {output_filename}")

        # Final check to ensure the dataset is balanced enough for training.
        df = pd.read_csv(output_filename)
        label_counts = df['label'].value_counts()
        print("\nNew dataset class distribution:")
        print(label_counts)
        if len(label_counts) < 2:
            print("\nWARNING: The new dataset still contains only one class.")
            print("The market might be very stable. Try running again later.")
        else:
            print("\nDataset contains multiple classes. Ready for training.")
    else:
        print("No data was collected, file not saved.")


if __name__ == "__main__":
    try:
        asyncio.run(collect_data())
    except KeyboardInterrupt:
        print("\n\nData collection interrupted by user.")
