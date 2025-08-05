# Real-Time Market Mid-Price Prediction

## Overview

This project captures real-time cryptocurrency order book data from the Binance WebSocket API, engineers predictive features, and uses a machine learning model to forecast the short-term direction of the mid-price. The project includes components for data streaming, feature engineering, model training, and live prediction, culminating in a real-time dashboard that visualises the order book and displays live predictions.

This is designed to be a portfolio project showcasing skills in real-time data processing, feature engineering for financial markets, machine learning, and data visualisation.

---

## Features

- **Real-Time Data Streaming**: Connects to Binance's WebSocket to stream live Level 2 order book data for BTC/USDT.
- **Microstructure Feature Engineering**: Calculates key order book features in real-time, such as:
    - Mid-Price
    - Bid-Ask Spread
    - Order Flow Imbalance (OFI)
- **Machine Learning Model**: Trains an XGBoost Classifier to predict whether the mid-price will increase or decrease over the next N ticks.
- **Live Prediction**: Applies the trained model to the live data stream to generate real-time predictions.
- **Data Visualisation**: A Dash-based web dashboard visualises the order book depth chart in real-time.

---

## Project Structure

```
.
├── data_stream/
│   ├── binance_stream.py       # Basic script to view the raw data stream
│   └── generate_dataset.py     # Script to collect and label data for training
├── microstructure/
│   ├── data_labeller.py        # Extracts and labels features from the stream
│   └── feature_engineering.py  # Functions for calculating individual features
├── model/
│   ├── train_model.py          # Trains the ML model on the generated dataset
│   ├── predict_live.py         # Runs live predictions using the trained model
│   └── xgboost_model.pkl       # The trained and saved model artefact
├── utils/
│   └── order_book_cache.py     # Manages the local state of the order book
├── visualiser/
│   └── order_book_dash.py      # The Dash application for visualisation
├── config.py                   # Central configuration for file paths, URLs, etc.
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd real-time-order-book
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run

The project has three main workflows: generating a dataset, training a model, and running live predictions.

### 1. Generate a Training Dataset

This script will connect to the Binance stream, process 500 order book updates, and save the engineered features and labels to a CSV file.

```bash
python data_stream/generate_dataset.py
```
> This will create `data/lob_features.csv`.

### 2. Train the Model

Once you have a dataset, you can train the XGBoost model.

```bash
python model/train_model.py
```
> This will save the trained model to `model/xgboost_model.pkl`.

### 3. Run Live Predictions & Visualisation

This is the final application. It streams live data, makes predictions, and visualises the order book.

```bash
python visualiser/order_book_dash.py
```
> Open your web browser and navigate to `http://127.0.0.1:8050/` to see the live dashboard.

## Author
- Quddus Bello, BSc Computer Science @ Newcastle University (2024–2027)
- LinkedIn: https://www.linkedin.com/in/quddus-bello-73482b317/