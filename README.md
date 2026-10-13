# Real-Time Market Mid-Price Prediction

## Overview

This project captures real-time cryptocurrency order book data from the Binance WebSocket API, engineers market microstructure features, and trains a machine learning model to forecast the short-term direction of the BTC/USDT mid-price. It includes data streaming, feature engineering, model training with time-series-aware evaluation, live prediction, and a real-time dashboard.

---

## Features

- **Correctly synchronised order book**: seeds a local Level 2 book from Binance's REST snapshot, then applies the diff-depth stream in update-ID order, re-syncing automatically if an update is missed.
- **Microstructure features**:
    - Bid-ask spread
    - Order Flow Imbalance (OFI): net change in size at the best bid and ask
    - Volume Order Imbalance (VOI): net change in bid vs. ask volume across the book
    - Microprice offset: where the size-weighted mid-price sits inside the spread, from -0.5 (at the bid) to +0.5 (at the ask)
- **Labels**: UP, DOWN or STABLE, based on whether the average mid-price over the next 30 book updates (about 3 seconds) moves by more than half the current spread.
- **Model**: XGBoost classifier, evaluated against a majority-class baseline on a chronological, purged train/test split.
- **Live prediction and dashboard**: the same feature code runs on the live stream, with a Dash dashboard showing book depth, price, metrics and the current prediction.

---

## Methodology

Short-horizon order book prediction is easy to get wrong in ways that make results look far better than they are. The pipeline is built to avoid the common traps:

1. **No look-ahead in labels.** Each row's features are computed from the book at time *t*, and its label describes the mid-price over updates *t+1* to *t+30*. A row is only written once that future window has arrived.
2. **Chronological, purged split.** The model trains on the earlier 80% of the data and is tested on the later 20%. Training rows whose label windows overlap the test period are dropped. A random split would mix near-identical moments seconds apart across train and test.
3. **Stationary features only.** The raw price level is excluded, so the model cannot learn the price range of the training period instead of order flow behaviour.
4. **Baseline comparison.** Short-horizon labels are usually imbalanced, so accuracy is always reported next to the accuracy of always predicting the training period's most common class, alongside balanced accuracy and macro F1.

---

## Project Structure

```
.
├── data_stream/
│   ├── binance_stream.py       # Prints the raw diff-depth stream
│   ├── generate_dataset.py     # Collects and labels a training dataset
│   └── lob_features.csv        # Collected dataset (created by generate_dataset)
├── microstructure/
│   ├── data_labeller.py        # Feature extraction and look-ahead-free labelling
│   └── feature_engineering.py  # Individual feature calculations
├── model/
│   ├── train_model.py          # Trains and evaluates the model, writes results.json
│   ├── predictor.py            # Loads the model for live use
│   ├── predict_live.py         # Console live predictions
│   └── results.json            # Latest evaluation results (created by train_model)
├── utils/
│   ├── order_book_cache.py     # Local order book with snapshot and update-ID handling
│   └── binance_feed.py         # Synchronised Binance feed shared by all scripts
├── visualiser/
│   └── order_book_dash.py      # Dash dashboard
├── tests/                      # Unit tests for book sync and labelling
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Q-Bello/Real-time-order-book.git
cd Real-time-order-book
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

Run all commands from the project root.

### 1. Collect a dataset

```bash
python -m data_stream.generate_dataset --rows 20000
```

At Binance's 100ms update rate, 20,000 rows takes roughly 35 to 40 minutes. Rows are saved in batches, so stopping early with Ctrl+C keeps what has been collected. Use `--append` to add to an existing dataset.

### 2. Train and evaluate

```bash
python -m model.train_model
```

Prints the model's accuracy against the majority-class baseline, balanced accuracy, a classification report and confusion matrix, and saves them to `model/results.json`. Add `--balanced` to weight classes equally during training.

### 3. Live predictions

```bash
python -m model.predict_live           # console
python visualiser/order_book_dash.py   # dashboard at http://127.0.0.1:8050/
```

### Tests

```bash
python -m pytest tests
```

---

## Results

See `model/results.json` for the latest run, including dataset size, test-period class balance, label horizon, model vs. baseline accuracy and feature importance.

---

## Author
- Quddus Bello, BSc Computer Science @ Newcastle University (2024–2027)
- LinkedIn: https://www.linkedin.com/in/quddus-bello-73482b317/
