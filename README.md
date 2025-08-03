# 📈 Real-Time Order Book Visualiser & Market Microstructure Analyser

This project is a Python-based real-time visualisation and modelling tool for analysing order book microstructure from high-frequency market data. It captures, processes, and models Level 2 (depth-of-market) data using WebSockets and machine learning.

---

##  Features

-  **Real-time data ingestion** from Binance WebSocket API (BTC/USDT pair)
-  **Live order book visualisation** using Dash and Plotly
-  **Feature engineering** from order book snapshots
-  **Machine learning models** (Logistic Regression, XGBoost) to classify price movement
-  Modular project structure suitable for data science pipelines
-  Data logging and replay functionality for training/testing

---

## Project Structure

real-time-order-book/
├── data_stream/ # Raw & processed data (CSV)
│ └── lob_features.csv
├── model/ # ML training scripts
│ └── train_model.py
├── visualiser/ # Dash-based live dashboard
│ └── app.py
├── utils/ # Helper scripts
│ └── feature_engineering.py
├── requirements.txt # Project dependencies
└── README.md

yaml
Copy
Edit

---

##  Machine Learning Models

- **Label Generation**: Binary label for upward price movement in the next 5 seconds.
- **Features**: Order book imbalance, spread, depth statistics, rolling volatility.
- **Models Used**:
  - Logistic Regression
  - XGBoost Classifier

Performance metrics such as accuracy and F1-score are printed to console after training.

---

## 🛠 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/Q-Bello/real-time-order-book.git
cd real-time-order-book

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install OpenMP if on macOS (for XGBoost)
brew install libomp

#To Train ML Models
Make sure the dataset exists at:
bash
data_stream/lob_features.csv
Then run:
bash
python model/train_model.py

#To Run Live Dashboard
bash
python visualiser/app.py
This will start a local server at http://127.0.0.1:8050/ showing real-time order book dynamics.

Notes
The current dataset was collected during live trading hours using the Binance WebSocket.

Label imbalance may require resampling or data augmentation in future iterations.

The visualiser supports basic price depth and bid/ask spread rendering.

Author
Quddus Bello
BSc Computer Science @ Newcastle University (2024–2027)