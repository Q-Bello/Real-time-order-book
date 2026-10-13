"""Project-wide settings. Paths are relative to the directory the commands are run from."""

from pathlib import Path

SYMBOL = "btcusdt"
TICK_SIZE = 0.01  # BTCUSDT price increment

# Binance combined stream: 100ms book diffs plus aggregated trades on one connection.
WS_URL = "wss://stream.binance.com:9443/stream?streams={s}@depth@100ms/{s}@aggTrade"
SNAPSHOT_URL = "https://api.binance.com/api/v3/depth?symbol={S}&limit=1000"

# Prediction horizons in seconds.
HORIZONS_S = (1, 2, 5, 10, 30)

# A gap in the data longer than this starts a new segment; labels never cross segments.
SEGMENT_GAP_MS = 2_000

RAW_DIR = Path("data/raw")
DATASET_PATH = Path("data/dataset.parquet")
OOF_DIR = Path("data/oof")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

# Fixed in advance (not chosen from results) for the headline charts.
PRIMARY_HORIZON_S = 5
PRIMARY_THRESHOLD = 0.2
