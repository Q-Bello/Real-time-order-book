# The main application file.
# It runs a multi-threaded Dash web server to display a live dashboard.
# One thread handles the live WebSocket data and model predictions, while the main
# thread runs the web server, updating the UI every second with the latest data.

import asyncio
import os
import sys
import threading
from collections import deque

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

# --- 1. Setup and Initialization ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
from utils.order_book_cache import OrderBook
from utils.binance_feed import synced_order_book
from microstructure.data_labeller import FeatureExtractor
from model.predictor import Predictor

try:
    predictor = Predictor()
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Could not load the trained model. {e}")
    print("Please run `python -m data_stream.generate_dataset` and `python -m model.train_model` first.")
    exit(1)

# Map prediction labels to display colours.
LABEL_COLOURS = {"STABLE": "secondary", "DOWN": "danger", "UP": "success"}


# This class holds the shared state between the WebSocket thread and the Dash app.
class AppState:
    def __init__(self):
        self.lock = threading.Lock()  # A lock to prevent race conditions.
        self.order_book = OrderBook(depth=50)
        self.feature_extractor = FeatureExtractor()
        self.timestamps = deque(maxlen=100)
        self.mid_prices = deque(maxlen=100)
        self.wmp_prices = deque(maxlen=100)
        self.latest_metrics = {}
        self.latest_prediction = {"label_text": "N/A", "color": "secondary", "confidence": "0%"}


app_state = AppState()


# --- 2. Background Data Collector ---
# Runs in a separate thread so it does not block the web server.
def websocket_runner():
    async def data_collector():
        print("Syncing order book and streaming data...")
        async for event_time, resynced in synced_order_book(app_state.order_book):
            # Use a lock to safely update the shared state.
            with app_state.lock:
                if resynced:
                    app_state.feature_extractor.reset()
                book = app_state.order_book
                row = app_state.feature_extractor.compute_features(book.bids, book.asks, timestamp=event_time)
                if row is None:
                    continue
                app_state.latest_metrics = {
                    "Spread": f"{row['spread']:.4f}",
                    "OFI": f"{row['ofi']:.2f}",
                    "VOI": f"{row['voi']:.2f}",
                }
                app_state.timestamps.append(pd.Timestamp(event_time, unit="ms"))
                app_state.mid_prices.append(row["mid_price"])
                app_state.wmp_prices.append(row["weighted_mid_price"])
                try:
                    label, confidence = predictor.predict(row)
                    app_state.latest_prediction = {
                        "label_text": label,
                        "color": LABEL_COLOURS.get(label, "secondary"),
                        "confidence": f"{confidence:.2%}",
                    }
                except Exception as e:
                    print(f"Prediction error: {e}")

    asyncio.run(data_collector())


# Start the background thread.
ws_thread = threading.Thread(target=websocket_runner, daemon=True)
ws_thread.start()

# --- 3. Dash Application Layout ---
# Use a Bootstrap theme for a professional look.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Define the layout using Bootstrap components for structure.
app.layout = dbc.Container(fluid=True, className="dbc", children=[
    dbc.Row(dbc.Col(html.H1("Real-Time Mid-Price Prediction Dashboard", className="text-center text-primary p-4"))),
    dbc.Row([
        # Left column for charts.
        dbc.Col(md=8, children=[
            dbc.Card(className="mb-4",
                     children=[dbc.CardBody(dcc.Graph(id='order-book-graph', style={'height': '40vh'}))]),
            dbc.Card(children=[dbc.CardBody(dcc.Graph(id='price-chart-graph', style={'height': '40vh'}))]),
        ]),
        # Right column for prediction and metrics.
        dbc.Col(md=4, children=[
            dbc.Card(className="mb-4", children=[dbc.CardHeader("Live Prediction"),
                                                 dbc.CardBody(id='prediction-display', className="text-center")]),
            dbc.Card(children=[
                dbc.CardHeader("Live Metrics"),
                dbc.CardBody(dbc.Row([
                    dbc.Col(id='spread-display'), dbc.Col(id='ofi-display'), dbc.Col(id='voi-display'),
                ]))
            ]),
        ]),
    ]),
    # Tooltips for the metrics cards.
    dbc.Tooltip("Bid-Ask Spread: The difference between the best ask and best bid.", target="spread-card"),
    dbc.Tooltip("Order Flow Imbalance (OFI): Aggressive buying/selling at the best bid/ask.", target="ofi-card"),
    dbc.Tooltip("Volume Order Imbalance (VOI): Total change in buy vs. sell volume across the book.",
                target="voi-card"),
    # This interval component triggers the update callback every second.
    dcc.Interval(id='interval-component', interval=1000)
])


# Helper function to create empty figures with a message.
def create_empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font_size=16)
    fig.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    return fig


# --- 4. Dash Callback for Live Updates ---
# This function is the engine of the dashboard. It's called every second.
@app.callback(
    Output('order-book-graph', 'figure'),
    Output('price-chart-graph', 'figure'),
    Output('prediction-display', 'children'),
    Output('spread-display', 'children'),
    Output('ofi-display', 'children'),
    Output('voi-display', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    # Safely get the latest data from the shared state.
    with app_state.lock:
        bids, asks = app_state.order_book.get_top_levels()
        timestamps, mid_prices, wmp_prices = list(app_state.timestamps), list(app_state.mid_prices), list(
            app_state.wmp_prices)
        prediction, metrics = app_state.latest_prediction, app_state.latest_metrics

    # Create the order book depth chart.
    order_book_fig = create_empty_figure("Waiting for order book data...")
    if bids and asks:
        bid_prices, bid_qtys = [b[0] for b in bids], [b[1] for b in bids]
        ask_prices, ask_qtys = [a[0] for a in asks], [a[1] for a in asks]
        order_book_fig = go.Figure(data=[
            go.Scatter(x=bid_prices, y=np.cumsum(bid_qtys), name='Bids', fill='tozeroy', mode='lines',
                       line={'color': 'green'}),
            go.Scatter(x=ask_prices, y=np.cumsum(ask_qtys), name='Asks', fill='tozeroy', mode='lines',
                       line={'color': 'red'})
        ])
    order_book_fig.update_layout(title_text="Live Order Book Depth", template='plotly_dark', uirevision='constant_ob',
                                 margin=dict(l=40, r=20, t=40, b=30))

    # Create the live price feed chart.
    price_chart_fig = create_empty_figure("Waiting for price data...")
    if timestamps and mid_prices:
        price_chart_fig = go.Figure(data=[
            go.Scatter(x=timestamps, y=mid_prices, mode='lines', name='Mid-Price', line={'color': '#00BFFF'}),
            go.Scatter(x=timestamps, y=wmp_prices, mode='lines', name='WMP', line={'color': '#FFD700'})
        ])
    price_chart_fig.update_layout(title_text="Live Price Feed", template='plotly_dark', uirevision='constant_price',
                                  margin=dict(l=40, r=20, t=40, b=30))

    # Create the prediction display component.
    prediction_div = dbc.Alert(
        [html.H3(prediction['label_text'], className="alert-heading"),
         html.P(f"Confidence: {prediction['confidence']}", className="mb-0")],
        color=prediction['color'], duration=4000, is_open=True,
    )

    # Create the metric card components.
    def create_metric_card(title, value, card_id):
        return dbc.Card(id=card_id,
                        children=[dbc.CardHeader(title), dbc.CardBody(html.H4(value, className="text-center"))],
                        className="text-center")

    spread_card = create_metric_card("Spread", metrics.get("Spread", "N/A"), "spread-card")
    ofi_card = create_metric_card("OFI", metrics.get("OFI", "N/A"), "ofi-card")
    voi_card = create_metric_card("VOI", metrics.get("VOI", "N/A"), "voi-card")

    # Return all the updated components to the layout.
    return order_book_fig, price_chart_fig, prediction_div, spread_card, ofi_card, voi_card


# --- 5. Start the Server ---
if __name__ == "__main__":
    print("Starting Dash server...")
    print("View your dashboard at http://127.0.0.1:8050/")
    app.run(debug=False)
