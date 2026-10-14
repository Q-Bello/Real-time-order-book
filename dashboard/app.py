"""Live dashboard: order book depth, price, the model's current call and its live hit rate.

A background thread runs the same feed, book, feature and model code as the research
pipeline; the Dash app redraws from shared state once a second.

Usage:
    python dashboard/app.py --horizon 5     then open http://127.0.0.1:8050/
"""

from __future__ import annotations

import argparse
import asyncio
import threading
from collections import deque
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

from lobpred.config import MODEL_DIR, PRIMARY_HORIZON_S
from lobpred.features import FeatureEngine
from lobpred.labels import LABEL_NAMES
from lobpred.live import LiveScorer, Predictor
from lobpred.market import MarketState
from lobpred.stream import live_records

COLOURS = {"STABLE": "secondary", "DOWN": "danger", "UP": "success"}


class AppState:
    def __init__(self, predictor: Predictor):
        self.lock = threading.Lock()
        self.predictor = predictor
        self.market = MarketState()
        self.engine = FeatureEngine()
        self.scorer = LiveScorer(predictor.horizon_s)
        self.times: deque = deque(maxlen=600)
        self.mids: deque = deque(maxlen=600)
        self.row: dict | None = None
        self.pred = ("N/A", np.zeros(3))


def feed_thread(state: AppState) -> None:
    async def run():
        async for record in live_records():
            with state.lock:
                update = state.market.on_record(record)
                if update is None:
                    continue
                if update.kind == "trade":
                    state.engine.on_trade(update.data)
                    continue
                if update.resynced:
                    state.engine.reset()
                    state.scorer.reset()
                row = state.engine.on_book(state.market.book, update.ts)
                if row is None:
                    continue
                state.scorer.update(update.ts, row["mid"])
                pred, proba = state.predictor.predict(row)
                state.scorer.add(update.ts, row["mid"], row["spread"], pred)
                state.row, state.pred = row, (LABEL_NAMES[pred], proba)
                state.times.append(pd.Timestamp(update.ts, unit="ms"))
                state.mids.append(row["mid"])

    asyncio.run(run())


def build_app(state: AppState) -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    h = state.predictor.horizon_s

    def card(title, body_id):
        return dbc.Card([dbc.CardHeader(title), dbc.CardBody(id=body_id, className="text-center")],
                        className="mb-3")

    app.layout = dbc.Container(fluid=True, children=[
        html.H2(f"BTC/USDT order book: live {h}s direction model", className="p-3"),
        dbc.Row([
            dbc.Col(md=8, children=[
                dbc.Card(dbc.CardBody(dcc.Graph(id="depth", style={"height": "38vh"})), className="mb-3"),
                dbc.Card(dbc.CardBody(dcc.Graph(id="price", style={"height": "38vh"}))),
            ]),
            dbc.Col(md=4, children=[
                card("Model call", "prediction"),
                card("Live hit rate", "hit-rate"),
                card("Signals", "metrics"),
            ]),
        ]),
        dcc.Interval(id="tick", interval=1000),
    ])

    @app.callback(
        Output("depth", "figure"), Output("price", "figure"), Output("prediction", "children"),
        Output("hit-rate", "children"), Output("metrics", "children"), Input("tick", "n_intervals"),
    )
    def refresh(_):
        with state.lock:
            book = state.market.book
            bids = book.top_bids(50) if book.synced and book.bids else []
            asks = book.top_asks(50) if book.synced and book.asks else []
            times, mids = list(state.times), list(state.mids)
            row, (label, proba) = state.row, state.pred
            n, correct = state.scorer.n, state.scorer.correct

        depth = go.Figure()
        if bids and asks:
            depth.add_scatter(x=[p for p, _ in bids], y=np.cumsum([q for _, q in bids]), name="Bids",
                              fill="tozeroy", line={"color": "#1baf7a"})
            depth.add_scatter(x=[p for p, _ in asks], y=np.cumsum([q for _, q in asks]), name="Asks",
                              fill="tozeroy", line={"color": "#e34948"})
        depth.update_layout(title="Cumulative depth (best 50 levels)", template="plotly_dark",
                            uirevision="depth", margin=dict(l=40, r=20, t=40, b=30))

        price = go.Figure(go.Scatter(x=times, y=mids, mode="lines", line={"color": "#3987e5"}, name="Mid"))
        price.update_layout(title="Mid-price (last 60s)", template="plotly_dark", uirevision="price",
                            margin=dict(l=40, r=20, t=40, b=30))

        prediction = dbc.Alert([html.H3(label), html.P(
            f"P(up) {proba[2]:.0%} · P(stable) {proba[0]:.0%} · P(down) {proba[1]:.0%}", className="mb-0")],
            color=COLOURS.get(label, "secondary"))
        hit_text = [
            html.H3(f"{correct / n:.1%}" if n else "warming up"),
            html.P(f"{n:,} predictions scored after {h}s", className="mb-0 text-muted"),
        ]

        metrics = "waiting for data"
        if row:
            metrics = html.Div([html.P(f"{name}: {row[key]:+.3f}", className="mb-1") for name, key in [
                ("Level-1 imbalance", "imb_l1"), ("OFI (10 updates)", "ofi_l1_10"),
                ("Trade flow (1s)", "tfi_1s"), ("Spread (ticks)", "spread_ticks")]])
        return depth, price, prediction, hit_text, metrics

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=PRIMARY_HORIZON_S)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    path = args.model_dir / f"xgb_{args.horizon}s.joblib"
    if not path.exists():
        raise SystemExit(f"No model at {path}. Run `python -m lobpred.evaluate` first.")
    state = AppState(Predictor(path))
    threading.Thread(target=feed_thread, args=(state,), daemon=True).start()
    print("Dashboard at http://127.0.0.1:8050/")
    build_app(state).run(debug=False)


if __name__ == "__main__":
    main()
