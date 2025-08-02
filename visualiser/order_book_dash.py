import asyncio
import json
import threading
import websockets
import dash
from dash import dcc, html
import plotly.graph_objs as go
from utils.order_book_cache import OrderBook

order_book = OrderBook(depth=20)

# Async WebSocket runner in background
def start_websocket():
    async def stream():
        uri = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
        async with websockets.connect(uri) as ws:
            while True:
                data = json.loads(await ws.recv())
                order_book.update(data)
    asyncio.run(stream())

t = threading.Thread(target=start_websocket)
t.daemon = True
t.start()

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Real-Time BTC/USDT Order Book"),
    dcc.Graph(id='order-book'),
    dcc.Interval(id='interval', interval=1000, n_intervals=0)
])

@app.callback(
    dash.dependencies.Output('order-book', 'figure'),
    [dash.dependencies.Input('interval', 'n_intervals')]
)
def update_graph(n):
    bids, asks = order_book.get_top_levels()

    bid_prices = [x[0] for x in bids]
    bid_qtys = [x[1] for x in bids]
    ask_prices = [x[0] for x in asks]
    ask_qtys = [x[1] for x in asks]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=bid_prices, y=bid_qtys, name='Bids', marker_color='green'))
    fig.add_trace(go.Bar(x=ask_prices, y=ask_qtys, name='Asks', marker_color='red'))
    fig.update_layout(title="Live Order Book Depth", xaxis_title="Price", yaxis_title="Quantity", barmode='overlay')
    return fig

if __name__ == "__main__":
    app.run(debug=True)
