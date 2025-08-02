import asyncio
import websockets
import json

async def stream_order_book():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"

    async with websockets.connect(uri) as websocket:
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            print(json.dumps(data, indent=2))

if __name__ == "__main__":
    asyncio.run(stream_order_book())
