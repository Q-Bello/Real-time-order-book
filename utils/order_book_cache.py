from collections import deque

class OrderBook:
    def __init__(self, depth=20):
        self.bids = {}
        self.asks = {}
        self.depth = depth

    def update(self, data):
        for bid in data.get("b", []):  # [price, qty]
            price, qty = float(bid[0]), float(bid[1])
            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for ask in data.get("a", []):  # [price, qty]
            price, qty = float(ask[0]), float(ask[1])
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

    def get_top_levels(self):
        top_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:self.depth]
        top_asks = sorted(self.asks.items(), key=lambda x: x[0])[:self.depth]
        return top_bids, top_asks
