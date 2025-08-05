from typing import Dict, List, Tuple

# This class manages a local, in-memory copy of the order book.
# It processes updates from the WebSocket stream to keep the book state current.
class OrderBook:
    def __init__(self, depth: int = 20):
        # Bids and asks are stored as dictionaries for efficient O(1) lookups and updates.
        # Key: price (float), Value: quantity (float)
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        # The number of levels to return when requested.
        self.depth: int = depth

    def update(self, data: Dict):
        # Process both bids ('b') and asks ('a') from the incoming message.
        for side, book in [('b', self.bids), ('a', self.asks)]:
            # Iterate through each price level update in the message.
            for price_str, qty_str in data.get(side, []):
                price, qty = float(price_str), float(qty_str)
                # If quantity is zero, the level has been removed from the book.
                if qty == 0:
                    # Use .pop with a default to avoid errors if the price level doesn't exist.
                    book.pop(price, None)
                # Otherwise, add the new level or update the existing one.
                else:
                    book[price] = qty

    def get_top_levels(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        # Sort bids by price in descending order (highest price first).
        top_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:self.depth]
        # Sort asks by price in ascending order (lowest price first).
        top_asks = sorted(self.asks.items(), key=lambda x: x[0])[:self.depth]
        return top_bids, top_asks

