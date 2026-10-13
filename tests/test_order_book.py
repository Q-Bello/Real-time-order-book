import pytest

from utils.order_book_cache import OrderBook, OrderBookGapError


def make_book():
    book = OrderBook()
    book.load_snapshot({"lastUpdateId": 100,
                        "bids": [["99.0", "1.0"], ["98.0", "2.0"]],
                        "asks": [["101.0", "1.5"], ["102.0", "3.0"]]})
    return book


def test_snapshot_seeds_book():
    book = make_book()
    assert max(book.bids) == 99.0 and min(book.asks) == 101.0
    assert book.last_update_id == 100


def test_stale_events_are_skipped():
    book = make_book()
    assert book.apply_diff({"U": 90, "u": 100, "b": [["99.0", "0"]], "a": []}) is False
    assert 99.0 in book.bids


def test_overlapping_event_is_applied_and_removes_levels():
    book = make_book()
    assert book.apply_diff({"U": 95, "u": 105, "b": [["99.0", "0"]], "a": [["100.5", "1.0"]]})
    assert max(book.bids) == 98.0 and min(book.asks) == 100.5
    assert book.last_update_id == 105


def test_gap_raises():
    book = make_book()
    with pytest.raises(OrderBookGapError):
        book.apply_diff({"U": 110, "u": 120, "b": [], "a": []})
