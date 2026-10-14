import pytest

from lobpred.book import GapError, OrderBook


@pytest.fixture
def book():
    b = OrderBook()
    b.load_snapshot({"lastUpdateId": 100,
                     "bids": [["99.0", "1.0"], ["98.0", "2.0"], ["97.0", "0"]],
                     "asks": [["101.0", "1.5"], ["102.0", "3.0"]]})
    return b


def test_snapshot_seeds_book_and_drops_empty_levels(book):
    assert book.best_bid() == (99.0, 1.0)
    assert book.best_ask() == (101.0, 1.5)
    assert 97.0 not in book.bids


def test_top_levels_are_best_first(book):
    assert book.top_bids(2) == [(99.0, 1.0), (98.0, 2.0)]
    assert book.top_asks(2) == [(101.0, 1.5), (102.0, 3.0)]


def test_events_covered_by_snapshot_are_skipped(book):
    assert book.apply_diff({"U": 90, "u": 100, "b": [["99.0", "0"]], "a": []}) is False
    assert book.best_bid()[0] == 99.0


def test_straddling_event_is_applied(book):
    assert book.apply_diff({"U": 95, "u": 105, "b": [["99.0", "0"]], "a": [["100.5", "1"]]})
    assert book.best_bid()[0] == 98.0 and book.best_ask()[0] == 100.5
    assert book.last_update_id == 105


def test_gap_raises(book):
    with pytest.raises(GapError):
        book.apply_diff({"U": 110, "u": 120, "b": [], "a": []})
