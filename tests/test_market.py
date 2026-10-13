from lobpred.market import MarketState
from lobpred.simulate import simulate_records

SNAP = {"lastUpdateId": 10, "bids": [["99", "1"]], "asks": [["101", "1"]]}


def depth(U, u, E, b=(), a=()):
    return {"k": "depth", "d": {"U": U, "u": u, "E": E, "b": list(b), "a": list(a)}}


def test_buffered_events_are_applied_after_snapshot():
    state = MarketState()
    assert state.on_record(depth(5, 12, 1000, b=[["100", "2"]])) is None  # before snapshot: buffered
    update = state.on_record({"k": "snap", "d": SNAP})
    assert update.kind == "book" and update.resynced and update.ts == 1000
    assert state.book.best_bid() == (100.0, 2.0)


def test_gap_unsyncs_until_next_snapshot():
    state = MarketState()
    state.on_record({"k": "snap", "d": SNAP})
    assert state.on_record(depth(11, 11, 1100)).kind == "book"
    assert state.on_record(depth(20, 21, 1200)) is None  # missed 12..19
    assert not state.book.synced
    update = state.on_record({"k": "snap", "d": {**SNAP, "lastUpdateId": 20}})
    assert update is not None and update.resynced


def test_simulated_feed_replays_with_one_resync_after_dropped_message():
    state = MarketState()
    resyncs = sum(1 for rec in simulate_records(3, gap_at_minute=1.5)
                  if (u := state.on_record(rec)) is not None and u.resynced)
    assert resyncs == 2  # the initial sync plus the one after the dropped update
