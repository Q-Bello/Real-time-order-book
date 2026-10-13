import numpy as np
import pandas as pd
import pytest

from lobpred.labels import add_labels


def frame(ts, mids, segments=None):
    return pd.DataFrame({"ts": ts, "mid": mids, "spread": 0.02,
                         "segment": segments if segments is not None else [0] * len(ts)})


def test_label_uses_first_update_at_or_after_horizon():
    df = add_labels(frame([0, 400, 1000, 1600], [100.0, 105.0, 101.0, 90.0]), [1])
    # From t=0 the label looks at t=1000 (mid 101), not at 400 or 1600.
    assert df.loc[0, "fwd_ret_1s_bps"] == pytest.approx(100.0)
    assert df.loc[0, "y_1s"] == 2


def test_no_label_when_future_is_missing():
    df = add_labels(frame([0, 500], [100.0, 99.0]), [1])
    assert np.isnan(df.loc[0, "y_1s"]) and np.isnan(df.loc[1, "y_1s"])


def test_labels_do_not_cross_segments():
    df = add_labels(frame([0, 1000, 5000, 6000], [100.0, 100.0, 50.0, 50.0], segments=[0, 0, 1, 1]), [1])
    assert df.loc[1, "y_1s"] != df.loc[1, "y_1s"]  # NaN: next segment is not its future
    assert df.loc[0, "y_1s"] == 0


def test_small_moves_inside_threshold_are_stable():
    df = add_labels(frame([0, 1000], [100.0, 100.005]), [1])
    assert df.loc[0, "y_1s"] == 0


def test_refuses_to_overwrite_columns():
    df = frame([0, 1000], [1.0, 1.0])
    df["y_1s"] = 0
    with pytest.raises(ValueError):
        add_labels(df, [1])
