import copy
import pandas as pd
from pandas.testing import assert_frame_equal
from src.validation import run_validations


# ---------------------------------------------------------------------------#
def _df(records):
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------#
def test_status_color_and_summary():
    records = [
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-01T10:00:00", exp="Valid"),
        dict(bmi=61, age=17, height=170, weight=70, date="2024-01-01T11:00:00", exp="Warning"),
        dict(bmi=50, age=25, height=170, weight=70, date="2024-01-01T12:00:00", exp="Anomaly"),
        dict(bmi=61, age=25, height=149, weight=70, date="2024-01-01T13:00:00", exp="Warning"),
        dict(bmi=61, age=25, height=170, weight=10,  date="2024-01-01T14:00:00", exp="Anomaly"),
    ]

    out, summary = run_validations(_df(records))

    # 1️⃣  status column in the same order as the original list
    assert list(out["status"]) == [r["exp"] for r in records]

    # 2️⃣  colour matches status
    cmap = {"Valid": "green", "Warning": "orange", "Anomaly": "red"}
    assert list(out["color"]) == [cmap[r["exp"]] for r in records]

    # 3️⃣  summary reflects the dataframe counts – cope with any column names
    counts_from_df = out["status"].value_counts().to_dict()

    counts_from_summary = {}
    for row in summary:
        # pick the first *string* value as label, the numeric one as count
        label = next(v for v in row.values() if isinstance(v, str))
        count = next(v for v in row.values() if isinstance(v, (int, float)))
        counts_from_summary[label] = count

    assert counts_from_summary == counts_from_df


# ---------------------------------------------------------------------------#
def test_duplicate_detection():
    recs = [
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-01T10:00:00"),
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-01T10:59:59"),  # 3599 s → dup
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-01T11:00:00"),  #   61 s → dup of previous
        dict(bmi=61, age=25, height=171, weight=70, date="2024-01-01T10:10:00"),  # diff height
        dict(bmi=61, age=None, height=170, weight=70, date="2024-01-01T10:20:00"),# null age → ignored
    ]

    out, _ = run_validations(_df(recs))

    # first record must never be considered a duplicate
    assert out.iloc[0]["dup_within_1h"] is False
    # there should be at least one True flag in the rest
    assert out["dup_within_1h"].iloc[1:].any()


# ---------------------------------------------------------------------------#
def test_sorting_and_input_immutability():
    recs = [
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-02T00:00:00"),
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-01T00:00:00"),
    ]
    original = _df(recs)
    clone    = original.copy(deep=True)

    out, _   = run_validations(original)

    # dates are ascending after the call
    assert list(out["date"]) == sorted(out["date"])

    # caller’s dataframe is untouched
    assert_frame_equal(original, clone)
