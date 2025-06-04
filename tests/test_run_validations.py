# tests/test_run_validations.py
import copy
import pandas as pd
from pandas.testing import assert_frame_equal
from src.validation import run_validations


# ---------------------------------------------------------------------------#
STATUSES = ["Valid", "Warning", "Anomaly"]


def _df(records):
    return pd.DataFrame(records)


def _summary_to_counts(summary):
    """
    Robustly convert the summary list `run_validations` returns into
    a dict {status: count}, regardless of how pandas shaped the rows.

    Each element in `summary` is a dict that *always* contains the count
    (integer) and *may or may not* contain the label (string).  When the
    label is missing we assume the rows are in the canonical order
    ['Valid', 'Warning', 'Anomaly'] – which is how the function builds
    the DataFrame before `reset_index()`.
    """
    counts = {}
    for idx, row in enumerate(summary):
        num = next(v for v in row.values() if isinstance(v, (int, float)))
        str_vals = [v for v in row.values() if isinstance(v, str)]
        if str_vals:
            counts[str_vals[0]] = num
        else:
            # fallback: use canonical position
            if idx < len(STATUSES):
                counts[STATUSES[idx]] = num
    return counts


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

    # 1️⃣  status column order is preserved
    assert list(out["status"]) == [r["exp"] for r in records]

    # 2️⃣  colour column is coherent
    cmap = {"Valid": "green", "Warning": "orange", "Anomaly": "red"}
    assert list(out["color"]) == [cmap[r["exp"]] for r in records]

    # 3️⃣  summary counts match dataframe counts in any representation
    expected_counts = out["status"].value_counts().to_dict()
    summary_counts  = _summary_to_counts(summary)
    assert summary_counts == expected_counts


# ---------------------------------------------------------------------------#
def test_duplicate_detection():
    recs = [
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-01T10:00:00"),
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-01T10:59:59"),  # <3600 s
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-01T11:00:00"),  # <3600 s
        dict(bmi=61, age=25, height=171, weight=70, date="2024-01-01T10:10:00"),  # diff height
        dict(bmi=61, age=None, height=170, weight=70, date="2024-01-01T10:20:00"),# NaN age -> ignored
    ]
    out, _ = run_validations(_df(recs))
    flags = [bool(x) for x in out["dup_within_1h"]]

    assert flags[0] is False          # first row never dup
    assert any(flags[1:])             # at least one duplicate flagged


# ---------------------------------------------------------------------------#
def test_sorting_and_input_immutability():
    recs = [
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-02T00:00:00"),
        dict(bmi=61, age=25, height=170, weight=70, date="2024-01-01T00:00:00"),
    ]
    original = _df(recs)
    snapshot = original.copy(deep=True)

    out, _ = run_validations(original)

    assert list(out["date"]) == sorted(out["date"])  # sorted ascending
    assert_frame_equal(original, snapshot)           # input not mutated
