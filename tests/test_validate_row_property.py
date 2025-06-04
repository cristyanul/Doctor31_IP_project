from hypothesis import given, strategies as st
from src.validation import validate_row


# ----- helper strategies ----------------------------------------------------
valid_bmi   = st.floats(min_value=60.000_000_1, max_value=1e6, allow_nan=False)

warn_age    = st.one_of(
    st.floats(min_value=0,          max_value=17.9999),
    st.floats(min_value=100.0001,   max_value=120)
)
valid_age   = st.floats(min_value=18, max_value=100)

warn_h      = st.floats(min_value=0,   max_value=149.9999)
valid_h     = st.floats(min_value=150, max_value=220)

valid_w     = st.floats(min_value=20,  max_value=300)


@given(bmi=valid_bmi, age=valid_age, height=valid_h, weight=valid_w)
def test_property_valid(bmi, age, height, weight):
    """Random valid records should be marked Valid."""
    assert validate_row(bmi, age, height, weight)[0] == "Valid"


@given(bmi=valid_bmi, age=warn_age, height=valid_h, weight=valid_w)
def test_property_warning_age(bmi, age, height, weight):
    """Out-of-range age but otherwise OK ⇒ Warning."""
    assert validate_row(bmi, age, height, weight)[0] == "Warning"


@given(bmi=valid_bmi, age=valid_age, height=warn_h, weight=valid_w)
def test_property_warning_height(bmi, age, height, weight):
    """Short stature but otherwise OK ⇒ Warning."""
    assert validate_row(bmi, age, height, weight)[0] == "Warning"
