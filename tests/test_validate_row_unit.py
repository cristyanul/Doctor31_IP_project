import math
import pandas as pd
import pytest
from src.validation import validate_row


# ---------- anomaly cases ---------------------------------------------------
@pytest.mark.parametrize(
    "bmi, age, height, weight",
    [
        (None, 30, 170, 70),
        (float("nan"), 30, 170, 70),
        (60, 30, 170, 70),          # BMI boundary 60
        (61, None, 170, 70),
        (61, float("nan"), 170, 70),
        (61, -1, 170, 70),
        (61, 121, 170, 70),
        (61, 30, None, 70),
        (61, 30, float("nan"), 70),
        (61, 30, -1, 70),
        (61, 30, 221, 70),
        (61, 30, 170, None),
        (61, 30, 170, float("nan")),
        (61, 30, 170, 19.9),
        (61, 30, 170, 301),
    ],
)
def test_anomaly_cases(bmi, age, height, weight):
    assert validate_row(bmi, age, height, weight) == ("Anomaly", "red")


# ---------- warning cases ---------------------------------------------------
@pytest.mark.parametrize(
    "age, height, field",
    [
        (17.999, 170, "age"),   # age < 18
        (101,     170, "age"),   # 100 < age ≤ 120
        (30,    149.999, "height"),
    ],
)
def test_warning_cases(age, height, field):
    assert validate_row(61, age, height, 70) == ("Warning", "orange")


# ---------- precedence ------------------------------------------------------
def test_anomaly_overrides_warning():
    # age would yield Warning but bmi None is Anomaly
    assert validate_row(None, 10, 160, 70) == ("Anomaly", "red")


# ---------- valid paths -----------------------------------------------------
@pytest.mark.parametrize(
    "bmi, age, height, weight",
    [
        (61,    18, 150, 20),
        (1000,  99, 200, 300),
        (61.1, 100, 220, 299.9),
    ],
)
def test_valid_cases(bmi, age, height, weight):
    assert validate_row(bmi, age, height, weight) == ("Valid", "green")
