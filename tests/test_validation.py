import pytest
from src.validation import validate_row

def test_bmi_anomaly():
    # BMI below lower limit should return Anomaly (red)
    assert validate_row(10, 30, 170, 70) == ("Anomaly", "red")
    # BMI above upper limit should return Anomaly (red)
    assert validate_row(70, 30, 170, 70) == ("Anomaly", "red")
    # BMI within valid range should return Valid (green)
    assert validate_row(25, 30, 170, 70) == ("Valid", "green")

def test_age_anomaly_warning():
    # Age below 18 should return Warning (orange)
    assert validate_row(25, 10, 170, 70) == ("Warning", "orange")
    # Age above 120 should return Anomaly (red)
    assert validate_row(25, 130, 170, 70) == ("Anomaly", "red")
    # Age between 100 and 120 should return Warning (orange)
    assert validate_row(25, 110, 170, 70) == ("Warning", "orange")
    # Age within valid adult range should return Valid (green)
    assert validate_row(25, 35, 170, 70) == ("Valid", "green")

def test_height_anomaly_warning():
    # Height below 0 should return Anomaly (red)
    assert validate_row(25, 30, -5, 70) == ("Anomaly", "red")
    # Height below 150 should return Warning (orange)
    assert validate_row(25, 30, 145, 70) == ("Warning", "orange")
    # Height above 220 should return Anomaly (red)
    assert validate_row(25, 30, 225, 70) == ("Anomaly", "red")
    # Height within valid range should return Valid (green)
    assert validate_row(25, 30, 170, 70) == ("Valid", "green")

def test_weight_anomaly():
    # Weight below 20 should return Anomaly (red)
    assert validate_row(25, 30, 170, 15) == ("Anomaly", "red")
    # Weight above 300 should return Anomaly (red)
    assert validate_row(25, 30, 170, 350) == ("Anomaly", "red")
    # Weight within valid range should return Valid (green)
    assert validate_row(25, 30, 170, 70) == ("Valid", "green")
