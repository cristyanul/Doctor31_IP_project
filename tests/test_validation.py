import pytest
from src.validation import validate_row, bmi_calculated

def test_bmi_anomalies():
    # None or nan
    assert validate_row(None, 25, 170, 70) == ("Anomaly", "red")
    # under 12
    assert validate_row(11.99, 25, 170, 70) == ("Anomaly", "red")
    # over 60
    assert validate_row(60.01, 25, 170, 70) == ("Anomaly", "red")

def test_bmi_valid_boundaries():
    # 12
    assert validate_row(12, 25, 170, 70) == ("Valid", "green")
    # 60
    assert validate_row(60, 25, 170, 70) == ("Valid", "green")
    # between 12 and 60
    assert validate_row(30, 25, 170, 70) == ("Valid", "green")

def test_age_anomaly_warning_valid():
    # Age < 0 anomaly
    assert validate_row(20, -0.01, 170, 70) == ("Anomaly", "red")
    # Age = 0 warning
    assert validate_row(20, 0, 170, 70) == ("Warning", "orange")
    # Age = 17.99 warning
    assert validate_row(20, 17.99, 170, 70) == ("Warning", "orange")
    # Age = 18 valid
    assert validate_row(20, 18, 170, 70) == ("Valid", "green")
    # Age = 100 valid
    assert validate_row(20, 100, 170, 70) == ("Valid", "green")
    # Age = 100.01 warning
    assert validate_row(20, 100.01, 170, 70) == ("Warning", "orange")
    # Age = 120 warning
    assert validate_row(20, 120, 170, 70) == ("Warning", "orange")
    # Age > 120 anomaly
    assert validate_row(20, 120.01, 170, 70) == ("Anomaly", "red")
    # None
    assert validate_row(20, None, 170, 70) == ("Anomaly", "red")

def test_height_anomaly_warning_valid():
    # Height < 0 anomaly
    assert validate_row(20, 25, -1, 70) == ("Anomaly", "red")
    # Height = 0 warning
    assert validate_row(20, 25, 0, 70) == ("Warning", "orange")
    # Height = 149.99 warning
    assert validate_row(20, 25, 149.99, 70) == ("Warning", "orange")
    # Height = 150 valid
    assert validate_row(20, 25, 150, 70) == ("Valid", "green")
    # Height = 220 valid
    assert validate_row(20, 25, 220, 70) == ("Valid", "green")
    # Height > 220 anomaly
    assert validate_row(20, 25, 220.01, 70) == ("Anomaly", "red")
    # None
    assert validate_row(20, 25, None, 70) == ("Anomaly", "red")

def test_weight_anomaly_valid():
    # Weight < 20 anomaly
    assert validate_row(20, 25, 170, 19.99) == ("Anomaly", "red")
    # Weight = 20 valid
    assert validate_row(20, 25, 170, 20) == ("Valid", "green")
    # Weight = 300 valid
    assert validate_row(20, 25, 170, 300) == ("Valid", "green")
    # Weight > 300 anomaly
    assert validate_row(20, 25, 170, 300.01) == ("Anomaly", "red")
    # None
    assert validate_row(20, 25, 170, None) == ("Anomaly", "red")

def test_bmi_calculated_function():
    # Nominal
    assert round(bmi_calculated(70, 170), 2) == 24.22
    # Height zero
    assert bmi_calculated(70, 0) is None
    # Weight None
    assert bmi_calculated(None, 170) is None
    # Height None
    assert bmi_calculated(70, None) is None

def test_mixed_priority():
    # anomaly > warning
    # BMI anomaly + age warning = anomaly
    assert validate_row(11, 10, 170, 70) == ("Anomaly", "red")
    # age anomaly + height warning = anomaly
    assert validate_row(20, -2, 149, 70) == ("Anomaly", "red")
    # warning only
    assert validate_row(20, 10, 149, 70) == ("Warning", "orange")