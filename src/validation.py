import pandas as pd

def bmi_calculated(weight, height):
    if (pd.isnull(weight) or pd.isnull(height) or
        height <= 0 or weight < 0):
        return None
    return weight / ((height / 100) ** 2)

def validate_row(bmi, age, height, weight):
    # Anomaly conditions (critical safety boundaries)
    anomaly_conditions = [
        pd.isnull(bmi) or bmi < 12 or bmi > 60,
        pd.isnull(age) or age < 0 or age > 120,
        pd.isnull(height) or height < 120,
        pd.isnull(weight) or weight < 20 or weight > 300
    ]

    if any(anomaly_conditions):
        return "Anomaly", "red"

    # Warning conditions (requires attention)
    warning_conditions = [
        age < 18 or age >= 100,  # Minors and elderly
        height < 150
    ]

    if any(warning_conditions):
        return "Warning", "orange"

    return "Valid", "green"
