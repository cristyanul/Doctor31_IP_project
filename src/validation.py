import pandas as pd

def bmi_calculated(weight, height):
    if pd.isnull(weight) or pd.isnull(height) or height == 0:
        return None
    return weight / ((height / 100) ** 2)

def validate_row(bmi, age, height, weight):
    anomaly = False
    warning = False

    # BMI: 
    if bmi is None or pd.isnull(bmi):
        anomaly = True
    elif bmi < 12 or bmi > 60:
        anomaly = True

    # AGE: 
    if age is None or pd.isnull(age):
        anomaly = True
    elif age < 0:
        anomaly = True
    elif age < 18:
        warning = True
    elif age > 100 and age <= 120:
        warning = True
    elif age > 120:
        anomaly = True

    # HEIGHT:
    if height is None or pd.isnull(height):
        anomaly = True
    elif height < 0:
        anomaly = True
    elif 0 <= height < 150:
        warning = True
    elif height > 220:
        anomaly = True

    # WEIGHT: 
    if weight is None or pd.isnull(weight):
        anomaly = True
    elif weight < 20 or weight > 300:
        anomaly = True

    if anomaly:
        return "Anomaly", "red"
    if warning:
        return "Warning", "orange"
    return "Valid", "green"