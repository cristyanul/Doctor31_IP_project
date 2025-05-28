import pandas as pd
def validate_row(bmi, age, height, weight):
    anomaly = False
    warning = False

    # BMI
    if bmi is None or pd.isnull(bmi):
        anomaly = True
    elif bmi <= 60:
        anomaly = True
    # valid dacă > 60

    # AGE
    if age is None or pd.isnull(age):
        anomaly = True
    elif age < 0 or age > 120:
        anomaly = True
    elif 0 <= age < 18 or (100 < age <= 120):
        warning = True
    # valid dacă 18 <= age <= 100

    # HEIGHT
    if height is None or pd.isnull(height):
        anomaly = True
    elif height < 0 or height > 220:
        anomaly = True
    elif 0 <= height < 150:
        warning = True
    # valid dacă 150 <= height <= 220

    # WEIGHT
    if weight is None or pd.isnull(weight):
        anomaly = True
    elif weight < 20 or weight > 300:
        anomaly = True
    # valid dacă 20 <= weight <= 300

    if anomaly:
        return "Anomaly", "red"
    if warning:
        return "Warning", "orange"
    return "Valid", "green"
def run_validations(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Applies validation and anomaly logic to the standardized dataframe.
    Assumes column names: 'age', 'weight', 'height', 'bmi', 'date'.
    Returns:
        - df: enriched with validation and status columns
        - summary: list of dicts with counts by status for frontend chart
    """
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    # Coerce types safely
    for col in ['age', 'weight', 'height', 'bmi']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df = df.sort_values('date')

    # Duplication check within 1 hour
    df['dup_within_1h'] = False
    valid_mask = df[['age', 'weight', 'height']].notnull().all(axis=1)
    valid_subset = df[valid_mask].copy()

    dup_check = (
        valid_subset
        .sort_values('date')
        .groupby(['age', 'weight', 'height'])['date']
        .diff()
        .dt.total_seconds()
        .lt(3600)
        .fillna(False)
    )

    df.loc[valid_subset.index, 'dup_within_1h'] = dup_check

    df[['status', 'color']] = df.apply(
        lambda row: pd.Series(validate_row(row.get('bmi'), row.get('age'), row.get('height'), row.get('weight'))),
        axis=1
    )

    # Prepare summary for frontend
    summary = (
        df['status']
        .value_counts()
        .reindex(['Valid', 'Warning', 'Anomaly'], fill_value=0)
        .reset_index()
        .rename(columns={'index': 'status', 'status': 'count'})
        .to_dict(orient='records')
    )

    return df, summary