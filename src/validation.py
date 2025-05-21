import pandas as pd
import numpy as np

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

    # Core validity checks
    df['bmi_valid']    = df['bmi'].between(12, 60, inclusive='both')
    df['age_valid']    = df['age'].between(0, 120, inclusive='both')
    df['weight_valid'] = df['weight'].between(20, 300, inclusive='both')
    df['height_valid'] = df['height'].between(120, 220, inclusive='both')

    # Categorize BMI
    df['bmi_cat'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 25, 30, 35, np.inf],
        labels=['underweight', 'normal', 'overweight', 'obese', 'extreme_obese']
    )

    # Elderly + obese flag
    df['elderly_obese'] = (df['age'] > 85) & df['bmi_cat'].isin(['obese', 'extreme_obese'])

    # Final status assignment
    red_flags = ~df[['age_valid', 'weight_valid', 'height_valid', 'bmi_valid']].all(axis=1)
    yellow_flags = df['dup_within_1h'] | df['elderly_obese']

    df['status'] = np.select(
        [red_flags, yellow_flags],
        ['Anomaly', 'Suspicious'],
        default='Valid'
    )

    df['color'] = df['status'].map({
        'Valid': 'lightgreen',
        'Suspicious': 'yellow',
        'Anomaly': 'red'
    })

    # Prepare summary for frontend
    summary = (
        df['status']
        .value_counts()
        .reindex(['Valid', 'Suspicious', 'Anomaly'], fill_value=0)
        .reset_index()
        .rename(columns={'index': 'status', 'status': 'count'})
        .to_dict(orient='records')
    )

    return df, summary