import pandas as pd
from scipy import stats
import numpy as np
from datetime import timedelta
#csv to be viewed in a nice gui, anomalous data highlighted, based on confidence
#self contained, monolith app
# Load your dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    # Convert timestamp column to datetime if it exists
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

# Function to check age validity
def is_age_valid(age):
    return 0 <= age <= 120

# Function to validate weight based on age
def is_weight_valid(weight, age):
    if age < 18:
        # For younger individuals
        return 30 <= weight <= 90
    else:
        # For adults
        return 40 <= weight <= 150

# Function to validate height based on age
def is_height_valid(height, age):
    if age < 2:
        return 50 <= height <= 100
    elif age < 18:
        return 100 <= height <= 200
    else:
        return 130 <= height <= 220

# Function to calculate BMI
def calculate_bmi(weight, height_cm):
    try:
        # Convert height from cm to meters
        height_m = height_cm / 100
        return weight / (height_m ** 2)
    except ZeroDivisionError:
        return None

# Function to check if BMI is compatible with life
def is_bmi_valid(bmi):
    if bmi is None:
        return False
    return 12 <= bmi <= 60

# Function to check for suspicious elderly obesity cases
def is_suspicious_elderly_obesity(age, bmi):
    if age > 85 and bmi >= 30:  # BMI >= 30 is considered obese
        return True
    return False

# Function to detect duplicates within a short timeframe
def detect_duplicates(data):
    # Create a copy to avoid modifying the original
    df = data.copy()

    # If timestamp column exists, use it for duplicate detection
    if 'timestamp' in df.columns:
        # Sort by age, weight, height, and timestamp
        df = df.sort_values(['age', 'weight', 'height', 'timestamp'])

        # Initialize duplicate column
        df['is_duplicate'] = False

        # Check for duplicates with same age, weight, height within 1 hour
        for i in range(1, len(df)):
            if (df.iloc[i]['age'] == df.iloc[i-1]['age'] and
                df.iloc[i]['weight'] == df.iloc[i-1]['weight'] and
                df.iloc[i]['height'] == df.iloc[i-1]['height']):

                # If timestamps are within one hour, mark as duplicate
                if (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']) < timedelta(hours=1):
                    df.loc[df.index[i], 'is_duplicate'] = True

        return df['is_duplicate']
    else:
        # If no timestamp column, return False for all rows
        return pd.Series(False, index=data.index)

# Main function to validate the dataset
def main():
    data = load_data('your_dataset.csv')

    # Validate age
    data['valid_age'] = data['age'].apply(is_age_valid)

    # Validate weight
    data['valid_weight'] = data.apply(lambda row: is_weight_valid(row['weight'], row['age']), axis=1)

    # Validate height
    data['valid_height'] = data.apply(lambda row: is_height_valid(row['height'], row['age']), axis=1)

    # Calculate BMI
    data['BMI'] = data.apply(lambda row: calculate_bmi(row['weight'], row['height']), axis=1)

    # Check for valid BMI (compatible with life)
    data['valid_bmi'] = data['BMI'].apply(is_bmi_valid)

    # Check for suspicious elderly obesity cases
    data['suspicious_elderly_obesity'] = data.apply(
        lambda row: is_suspicious_elderly_obesity(row['age'], row['BMI']), axis=1
    )

    # Detect outliers using Z-score for weight and height
    data['z_weight'] = stats.zscore(data['weight'])
    data['z_height'] = stats.zscore(data['height'])

    # Detect possible duplicates
    data['is_duplicate'] = detect_duplicates(data)

    # Flag unrealistic entries
    data['unrealistic'] = False
    data.loc[((data['valid_age'] == False) |
              (data['valid_weight'] == False) |
              (data['valid_height'] == False) |
              (data['valid_bmi'] == False) |
              (data['suspicious_elderly_obesity'] == True) |
              (data['is_duplicate'] == True)) |
             ((abs(data['z_weight']) > 3) |
              (abs(data['z_height']) > 3)), 'unrealistic'] = True

    # Save the results
    unrealistic_entries = data[data['unrealistic']]
    unrealistic_entries.to_csv('unrealistic_entries.csv', index=False)

    # Save details about why entries were flagged
    reasons = []
    for _, row in unrealistic_entries.iterrows():
        reason_list = []
        if not row['valid_age']:
            reason_list.append("Invalid age")
        if not row['valid_weight']:
            reason_list.append("Invalid weight")
        if not row['valid_height']:
            reason_list.append("Invalid height")
        if not row['valid_bmi']:
            reason_list.append(f"BMI incompatible with life ({row['BMI']:.1f})")
        if row['suspicious_elderly_obesity']:
            reason_list.append(f"Suspicious elderly obesity (age: {row['age']}, BMI: {row['BMI']:.1f})")
        if row['is_duplicate']:
            reason_list.append("Potential duplicate entry")
        if abs(row['z_weight']) > 3:
            reason_list.append(f"Weight is a statistical outlier (z-score: {row['z_weight']:.2f})")
        if abs(row['z_height']) > 3:
            reason_list.append(f"Height is a statistical outlier (z-score: {row['z_height']:.2f})")

        reasons.append("; ".join(reason_list))

    unrealistic_entries['anomaly_reason'] = reasons
    unrealistic_entries.to_csv('unrealistic_entries_with_reasons.csv', index=False)

    print(f"Identified {len(unrealistic_entries)} unrealistic entries out of {len(data)} total records.")
    print("Unrealistic entries have been saved to unrealistic_entries.csv")
    print("Detailed reasons for anomalies saved to unrealistic_entries_with_reasons.csv")

if __name__ == "__main__":
    main()