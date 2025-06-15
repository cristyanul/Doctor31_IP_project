from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import socket
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import validate_row, bmi_calculated

# Setup directories
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')
os.makedirs(static_dir, exist_ok=True)
os.makedirs(template_dir, exist_ok=True)

app = Flask(__name__,
            template_folder=template_dir,
            static_folder=static_dir)

DEFAULT_COLUMN_MAPPING = {
    'case_id': 'id_cases',
    'age': 'age_v',
    'sex': 'sex_v',
    'consent': 'agreement',
    'weight': 'greutate',
    'height': 'inaltime',
    'bmi': 'IMC',
    'date': 'data1',
    'completed': 'finalizat',
    'test_flag': 'testing',
    'bmi_index': 'imcINdex'
}

# Global variables
data = None
column_mappings = None
processed_data = None

def _records_safe(df):
    """Convert DataFrame to records with NaN handling and proper formatting."""
    df_copy = df.replace({np.nan: None}).copy()
    
    # Ensure BMI calculated values are properly formatted to 2 decimal places
    if 'bmi_calculated' in df_copy.columns:
        df_copy['bmi_calculated'] = df_copy['bmi_calculated'].apply(
            lambda x: round(float(x), 2) if x is not None and pd.notnull(x) else x
        )
    
    return df_copy.to_dict('records')

def apply_column_mapping_and_clean(df, mappings):
    """Apply column mappings and clean data."""
    reverse_mapping = {v: k for k, v in mappings.items() if v in df.columns}
    mapped_df = df.rename(columns=reverse_mapping)
    mapped_df.columns = mapped_df.columns.str.lower()
    
    # Clean numeric columns
    for col in ['age', 'weight', 'height', 'bmi']:
        if col in mapped_df.columns:
            mapped_df[col] = mapped_df[col].astype(str).replace(r'^\s*$', np.nan, regex=True)
            mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce')
    
    # Clean date column
    if 'date' in mapped_df.columns:
        mapped_df['date'] = pd.to_datetime(mapped_df['date'], errors='coerce')
        mapped_df['date'] = mapped_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate BMI with safe rounding
    if 'weight' in mapped_df.columns and 'height' in mapped_df.columns:
        def safe_bmi_calculation(row):
            result = bmi_calculated(row['weight'], row['height'])
            if result is not None:
                try:
                    bmi_value = float(result)
                    return round(bmi_value, 2) if not np.isnan(bmi_value) else None
                except (ValueError, TypeError):
                    return None
            return None
        
        mapped_df['bmi_calculated'] = mapped_df.apply(safe_bmi_calculation, axis=1)
    else:
        mapped_df['bmi_calculated'] = None

    return mapped_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or file.filename is None:
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            global data, column_mappings, processed_data
            data = pd.read_csv(file.stream)
            colnames = data.columns.tolist()
            column_mappings = None
            processed_data = None
            return jsonify({
                'message': 'File uploaded successfully',
                'columns': colnames
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/map-columns', methods=['POST'])
def map_columns():
    global column_mappings, processed_data
    user_map = request.json or {}
    
    if set(user_map.keys()) != set(DEFAULT_COLUMN_MAPPING.keys()):
        return jsonify({'error': 'You must map all fields.'}), 400
    
    column_mappings = user_map
    processed_data = None
    return jsonify({'message': 'Column mappings saved'})

@app.route('/preview', methods=['POST'])
def preview_data_route():
    global data, column_mappings, processed_data
    
    if data is None or column_mappings is None:
        return jsonify({'error': 'No data or column mappings available'}), 400
    
    try:
        if processed_data is None:
            processed_data = apply_column_mapping_and_clean(data, column_mappings)
        
        preview = _records_safe(processed_data.head(200))
        return jsonify({'preview': preview})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/analyze', methods=['POST'])
def analyze_data():
    global data, column_mappings, processed_data
    
    if data is None or column_mappings is None:
        return jsonify({'error': 'No data or column mappings available'}), 400

    try:
        if processed_data is None:
            processed_data = apply_column_mapping_and_clean(data, column_mappings)
        
        analysis_df = processed_data.copy()
        
        # Validate required columns
        validation_result = _validate_required_columns(analysis_df)
        if validation_result:
            return validation_result
        
        # Apply validation layers
        analysis_df = _apply_layer1_validation(analysis_df)
        analysis_df = _apply_layer2_isolation_forest(analysis_df)
        
        # Generate results
        return _generate_analysis_results(analysis_df)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def _validate_required_columns(analysis_df):
    """Validate that required columns are available for analysis."""
    required_cols = ['age', 'weight', 'height', 'bmi']
    available_cols = [col for col in required_cols if col in analysis_df.columns]
    
    if not available_cols:
        return jsonify({
            'error': f'None of the required columns ({required_cols}) are available after mapping. Available columns: {analysis_df.columns.tolist()}'
        }), 400
    
    return None

def _apply_layer1_validation(analysis_df):
    """Apply Layer 1 - classic validation rules."""
    analysis_df['status'] = None
    analysis_df['color'] = None
    
    for idx, row in analysis_df.iterrows():
        bmi_c = row.get("bmi_calculated")
        age = row.get("age")
        height = row.get("height")
        weight = row.get("weight")
        status, color = validate_row(bmi_c, age, height, weight)
        analysis_df.at[idx, "status"] = status
        analysis_df.at[idx, "color"] = color
    
    return analysis_df

def _apply_layer2_isolation_forest(analysis_df):
    """Apply Layer 2 - Isolation Forest anomaly detection."""
    required_cols = ['age', 'weight', 'height', 'bmi']
    available_cols = [col for col in required_cols if col in analysis_df.columns]
    
    analysis_df['isolation_score'] = np.nan
    analysis_df['anomaly'] = np.nan

    complete_data = analysis_df.loc[:, available_cols].dropna()

    if len(complete_data) < 2:
        return analysis_df
    
    # Train and apply Isolation Forest
    isolation_results = _train_isolation_forest(analysis_df, complete_data, available_cols)
    analysis_df = _apply_isolation_results(analysis_df, complete_data, isolation_results)
    
    return analysis_df

def _train_isolation_forest(analysis_df, complete_data, available_cols):
    """Train Isolation Forest model and return results."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(complete_data)
    
    # Create target labels based on Layer 1 validation
    layer1_labels = [
        1 if analysis_df.loc[idx, 'status'] == 'Valid' else -1 
        for idx in complete_data.index
    ]
    
    # Calculate contamination based on Layer 1 results
    invalid_ratio = sum(1 for label in layer1_labels if label == -1) / len(layer1_labels)
    contamination = max(0.01, min(0.5, invalid_ratio))
    
    isolation_forest = IsolationForest(
        contamination=contamination,
        random_state=123,
        n_estimators=200
    )
    
    isolation_forest.fit(x_scaled)
    
    return {
        'predictions': isolation_forest.predict(x_scaled),
        'scores': isolation_forest.decision_function(x_scaled),
        'contamination': contamination
    }

def _apply_isolation_results(analysis_df, complete_data, isolation_results):
    """Apply Isolation Forest results to the dataframe."""
    analysis_df.loc[complete_data.index, 'isolation_score'] = isolation_results['scores']
    analysis_df.loc[complete_data.index, 'anomaly'] = isolation_results['predictions']
    
    # Update status based on combined Layer 1 + Layer 2 results
    for idx in complete_data.index:
        layer1_status = analysis_df.loc[idx, 'status']
        isolation_anomaly = analysis_df.loc[idx, 'anomaly']
        
        if layer1_status == 'Valid' and isolation_anomaly == -1:
            analysis_df.loc[idx, 'status'] = 'Anomaly'
            analysis_df.loc[idx, 'color'] = 'red'
    
    return analysis_df

def _generate_analysis_results(analysis_df):
    """Generate final analysis results and summary."""
    status_counts = analysis_df['status'].value_counts()
    status_colors = {
        'Valid': '#d1e7dd',
        'Anomaly': '#f8d7da',
        'Warning': '#fff3cd'
    }

    summary = [
        {
            'status': status,
            'count': int(status_counts.get(status, 0)),
            'color': status_colors.get(status, '#e0e0e0')
        }
        for status in ['Valid', 'Anomaly', 'Not analyzed', 'Warning']
    ]

    # Calculate metrics
    total_rows = len(analysis_df)
    anomaly_count = int(status_counts.get('Anomaly', 0))
    percent_anomaly = round(anomaly_count * 100 / total_rows, 2) if total_rows > 0 else 0

    # Reorder columns for display
    analysis_df = _reorder_columns_for_display(analysis_df)
    preview = _records_safe(analysis_df.head(200))
    
    # Count final valid rows after both validation layers
    final_valid_count = len(analysis_df[analysis_df['status'] == 'Valid'])
    
    required_cols = ['age', 'weight', 'height', 'bmi']
    available_cols = [col for col in required_cols if col in analysis_df.columns]
    missing_cols = [col for col in required_cols if col not in analysis_df.columns]
    
    return jsonify({
        'message': 'Analysis complete',
        'preview': preview,
        'summary': summary,
        'percent_anomaly': percent_anomaly,
        'total_analyzed': final_valid_count,
        'total_rows': total_rows,
        'available_columns': available_cols,
        'missing_columns': missing_cols
    })

def _reorder_columns_for_display(analysis_df):
    """Reorder columns to show most important ones first."""
    shown_cols = ['case_id', 'weight', 'height', 'bmi_calculated', 'date']
    rest_cols = [c for c in analysis_df.columns if c not in shown_cols]
    ordered_cols = shown_cols + rest_cols
    
    if set(shown_cols).issubset(set(analysis_df.columns)):
        return analysis_df[ordered_cols]
    return analysis_df

def find_free_port(start_port=5000, max_port=5100):
    """Find an available port starting from start_port."""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found between {start_port} and {max_port}")

def create_app():
    """Create and configure the Flask application."""
    return app

if __name__ == '__main__':
    try:
        port = find_free_port()
        app.run(host='127.0.0.1', port=port, debug=False)
    except RuntimeError as e:
        print(f"Error: {e}")
        exit(1)