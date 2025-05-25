from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import run_validations, validate_row
from src.log_config import setup_logger
logger = setup_logger()

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')
os.makedirs(static_dir, exist_ok=True)

app = Flask(__name__,
            template_folder=template_dir,
            static_folder=static_dir)
app.debug = True

DEFAULT_COLUMN_MAPPING = {
    'case_id':    'id_cases',
    'age':        'age_v',
    'sex':        'sex_v',
    'consent':    'agreement',
    'weight':     'greutate',
    'height':     'inaltime',
    'bmi':        'IMC',
    'date':       'data1',
    'completed':  'finalizat',
    'test_flag':  'testing',
    'bmi_index':  'imcINdex'
}

data = None
column_mappings = None
processed_data = None  # Store processed data to avoid reprocessing

def _records_safe(df):
    return df.replace({np.nan: None}).to_dict('records')

def apply_column_mapping_and_clean(df, mappings):
    """Apply column mapping and clean numeric columns"""
    reverse_mapping = {v: k for k, v in mappings.items() if v in df.columns}
    mapped_df = df.rename(columns=reverse_mapping)
    numeric_cols = ['age', 'weight', 'height', 'bmi']
    for col in numeric_cols:
        if col in mapped_df.columns:
            mapped_df[col] = mapped_df[col].astype(str).replace(r'^\s*$', np.nan, regex=True)
            mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce')
    return mapped_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and file.filename.endswith('.csv'):
        try:
            global data, column_mappings, processed_data
            data = pd.read_csv(file)
            colnames = data.columns.tolist()
            column_mappings = None
            processed_data = None  # Reset processed data
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
    processed_data = None  # Reset processed data when mappings change
    return jsonify({'message': 'Column mappings saved'})

@app.route('/preview', methods=['POST'])
def preview_data_route():
    global data, column_mappings, processed_data
    if data is None or column_mappings is None:
        return jsonify({'error': 'No data or column mappings available'}), 400
    try:
        # Process data if not already done
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
        # Process data if not already done or reuse existing processed data
        if processed_data is None:
            processed_data = apply_column_mapping_and_clean(data, column_mappings)
        
        # Create a copy for analysis to avoid modifying the original
        analysis_df = processed_data.copy()
        
        # Debug: Check what columns we have
        logger.debug(f"Available columns after mapping: {analysis_df.columns.tolist()}")
        
        # Required columns for anomaly detection
        required_cols = ['age', 'weight', 'height', 'bmi']
        
        # Check which columns are actually available
        available_cols = [col for col in required_cols if col in analysis_df.columns]
        missing_cols = [col for col in required_cols if col not in analysis_df.columns]
        
        logger.debug(f"Available required columns: {available_cols}")
        logger.debug(f"Missing required columns: {missing_cols}")
        
        if not available_cols:
            return jsonify({
                'error': f'None of the required columns ({required_cols}) are available after mapping. Available columns: {analysis_df.columns.tolist()}'
            }), 400
        
        # --- START: Layer 1 (classic validation rules with validate_row) ---
        analysis_df['status'] = None
        analysis_df['color'] = None
        for idx, row in analysis_df.iterrows():
            bmi = row.get("bmi")
            age = row.get("age")
            height = row.get("height")
            weight = row.get("weight")
            status, color = validate_row(bmi, age, height, weight)
            analysis_df.at[idx, "status"] = status
            analysis_df.at[idx, "color"] = color
        logger.debug("Layer 1: Classic validation rules applied.")
        # --- END: Layer 1 ---

        # --- START: Layer 2 (Isolation Forest only on Layer 1 valid rows) ---
        analysis_df['isolation_score'] = np.nan
        analysis_df['anomaly'] = np.nan
        valid_mask = analysis_df["status"] == "Valid"
        valid_data = analysis_df.loc[valid_mask, available_cols].copy()
        if len(valid_data) >= 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(valid_data)
            isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=123,
                n_estimators=200
            )
            anomaly_predictions = isolation_forest.fit_predict(X_scaled)
            anomaly_scores = isolation_forest.decision_function(X_scaled)
            analysis_df.loc[valid_mask, 'isolation_score'] = anomaly_scores
            analysis_df.loc[valid_mask, 'anomaly'] = anomaly_predictions
            # Outliers detected by Isolation Forest are marked as Anomaly/red
            outlier_idx = valid_data.index[anomaly_predictions == -1]
            analysis_df.loc[outlier_idx, "status"] = "Anomaly"
            analysis_df.loc[outlier_idx, "color"] = "red"
            logger.info("Layer 2: Isolation Forest applied. Outliers marked as Anomaly.")
        else:
            logger.info("Layer 2: Not enough valid rows for Isolation Forest anomaly detection.")
        # --- END: Layer 2 ---

        # Create summary statistics
        status_counts = analysis_df['status'].value_counts()
        status_colors = {
            'Valid': '#d1e7dd',
            'Anomaly': '#f8d7da',
            'Warning': '#fff3cd'
        }

        summary = []
        for status in ['Valid', 'Anomaly', 'Not analyzed', 'Warning']:
            count = status_counts.get(status, 0)
            summary.append({
                'status': status,
                'count': int(count),
                'color': status_colors.get(status, '#e0e0e0')  # fallback color
            })

        # Calculate metrics
        total_rows = len(analysis_df)
        anomaly_count = int(status_counts.get('Anomaly', 0))
        percent_anomaly = round(anomaly_count * 100 / total_rows, 2) if total_rows > 0 else 0

        with open("debug_status.txt", "w") as f:
         f.write(str(analysis_df[['status', 'color']].head(10)))
        
        preview = _records_safe(analysis_df.head(200))
        
        return jsonify({
            'message': 'Analysis complete',
            'preview': preview,
            'summary': summary,
            'percent_anomaly': percent_anomaly,
            'total_analyzed': int(valid_mask.sum()),
            'total_rows': total_rows,
            'available_columns': available_cols,
            'missing_columns': missing_cols
        })
        
    except Exception as e:
        import traceback
        logger.debug(f"Error in analyze_data: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

def create_app():
    if not os.path.exists('templates'):
        os.makedirs('templates')
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=4000)

