from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import run_validations
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
    # Create reverse mapping (original_col -> standard_name)
    reverse_mapping = {v: k for k, v in mappings.items() if v in df.columns}
    
    # Apply mapping
    mapped_df = df.rename(columns=reverse_mapping)
    
    # Clean and convert numeric columns
    numeric_cols = ['age', 'weight', 'height', 'bmi']
    for col in numeric_cols:
        if col in mapped_df.columns:
            # Replace empty strings and whitespace with NaN
            mapped_df[col] = mapped_df[col].astype(str).replace(r'^\s*$', np.nan, regex=True)
            # Convert to numeric, coercing errors to NaN
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
        
        # Initialize analysis columns
        analysis_df['isolation_score'] = np.nan
        analysis_df['anomaly'] = np.nan
        analysis_df['status'] = 'Not analyzed'
        analysis_df['color'] = '#e0e0e0'
        
        # Debug: Check data quality for available columns
        for col in available_cols:
            logger.debug(f"Column '{col}' - Data type: {analysis_df[col].dtype}")
            logger.debug(f"Column '{col}' - Non-null count: {analysis_df[col].count()}/{len(analysis_df)}")
            logger.debug(f"Column '{col}' - Sample values: {analysis_df[col].dropna().head().tolist()}")
            logger.debug(f"Column '{col}' - Value counts: {analysis_df[col].value_counts().head()}")
            logger.debug("---")
        
        # Use only available columns for analysis
        feature_data = analysis_df[available_cols].copy()
        
        # Check for complete cases
        complete_mask = feature_data.notna().all(axis=1)
        complete_data = feature_data[complete_mask]
        
        logger.debug(f"Total rows: {len(analysis_df)}")
        logger.debug(f"Complete rows for analysis: {len(complete_data)}")
        
        # If we have less than 2 complete rows, try with partial data
        if len(complete_data) < 2:
            logger.debug("Not enough complete cases, trying with partial data...")
            # Try with rows that have at least 2 non-null values
            partial_mask = feature_data.count(axis=1) >= 2
            partial_data = feature_data[partial_mask].fillna(feature_data.mean())
            
            logger.debug(f"Rows with at least 2 non-null values: {len(partial_data)}")
            
            if len(partial_data) >= 2:
                complete_data = partial_data
                complete_mask = partial_mask
                logger.debug("Using partial data with mean imputation")
        
        if len(complete_data) >= 2:
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(complete_data)
            
            # Fit isolation forest
            isolation_forest = IsolationForest(
                contamination=0.1, 
                random_state=42,
                n_estimators=100
            )
            
            # Get predictions and scores
            anomaly_predictions = isolation_forest.fit_predict(X_scaled)
            anomaly_scores = isolation_forest.decision_function(X_scaled)
            
            logger.debug(f"Anomaly predictions shape: {anomaly_predictions.shape}")
            logger.debug(f"Anomaly scores shape: {anomaly_scores.shape}")
            logger.debug(f"Complete data indices: {complete_data.index.tolist()[:10]}...")  # First 10 indices
            
            # Map results back to original dataframe using indices
            complete_indices = complete_data.index
            
            # Round isolation scores to 3 decimal places
            analysis_df.loc[complete_indices, 'isolation_score'] = np.round(anomaly_scores, 3)
            analysis_df.loc[complete_indices, 'anomaly'] = anomaly_predictions
            
            # Update status and color based on anomaly detection
            analysis_df.loc[analysis_df['anomaly'] == 1, 'status'] = 'Valid'
            analysis_df.loc[analysis_df['anomaly'] == 1, 'color'] = 'lightgreen'
            analysis_df.loc[analysis_df['anomaly'] == -1, 'status'] = 'Anomaly'
            analysis_df.loc[analysis_df['anomaly'] == -1, 'color'] = 'red'
            
            logger.debug("Status value counts:")
            logger.debug(analysis_df['status'].value_counts())
        else:
            logger.debug("Not enough data for anomaly detection even with partial data")
        
        # Create summary statistics
        status_counts = analysis_df['status'].value_counts()
        status_colors = {
            'Valid': '#d1e7dd',
            'Anomaly': '#f8d7da',
            'Not analyzed': '#e0e0e0'
        }

        summary = []
        for status in ['Valid', 'Anomaly', 'Not analyzed']:
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
        
        # Prepare preview data (first 200 rows)
        preview = _records_safe(analysis_df.head(200))
        
        return jsonify({
            'message': 'Analysis complete',
            'preview': preview,
            'summary': summary,
            'percent_anomaly': percent_anomaly,
            'total_analyzed': len(complete_data) if len(complete_data) >= 2 else 0,
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