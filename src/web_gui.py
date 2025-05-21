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

def _records_safe(df):
    return df.replace({np.nan: None}).to_dict('records')

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
            global data, column_mappings
            data = pd.read_csv(file)
            colnames = data.columns.tolist()
            column_mappings = None
            return jsonify({
                'message': 'File uploaded successfully',
                'columns': colnames
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/map-columns', methods=['POST'])
def map_columns():
    global column_mappings
    user_map = request.json or {}
    if set(user_map.keys()) != set(DEFAULT_COLUMN_MAPPING.keys()):
        return jsonify({'error': 'You must map all fields.'}), 400
    column_mappings = user_map
    return jsonify({'message': 'Column mappings saved'})

@app.route('/preview', methods=['POST'])
def preview_data_route():
    global data, column_mappings
    if data is None or column_mappings is None:
        return jsonify({'error': 'No data or column mappings available'}), 400
    try:
        mapped_data = data.rename(columns={v: k for k, v in column_mappings.items() if v in data.columns})
        for col in ['age', 'weight', 'height', 'bmi']:
            if col in mapped_data.columns:
                mapped_data[col] = mapped_data[col].replace(r'^\s*$', np.nan, regex=True)
                mapped_data[col] = pd.to_numeric(mapped_data[col], errors='coerce')
        preview = mapped_data.head(200).replace({np.nan: None}).to_dict('records')
        return jsonify({'preview': preview})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/analyze', methods=['POST'])
def analyze_data():
    global data, column_mappings
    if data is None or column_mappings is None:
        return jsonify({'error': 'No data or column mappings available'}), 400

    try:
        mapped_data = data.rename(columns={v: k for k, v in column_mappings.items() if v in data.columns})

        needed_cols = ['age', 'weight', 'height', 'bmi']
        for col in needed_cols:
            if col in mapped_data.columns:
                mapped_data[col] = mapped_data[col].replace(r'^\s*$', np.nan, regex=True)
                mapped_data[col] = pd.to_numeric(mapped_data[col], errors='coerce')

        preview_df = mapped_data.copy()  # ce vad in preview
        X = preview_df[needed_cols].dropna()

        preview_df['isolation_score'] = np.nan
        preview_df['anomaly'] = np.nan
        preview_df['status'] = 'Neanalizat'
        preview_df['color'] = '#e0e0e0'  

        if X.shape[0] >= 2:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = IsolationForest(contamination=0.1, random_state=42)
            preds = clf.fit_predict(X_scaled)
            scores = clf.decision_function(X_scaled)

            preview_df.loc[X.index, 'isolation_score'] = scores
            preview_df.loc[X.index, 'anomaly'] = preds
            preview_df.loc[(preview_df['anomaly'] == 1), 'status'] = 'Valid'
            preview_df.loc[(preview_df['anomaly'] == 1), 'color'] = 'lightgreen'
            preview_df.loc[(preview_df['anomaly'] == -1), 'status'] = 'Anomaly'
            preview_df.loc[(preview_df['anomaly'] == -1), 'color'] = 'red'

        summary = (
            preview_df['status']
            .value_counts()
            .reindex(['Valid', 'Anomaly', 'Neanalizat'], fill_value=0)
            .reset_index()
            .rename(columns={'index': 'status', 'status': 'count'})
            .to_dict(orient='records')
        )

        preview = preview_df.head(200).replace({np.nan: None}).to_dict('records')
        total = preview_df.shape[0]
        anomaly = (preview_df['status'] == 'Anomaly').sum()
        percent_anomaly = round(anomaly * 100 / total, 2) if total else 0

        return jsonify({
            'message': 'Analysis complete',
            'preview': preview,
            'summary': summary,
            'percent_anomaly': percent_anomaly
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def create_app():
    if not os.path.exists('templates'):
        os.makedirs('templates')
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=4000)