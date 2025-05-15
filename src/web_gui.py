from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import pandas as pd
import os
from openpyxl import Workbook
from datetime import datetime

# Paths update
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')

os.makedirs(static_dir, exist_ok=True)

# Flask app setup
app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)
app.debug = True

@app.route('/help')
def test():
    return "Hello! Testing."

data = None
column_mappings = None
REQUIRED_COLUMNS = ['Age', 'Weight', 'Height'] # TODO: Add more columns to his list

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle CSV file upload, parse it, return column info and preview.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            global data
            data = pd.read_csv(file)
            return jsonify({
                'message': 'File uploaded successfully',
                'columns': data.columns.tolist(),
                'preview': data.head().to_dict('records')
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/map-columns', methods=['POST'])
def map_columns():
    global column_mappings
    mappings = request.json
    
    if not all(col in mappings.values() for col in REQUIRED_COLUMNS):
        return jsonify({'error': 'All required columns must be mapped'}), 400
    
    column_mappings = mappings
    return jsonify({'message': 'Column mapping successful'})

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """
    Rename columns, perform validations and model predictions, then generate report.
    """
    if data is None or column_mappings is None:
        return jsonify({'error': 'No data or column mappings available'}), 400
    
    try:
        mapped_data = data.rename(columns={v: k for k, v in column_mappings.items()})

        # Placeholders
        # TODO: developer logic for validation and developer the ML model
        validation_results = []
        model_results = []

        # Generate report
        report_path = generate_report(mapped_data, validation_results, model_results)
        
        return jsonify({
            'message': 'Analysis complete',
            'validation_results': validation_results,
            'report_path': report_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def generate_report(data, validation_results, model_results):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_path = f'reports/analysis_report_{timestamp}.xlsx'
    
    # Checks dir
    os.makedirs('reports', exist_ok=True)
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Analysis Report"
    
    # Add headers
    headers = ['Row', 'Age', 'Weight']
    ws.append(headers)
    
    # TODO: Develop logic for report generation

    wb.save(excel_path)
    return excel_path

@app.route('/download-report/<timestamp>')
def download_report(timestamp):
    report_path = f'reports/analysis_report_{timestamp}.xlsx'
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return jsonify({'error': 'Report not found'}), 404

def create_app():
    if not os.path.exists('templates'):
        os.makedirs('templates')
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=4000)