# Doctor31 Data Validation Tool

A Python-based web application that detects unrealistic or impossible patient entries in self-reported medical datasets using a two-layer validation approach combining rule-based validation with machine learning anomaly detection.

## What It Does

This tool analyzes patient health data (age, weight, height, BMI) and automatically identifies:

- **Anomalies**: Medically impossible values (e.g., BMI < 12 or > 60, age < 0 or > 120)
- **Warnings**: Suspicious but possible values (e.g., minors, elderly patients, very short height)
- **Valid**: Normal, acceptable patient data

## Key Features

### 🔍 **Two-Layer Validation System**
- **Layer 1**: Rule-based validation using medical safety boundaries
- **Layer 2**: Isolation Forest machine learning for statistical anomaly detection

### 🌐 **Web Interface**
- Upload CSV files through a modern, responsive web interface
- Column mapping for different data formats
- Real-time data preview and validation results
- Interactive charts and summary statistics

### 📊 **Validation Rules**
- **BMI**: Must be between 12-60
- **Age**: 0-120 years (warnings for <18 or ≥100)
- **Height**: Must be >120cm (warnings for <150cm) 
- **Weight**: Must be 20-300kg

### 🧮 **Smart BMI Calculation**
- Automatically calculates BMI from weight/height when missing
- Handles missing data gracefully

## Quick Start

### Option 1: Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the web application
python src/web_gui.py
```

### Option 2: Docker
```bash
# Build and run with Docker
docker build -t doctor31 .
docker run -p 4000:4000 doctor31
```

Then open your browser to `http://localhost:4000`

## How to Use

1. **Upload** your CSV file containing patient data
2. **Map columns** to match your data format (age, weight, height, etc.)
3. **Preview** your data to verify column mapping
4. **Validate** to run the analysis and see results

The tool will highlight each row with color coding:
- 🟢 **Green**: Valid data
- 🟡 **Yellow**: Warning (needs attention)
- 🔴 **Red**: Anomaly (likely data error)

## Project Structure

```
src/
├── web_gui.py          # Flask web application
├── validation.py       # Core validation logic
├── templates/          # HTML templates
└── static/            # CSS/JS assets

tests/                 # Comprehensive test suite
docs/                  # Generated documentation
```

## Testing

Run the comprehensive test suite:

```bash
# Run tests with coverage
./run-unit-tests-and-coverage.sh

# Or run tests directly
pytest tests/
```

## Development

The project includes:
- Automated testing with pytest and property-based testing
- Code coverage reporting
- Docker development environment
- Documentation generation with pdoc
- Linting and code quality tools

## Technical Stack

- **Backend**: Python, Flask, pandas, scikit-learn
- **Frontend**: HTML5, Bootstrap, Chart.js
- **ML**: Isolation Forest for anomaly detection
- **Testing**: pytest, hypothesis for property-based testing
- **Deployment**: Docker, development containers

## Use Cases

Perfect for:
- Healthcare diagnostic app data quality assurance

---

Built for reliable detection of data anomalies in medical datasets with a focus on patient safety and data integrity.
