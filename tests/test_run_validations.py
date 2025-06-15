import pytest
import pandas as pd
from hypothesis import given, strategies as st
from src.validation import bmi_calculated, validate_row


# =============================================================================
# BMI CALCULATION TESTS
# =============================================================================
class TestBMICalculated:
    def test_normal_bmi_calculation(self):
        """Test BMI calculation with normal values"""
        bmi = bmi_calculated(70, 175)  # 70kg, 175cm
        assert abs(bmi - 22.86) < 0.01

    def test_zero_height(self):
        """Test BMI calculation with zero height"""
        assert bmi_calculated(70, 0) is None

    def test_null_values(self):
        """Test BMI calculation with null values"""
        assert bmi_calculated(None, 175) is None
        assert bmi_calculated(70, None) is None
        assert bmi_calculated(None, None) is None

    def test_negative_values(self):
        """Test BMI calculation with negative values"""
        assert bmi_calculated(-70, 175) is None
        assert bmi_calculated(70, -175) is None

    def test_edge_cases(self):
        """Test BMI calculation edge cases"""
        # Very small positive values
        bmi = bmi_calculated(0.1, 0.1)
        assert bmi is not None

        # Large values
        bmi = bmi_calculated(500, 250)
        assert bmi is not None


# =============================================================================
# VALIDATION TESTS - ANOMALY CASES
# =============================================================================
@pytest.mark.parametrize(
    "bmi, age, height, weight, reason",
    [
        # BMI anomalies
        (None, 30, 170, 70, "null BMI"),
        (float("nan"), 30, 170, 70, "nan BMI"),
        (11.9, 30, 170, 70, "BMI < 12"),
        (60.1, 30, 170, 70, "BMI > 60"),

        # Age anomalies
        (30, None, 170, 70, "null age"),
        (30, float("nan"), 170, 70, "nan age"),
        (30, -1, 170, 70, "negative age"),
        (30, 121, 170, 70, "age > 120"),

        # Height anomalies
        (30, 30, None, 70, "null height"),
        (30, 30, float("nan"), 70, "nan height"),
        (30, 30, 119.9, 70, "height < 120"),

        # Weight anomalies
        (30, 30, 170, None, "null weight"),
        (30, 30, 170, float("nan"), "nan weight"),
        (30, 30, 170, 19.9, "weight < 20"),
        (30, 30, 170, 301, "weight > 300"),
    ],
)
def test_anomaly_cases(bmi, age, height, weight, reason):
    """Test all cases that should return Anomaly status"""
    status, color = validate_row(bmi, age, height, weight)
    assert status == "Anomaly", f"Expected Anomaly for {reason}"
    assert color == "red"


# =============================================================================
# VALIDATION TESTS - WARNING CASES
# =============================================================================
@pytest.mark.parametrize(
    "bmi, age, height, weight, reason",
    [
        (25, 17, 170, 70, "minor age"),
        (25, 0, 170, 70, "minor age boundary"),
        (25, 100, 170, 70, "elderly age"),
        (25, 120, 170, 70, "elderly age boundary"),
        (25, 30, 130, 70, "short height"),
        (25, 30, 149, 70, "short height boundary"),

        # Multiple warnings should still return Warning
        (25, 16, 140, 70, "minor + short"),
    ],
)
def test_warning_cases(bmi, age, height, weight, reason):
    """Test all cases that should return Warning status"""
    status, color = validate_row(bmi, age, height, weight)
    assert status == "Warning", f"Expected Warning for {reason}"
    assert color == "orange"


# =============================================================================
# VALIDATION TESTS - VALID CASES
# =============================================================================
@pytest.mark.parametrize(
    "bmi, age, height, weight, reason",
    [
        (22.5, 30, 175, 70, "typical adult"),
        (12.0, 18, 150, 20, "lower bounds"),
        (60.0, 99, 200, 300, "upper bounds"),
        (25, 25, 200, 80, "young adult"),
    ],
)
def test_valid_cases(bmi, age, height, weight, reason):
    """Test all cases that should return Valid status"""
    status, color = validate_row(bmi, age, height, weight)
    assert status == "Valid", f"Expected Valid for {reason}"
    assert color == "green"


# =============================================================================
# BOUNDARY TESTS
# =============================================================================
@pytest.mark.parametrize(
    "bmi, age, height, weight, expected_status, reason",
    [
        # BMI boundaries
        (12.0, 30, 170, 70, "Valid", "BMI at lower bound"),
        (11.99, 30, 170, 70, "Anomaly", "BMI just below lower bound"),
        (60.0, 30, 170, 70, "Valid", "BMI at upper bound"),
        (60.01, 30, 170, 70, "Anomaly", "BMI just above upper bound"),

        # Age boundaries
        (25, 18.0, 170, 70, "Valid", "Age at adult threshold"),
        (25, 17.99, 170, 70, "Warning", "Age just below adult"),
        (25, 100.0, 170, 70, "Warning", "Age at elderly threshold"),
        (25, 99.99, 170, 70, "Valid", "Age just below elderly"),
        (25, 120.0, 170, 70, "Warning", "Age at max warning"),
        (25, 120.01, 170, 70, "Anomaly", "Age just above max"),
        (25, 0.0, 170, 70, "Warning", "Age at warning lower bound"),
        (25, -0.01, 170, 70, "Anomaly", "Age below zero"),

        # Height boundaries
        (25, 30, 150.0, 70, "Valid", "Height at lower valid bound"),
        (25, 30, 149.99, 70, "Warning", "Height just below valid bound"),
        (25, 30, 120.0, 70, "Warning", "Height at warning lower bound"),
        (25, 30, 119.99, 70, "Anomaly", "Height just below warning bound"),

        # Weight boundaries
        (25, 30, 170, 20.0, "Valid", "Weight at lower bound"),
        (25, 30, 170, 19.99, "Anomaly", "Weight just below lower bound"),
        (25, 30, 170, 300.0, "Valid", "Weight at upper bound"),
        (25, 30, 170, 300.01, "Anomaly", "Weight just above upper bound"),
    ],
)
def test_boundary_conditions(bmi, age, height, weight, expected_status, reason):
    """Test boundary conditions for all parameters"""
    status, color = validate_row(bmi, age, height, weight)
    assert status == expected_status, f"Expected {expected_status} for {reason}"


# =============================================================================
# PRECEDENCE TESTS
# =============================================================================
def test_anomaly_overrides_warning():
    """Test that anomaly takes precedence over warning conditions"""
    # Multiple conditions: minor age (warning) + extreme weight (anomaly)
    status, color = validate_row(25, 16, 140, 350)
    assert status == "Anomaly"
    assert color == "red"

def test_anomaly_overrides_multiple_warnings():
    """Test anomaly overrides multiple warning conditions"""
    # Minor age + short height (warnings) + null BMI (anomaly)
    status, color = validate_row(None, 16, 140, 70)
    assert status == "Anomaly"
    assert color == "red"


# =============================================================================
# PROPERTY-BASED TESTS (using Hypothesis) - CORRECTED RANGES
# =============================================================================
# Define corrected strategy ranges
valid_bmi_range = st.floats(min_value=12.0, max_value=60.0, allow_nan=False, allow_infinity=False)
valid_age_range = st.floats(min_value=18.0, max_value=99.999, allow_nan=False, allow_infinity=False)
valid_height_range = st.floats(min_value=150.0, max_value=1000.0, allow_nan=False, allow_infinity=False)  # No upper limit in new logic
valid_weight_range = st.floats(min_value=20.0, max_value=300.0, allow_nan=False, allow_infinity=False)

# Corrected warning ranges
warning_age_range = st.one_of(
    st.floats(min_value=0.0, max_value=17.999, allow_nan=False, allow_infinity=False),
    st.floats(min_value=100.0, max_value=120.0, allow_nan=False, allow_infinity=False)
)
warning_height_range = st.floats(min_value=120.0, max_value=149.999, allow_nan=False, allow_infinity=False)


@given(bmi=valid_bmi_range, age=valid_age_range, height=valid_height_range, weight=valid_weight_range)
def test_property_all_valid_ranges_return_valid(bmi, age, height, weight):
    """Property test: all parameters in valid ranges should return Valid"""
    status, color = validate_row(bmi, age, height, weight)
    assert status == "Valid"
    assert color == "green"


@given(bmi=valid_bmi_range, age=warning_age_range, height=valid_height_range, weight=valid_weight_range)
def test_property_warning_age_returns_warning(bmi, age, height, weight):
    """Property test: warning age with valid other params should return Warning"""
    status, color = validate_row(bmi, age, height, weight)
    assert status == "Warning"
    assert color == "orange"


@given(bmi=valid_bmi_range, age=valid_age_range, height=warning_height_range, weight=valid_weight_range)
def test_property_warning_height_returns_warning(bmi, age, height, weight):
    """Property test: warning height with valid other params should return Warning"""
    status, color = validate_row(bmi, age, height, weight)
    assert status == "Warning"
    assert color == "orange"


# =============================================================================
# ADDITIONAL EDGE CASE TESTS
# =============================================================================
class TestEdgeCases:
    def test_extreme_values_combination(self):
        """Test combinations of extreme but valid values"""
        status, color = validate_row(12.0, 18.0, 150.0, 20.0)
        assert status == "Valid"
        assert color == "green"

    def test_float_precision_edge_cases(self):
        """Test float precision at boundaries"""
        # Test values very close to boundaries
        status, color = validate_row(11.9999999, 30, 170, 70)
        assert status == "Anomaly"

        status, color = validate_row(12.0000001, 30, 170, 70)
        assert status == "Valid"

    def test_multiple_anomalies(self):
        """Test multiple anomaly conditions"""
        status, color = validate_row(None, -1, -1, None)
        assert status == "Anomaly"
        assert color == "red"

    def test_mixed_conditions(self):
        """Test mixed valid and warning conditions"""
        # Valid BMI, warning age, valid height, valid weight -> Warning overall
        status, color = validate_row(25, 16, 170, 70)
        assert status == "Warning"
        assert color == "orange"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================
class TestIntegration:
    def test_bmi_calculation_with_validation_valid(self):
        """Test BMI calculation followed by validation - valid case"""
        weight, height = 70, 175
        bmi = bmi_calculated(weight, height)
        status, color = validate_row(bmi, 30, height, weight)

        assert abs(bmi - 22.86) < 0.01
        assert status == "Valid"
        assert color == "green"

    def test_bmi_calculation_with_validation_anomaly(self):
        """Test BMI calculation followed by validation - anomaly case"""
        weight, height = 70, 0  # Zero height
        bmi = bmi_calculated(weight, height)
        status, color = validate_row(bmi, 30, height, weight)

        assert bmi is None
        assert status == "Anomaly"
        assert color == "red"

    def test_full_workflow_warning(self):
        """Test full workflow resulting in warning"""
        weight, height = 70, 140  # Short height
        bmi = bmi_calculated(weight, height)
        status, color = validate_row(bmi, 30, height, weight)

        assert bmi is not None
        assert status == "Warning"
        assert color == "orange"