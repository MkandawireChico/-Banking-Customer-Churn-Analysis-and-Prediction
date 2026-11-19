"""Boundary tests for customer churn risk-band assignment."""
import pytest
from src.retention import RetentionRecommender


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.0, "Low Risk"),
        (0.329999, "Low Risk"),
        (0.33, "Medium Risk"),
        (0.659999, "Medium Risk"),
        (0.66, "High Risk"),
        (1.0, "High Risk"),
    ],
)
def test_risk_band_boundaries_are_consistent(probability, expected):
    assert RetentionRecommender().assign_risk_band(probability) == expected


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_risk_band_out_of_range_defaults_to_high(probability):
    # Out-of-range scores should not silently be labelled low risk.
    assert RetentionRecommender().assign_risk_band(probability) == "High Risk"
