"""Boundary and validation tests for customer churn risk bands."""
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


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf"), float("-inf")])
def test_risk_band_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="finite value between 0 and 1"):
        RetentionRecommender().assign_risk_band(probability)
