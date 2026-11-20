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



def test_manager_allocation_is_balanced_and_preserves_queue_order():
    import pandas as pd
    from src.retention import allocate_relationship_managers

    queue = pd.DataFrame({"Customer_ID": ["C1", "C2", "C3", "C4", "C5"]})
    result = allocate_relationship_managers(queue, ["Team A", "Team B"])

    assert result["Customer_ID"].tolist() == ["C1", "C2", "C3", "C4", "C5"]
    assert result["Assigned_Manager"].tolist() == [
        "Team A", "Team B", "Team A", "Team B", "Team A"
    ]
    assert result["Assigned_Manager"].value_counts().to_dict() == {"Team A": 3, "Team B": 2}


def test_manager_allocation_rejects_empty_or_duplicate_names():
    import pandas as pd
    from src.retention import allocate_relationship_managers

    with pytest.raises(ValueError, match="At least one"):
        allocate_relationship_managers(pd.DataFrame({"Customer_ID": ["C1"]}), [])
    with pytest.raises(ValueError, match="unique"):
        allocate_relationship_managers(pd.DataFrame({"Customer_ID": ["C1"]}), ["Team A", "Team A"])
