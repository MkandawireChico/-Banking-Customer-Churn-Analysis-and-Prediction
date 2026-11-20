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


def test_case_register_starter_selects_highest_risk_customers():
    import pandas as pd
    from src.case_management import build_starter_case_register

    customers = pd.DataFrame({
        "Customer_ID": ["C1", "C2", "C3"],
        "Churn_Probability": [0.2, 0.9, 0.7],
        "Risk_Band": ["Low Risk", "High Risk", "High Risk"],
    })
    result = build_starter_case_register(customers, limit=1)
    assert result["Customer_ID"].tolist() == ["C2"]
    assert result.iloc[0]["Case_Status"] == "New"


def test_case_register_rejects_duplicate_ids_and_invalid_status():
    import pandas as pd
    import pytest
    from src.case_management import build_starter_case_register, validate_case_register

    starter = build_starter_case_register(pd.DataFrame({
        "Customer_ID": ["C1", "C2"], "Churn_Probability": [0.8, 0.7],
        "Risk_Band": ["High Risk", "High Risk"],
    }))
    duplicate = pd.concat([starter, starter.iloc[[0]]], ignore_index=True)
    duplicate.loc[len(duplicate) - 1, "Case_ID"] = "RET-99999"
    with pytest.raises(ValueError, match="Customer_ID values must be unique"):
        validate_case_register(duplicate)
    starter.loc[0, "Case_Status"] = "Unknown"
    with pytest.raises(ValueError, match="Invalid case status"):
        validate_case_register(starter)


def test_overdue_follow_up_excludes_closed_cases_and_future_dates():
    import pandas as pd
    from datetime import date
    from src.case_management import add_case_monitoring_fields

    frame = pd.DataFrame({
        "Case_Status": ["Follow-up required", "Retained", "Assigned"],
        "Next_Follow_Up": ["2026-11-19", "2026-11-19", "2026-11-21"],
    })
    result = add_case_monitoring_fields(frame, today=date(2026, 11, 20))
    assert result["Follow_Up_Overdue"].tolist() == [True, False, False]
