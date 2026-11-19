"""Tests for reproducible dataset generation."""
import importlib.util
from pathlib import Path
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_dataset", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_generator_default_output_is_repository_relative():
    expected = Path(__file__).resolve().parents[1] / "data" / "raw" / "churn_customers_synthetic.csv"
    assert generator.PROJECT_ROOT / "data" / "raw" / "churn_customers_synthetic.csv" == expected


def test_clean_dataset_is_reproducible():
    first = generator.generate_clean_dataset(random_state=17, n=120)
    second = generator.generate_clean_dataset(random_state=17, n=120)
    pd.testing.assert_frame_equal(first, second)


def test_quality_issue_injection_is_reproducible_and_preserves_schema():
    clean = generator.generate_clean_dataset(random_state=42, n=500)
    first = generator.inject_data_quality_issues(clean, random_state=42)
    second = generator.inject_data_quality_issues(clean, random_state=42)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) > len(clean)
    assert first.duplicated().any()
    assert first["Monthly_Income"].isna().any()
    assert first["Marital_Status"].isna().any()
    assert {"Customer_ID", "Churn", "Customer_Segment", "Branch_Location"}.issubset(first.columns)
