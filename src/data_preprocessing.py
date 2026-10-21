"""Data loading and validation for Telco Customer Churn."""
from pathlib import Path
import pandas as pd

TARGET = "Churn"
ID_COLUMNS = ["customerID"]


def load_and_clean_data(path):
    """Load raw data and perform cleaning.

    Numeric/categorical imputation is intentionally NOT done here: it is fitted
    inside each sklearn pipeline using training data only.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Put the Telco CSV in data/.")
    df = pd.read_csv(path)
    original_rows = len(df)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop_duplicates().copy()
    duplicate_rows_removed = original_rows - len(df)

    if TARGET not in df:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    if "TotalCharges" in df:
        # The source contains whitespace-only strings; coerce them to missing.
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype("string").str.strip().replace("", pd.NA),
            errors="coerce",
        )
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    # Remove records without a usable target; never impute the label.
    df[TARGET] = df[TARGET].astype("string").str.strip()
    df = df[df[TARGET].isin(["Yes", "No"])].copy()
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    for col in ID_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)

    audit = {
        "rows_before_duplicate_removal": int(original_rows),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "rows_after_cleaning": int(len(df)),
        "target_missing_or_invalid_rows_removed": int(original_rows - duplicate_rows_removed - len(df)),
        "missing_values_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "numeric_columns": [str(c) for c in df.drop(columns=[TARGET]).select_dtypes(include="number").columns],
        "categorical_columns": [str(c) for c in df.drop(columns=[TARGET]).select_dtypes(exclude="number").columns],
        "target_distribution": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
    }
    return df, audit
