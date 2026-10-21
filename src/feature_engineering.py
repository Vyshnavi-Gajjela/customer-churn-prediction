"""Deterministic, leakage-safe feature engineering."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain features without learning statistics from validation/test rows."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "tenure" in X and "TotalCharges" in X:
            tenure = pd.to_numeric(X["tenure"], errors="coerce")
            total = pd.to_numeric(X["TotalCharges"], errors="coerce")
            monthly = pd.to_numeric(X.get("MonthlyCharges", pd.Series(np.nan, index=X.index)), errors="coerce")
            # For tenure=0, use current monthly charge rather than divide by zero.
            X["AvgMonthlySpend"] = np.where(tenure > 0, total / tenure, monthly)
        if all(c in X.columns for c in SERVICE_COLUMNS):
            X["TotalServices"] = sum((X[c].astype("string").str.strip() == "Yes").astype("int8") for c in SERVICE_COLUMNS)
        if "tenure" in X:
            t = pd.to_numeric(X["tenure"], errors="coerce")
            X["TenureGroup"] = pd.cut(
                t, bins=[-np.inf, 12, 24, 48, np.inf],
                labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
                right=True, include_lowest=True,
            ).astype("string")
        if "Contract" in X:
            X["MonthlyContractRisk"] = (X["Contract"].astype("string").str.strip() == "Month-to-month").map({True: "Yes", False: "No"}).astype("string")
        return X
