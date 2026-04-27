"""
Feature engineering — assembles the (X, y) matrix fed into ML models.

Public API
----------
create_feature_matrix(df_processed) -> tuple[pd.DataFrame, pd.Series, list[str]]
"""

from __future__ import annotations

import pandas as pd

# Ordered list of columns included in every model.
FEATURE_COLUMNS: list[str] = [
    # Numeric
    "Age",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    # Encoded categoricals
    "BusinessTravel_Encoded",
    "Department_Encoded",
    "EducationField_Encoded",
    "Gender_Encoded",
    "JobRole_Encoded",
    "MaritalStatus_Encoded",
    "OverTime_Encoded",
    # Derived / interaction features
    "OT_Binary",
    "Single",
    "Income_per_Year",
    "Years_Since_Promo_Ratio",
    "Satisfaction_Avg",
    "Young_New",
    "No_StockOption",
    "Travel_Freq",
]


def create_feature_matrix(
    df_processed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build the feature matrix ``X`` and target vector ``y``.

    Only columns actually present in *df_processed* are included, so the
    function is safe even if upstream preprocessing evolves over time.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary attrition target (0 = stayed, 1 = left).
    available_cols : list[str]
        Names of columns retained in ``X``.
    """
    available_cols = [c for c in FEATURE_COLUMNS if c in df_processed.columns]
    X = df_processed[available_cols]
    y = df_processed["Attrition_Binary"]
    return X, y, available_cols
