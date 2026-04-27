"""
Data ingestion and preprocessing pipeline.

Public API
----------
load_data()        -> pd.DataFrame
preprocess_data()  -> tuple[pd.DataFrame, dict[str, LabelEncoder]]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Default path relative to the project root
_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


@st.cache_data(show_spinner=False)
def load_data(path: str | Path = _DATA_PATH) -> pd.DataFrame:
    """Load the raw IBM HR Attrition dataset from *path*."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Enrich and encode the raw dataframe.

    Steps
    -----
    1. Binary target column ``Attrition_Binary``.
    2. Derived interaction features (overtime flag, satisfaction index, …).
    3. Label-encode every categorical column (except ``Attrition`` itself).

    Returns
    -------
    df_processed : pd.DataFrame
        Original rows augmented with encoded & engineered columns.
    le_dict : dict[str, LabelEncoder]
        One fitted encoder per categorical column (keyed by original column name).
    """
    df_processed = df.copy()

    # ── Binary target ────────────────────────────────────────────────────────
    df_processed["Attrition_Binary"] = (df_processed["Attrition"] == "Yes").astype(int)

    # ── Derived features ─────────────────────────────────────────────────────
    df_processed["OT_Binary"] = (df_processed["OverTime"] == "Yes").astype(int)
    df_processed["Single"] = (df_processed["MaritalStatus"] == "Single").astype(int)
    df_processed["Income_per_Year"] = (
        df_processed["MonthlyIncome"] / (df_processed["TotalWorkingYears"] + 1)
    )
    df_processed["Years_Since_Promo_Ratio"] = (
        df_processed["YearsSinceLastPromotion"] / (df_processed["YearsAtCompany"] + 1)
    )
    df_processed["Satisfaction_Avg"] = (
        df_processed["JobSatisfaction"]
        + df_processed["EnvironmentSatisfaction"]
        + df_processed["RelationshipSatisfaction"]
        + df_processed["WorkLifeBalance"]
    ) / 4
    df_processed["Young_New"] = (
        (df_processed["Age"] < 30) & (df_processed["YearsAtCompany"] < 3)
    ).astype(int)
    df_processed["No_StockOption"] = (df_processed["StockOptionLevel"] == 0).astype(int)
    df_processed["Travel_Freq"] = (
        df_processed["BusinessTravel"] == "Travel_Frequently"
    ).astype(int)

    # ── Label encoding ───────────────────────────────────────────────────────
    le_dict: dict[str, LabelEncoder] = {}
    categorical_cols = df_processed.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        if col == "Attrition":
            continue
        le = LabelEncoder()
        df_processed[f"{col}_Encoded"] = le.fit_transform(df_processed[col])
        le_dict[col] = le

    return df_processed, le_dict
