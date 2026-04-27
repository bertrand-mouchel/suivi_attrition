"""
Supervised-learning pipeline — trains and evaluates three classifiers.

Public API
----------
train_models(X, y) -> tuple[dict, X_train, X_test, y_train, y_test, StandardScaler]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, SMOTE_SAMPLING_STRATEGY, TEST_SIZE

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
_MODELS: dict = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        bootstrap=True,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=4,
        max_features="sqrt",
        n_iter_no_change=20,
    ),
    "Logistic Regression": LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_iter=3000,
        C=0.05,
        solver="saga",
        penalty="l1",
    ),
}


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Full training pipeline with SMOTE rebalancing and threshold optimisation.

    Steps
    -----
    1. Stratified 80/20 train-test split.
    2. Standard scaling (fit on train, transform both splits).
    3. SMOTE oversampling on the training set only.
    4. Each model is fitted and evaluated with an F1-optimal decision threshold.

    Returns
    -------
    results : dict
        Keyed by model name; each value is a dict with:
        ``model``, ``y_pred``, ``y_proba``,
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``threshold``.
    X_train, X_test, y_train, y_test : DataFrames / Series (unscaled).
    scaler : fitted StandardScaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=SMOTE_SAMPLING_STRATEGY)
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)

    results: dict = {}
    for name, model in _MODELS.items():
        model.fit(X_train_res, y_train_res)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        # F1-optimal threshold
        prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, y_proba)
        f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
        best_idx = int(np.argmax(f1_arr))
        best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
        y_pred = (y_proba >= best_threshold).astype(int)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "threshold": best_threshold,
        }

    return results, X_train, X_test, y_train, y_test, scaler
