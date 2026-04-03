"""Classification pipeline for NeuroSpiral features.

Provides standard RF classification with subject-stratified cross-validation,
plus utilities for per-class metrics and confusion matrices.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def classify_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 15,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Subject-stratified cross-validated classification.

    Parameters
    ----------
    X : Feature matrix (n_samples, n_features).
    y : Labels (n_samples,).
    groups : Subject IDs for stratified group k-fold.
    n_estimators : Number of trees.
    max_depth : Maximum tree depth.
    n_splits : Number of CV folds.

    Returns
    -------
    dict with kappa, f1_macro, y_pred, y_true.
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    cv = StratifiedGroupKFold(n_splits=n_splits)
    y_pred = cross_val_predict(clf, X_s, y, groups=groups, cv=cv)

    return {
        "kappa": float(cohen_kappa_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "y_pred": y_pred,
        "y_true": y,
    }
