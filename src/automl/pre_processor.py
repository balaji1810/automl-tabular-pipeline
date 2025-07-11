from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def build_numeric_preprocessor() -> Pipeline:
    """
    Returns a sklearn Pipeline that:
      1. applies a RobustScaler to mitigate outliers
    Assumes input X has no NaNs or categorical columns.
    """
    steps: list[tuple[str, object]] = []

    steps.append(
        ("robust_scaler", RobustScaler()) # for now it only handles outliers, Need to add more pre-processing methods
    )

    return Pipeline(steps)
