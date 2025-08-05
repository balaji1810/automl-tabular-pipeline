from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
# from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler


def detect_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Detect numerical and categorical columns in a DataFrame.
    Returns lists of column names.
    """
    # A simple heuristic: dtype kind
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # Also treat low-cardinality ints as categorical
    # for col in X.select_dtypes(include=["int"]):
    #     if X[col].nunique() < 20 and col not in cat_cols:
    #         cat_cols.append(col)
    #         num_cols.remove(col)
    return num_cols, cat_cols


def build_preprocessor(
    X: pd.DataFrame
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing:
      - numeric: impute + optional scaling
      - categorical: impute + one-hot encode
    """
    num_cols, cat_cols = detect_column_types(X)

    transformers = []
    if num_cols:
        num_steps = []
        num_steps.append(("imputer", SimpleImputer(strategy="mean")))
        # num_steps.append(("scaler", StandardScaler()))
        num_steps.append(("robust_scaler", RobustScaler()))
        transformers.append(("numerical", Pipeline(steps=num_steps), num_cols))
    if cat_cols:
        cat_steps = []
        cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        cat_steps.append(("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        transformers.append(("categorical", Pipeline(steps=cat_steps), cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    preprocessor.set_output(transform="pandas")
    return preprocessor