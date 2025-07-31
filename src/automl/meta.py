from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor

import torch

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tabpfn import TabPFNRegressor


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

algorithms = [
    LGBMRegressor(n_jobs=-1),
    XGBRegressor(enable_categorical = True, n_jobs = -1),
    RandomForestRegressor(n_jobs=-1),
    LinearRegression(n_jobs=-1),
    SVR(),
    # TabPFNRegressor(n_jobs=-1, ignore_pretraining_limits=True),
    # device="cuda" if torch.cuda.is_available() else "cpu",
]


datasets = [
    "bike_sharing_demand",
    "brazilian_houses",
    "superconductivity",
    "wine_quality",
    "yprop_4_1"
]

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
    X: pd.DataFrame,
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
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        transformers.append(("categorical", Pipeline(steps=cat_steps), cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    preprocessor.set_output(transform="pandas")
    return preprocessor

def extract_meta_features(X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute numeric-only meta-features for regression datasets."""
    # Basic sizes
    n, d = X.shape
    meta = {
        "n_samples": n,
        "n_features": d,
        "feature_ratio": d / n if n else 0.0,
        "target_std": float(y.std()) if len(y) else 0.0,
        "target_skew": float(skew(y)) if len(y) else 0.0,
    }

    # Restrict to numeric cols
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        # no numeric features → fill zeros
        meta.update({
            "mean_feature_skew": 0.0,
            "mean_feature_kurtosis": 0.0,
            "zero_var_pct": 1.0,
            "mean_abs_corr": 0.0,
            "max_abs_corr": 0.0,
        })
        return meta

    # 1) Per-feature mean skew/kurtosis
    # Compute skew/kurtosis across columns means (column-wise)
    means = X_num.mean(axis=0)
    meta["mean_feature_skew"]     = float(skew(means))
    meta["mean_feature_kurtosis"] = float(kurtosis(means))

    # 2) Zero-variance percentage
    zero_var = (X_num.var(axis=0) == 0).sum()
    meta["zero_var_pct"] = float(zero_var / X_num.shape[1])

    # 3) Pairwise correlations (only if >1 numeric column)
    if X_num.shape[1] > 1:
        corr = X_num.corr().abs()
        # take upper triangle without diagonal
        iu = np.triu_indices(corr.shape[0], k=1)
        tri_vals = corr.values[iu]
        meta["mean_abs_corr"] = float(np.nanmean(tri_vals))
        meta["max_abs_corr"]  = float(np.nanmax(tri_vals))
    else:
        meta["mean_abs_corr"] = 0.0
        meta["max_abs_corr"]  = 0.0

    return meta


def load_dataset(name: str, fold: int = 1):
    """Load X_train, y_train, X_test, y_test for a given dataset and fold."""
    base = Path(__file__).resolve().parents[2] / "data" / name / str(fold)
    X_train = pd.read_parquet(base / "X_train.parquet")
    y_train = pd.read_parquet(base / "y_train.parquet").iloc[:, 0]
    X_test  = pd.read_parquet(base / "X_test.parquet")
    y_test  = pd.read_parquet(base / "y_test.parquet").iloc[:, 0]
    return X_train, y_train, X_test, y_test


def main():
    records = []

    for ds in datasets:
        print(f"→ Processing dataset: {ds}")
        X_train, y_train, X_test, y_test = load_dataset(ds)

        preprocessor = build_preprocessor(pd.concat([X_train, X_test], ignore_index=True))

        # 1) extract meta-features
        meta = extract_meta_features(X_train, y_train)

        # 2) evaluate each algorithm
        scores = {}
        for Algo in algorithms:
            name = Algo.__class__.__name__
            print(f"   • Training {name}...", end=" ", flush=True)
            try:
                # model = Algo()
                model = Pipeline([
                            ("preproc", preprocessor),
                            ("model", Algo)
                        ])
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
            except Exception as e:
                print(f"[Error: {e}]")
                # fallback to dummy
                dummy = DummyRegressor()
                dummy.fit(X_train, y_train)
                r2 = r2_score(y_test, dummy.predict(X_test))

            scores[name] = float(r2)
            print(f"R²={r2:.4f}")

        # 3) compute ranks (1 = best)
        sorted_names = sorted(scores, key=lambda k: -scores[k])
        ranks = { name: idx+1 for idx, name in enumerate(sorted_names) }

        # 4) assemble record
        record = {
            "dataset": ds,
            **meta,
            **{ f"{n}_r2": scores[n] for n in scores },
            **{ f"{n}_rank": ranks[n] for n in ranks },
        }
        records.append(record)

    # 5) save to CSV
    df = pd.DataFrame(records)
    df.to_csv("meta_dataset.csv", index=False)
    print("\nSaved meta-dataset to meta_dataset.csv")


if __name__ == "__main__":
    main()
