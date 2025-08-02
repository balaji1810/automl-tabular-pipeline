from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy import sparse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from pre_processor import build_preprocessor
from constants import algorithms_dict

def _to_dataframe(X):
    """Convert various array types to DataFrame safely."""
    if isinstance(X, pd.DataFrame):
        return X
    elif sparse.issparse(X):
        return pd.DataFrame(X.toarray())
    else:
        return pd.DataFrame(X)


def _to_dense_array(X):
    """Convert various array types to dense numpy array safely."""
    if sparse.issparse(X):
        return X.toarray()
    elif isinstance(X, pd.DataFrame):
        return X.values
    else:
        return X


def extract_meta_features(X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> dict:
    """Compute meta-features for regression datasets."""
    # Store original dimensions before preprocessing
    n, d = X.shape
    print("=========== Extracting meta-features ===========")
    print(" X Type:", type(X), "Shape:", X.shape)
    print(" y Type:", type(y), "Shape:", y.shape)
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series")
    # Preprocess the data in X first
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    # Convert back to DataFrame for consistent handling
    # X_processed = _to_dataframe(X_processed)

    # if hasattr(X_processed, 'toarray'):  # Check if it's a sparse matrix
    #     print(" sparse matrix detected")
    
    # Get dimensions after preprocessing
    # n, d = X_processed.shape
    
    # Basic sizes (using both original and processed dimensions)
    meta_features = {
        "n_samples": n,
        "n_features": d,
        # "n_features_processed": d,
        "log_n_samples": np.log(n + 1) if n > 0 else 0.0,
        "log_n_features": np.log(d + 1) if d > 0 else 0.0,
        "feature_ratio": d / n if n else 0.0,
        # "feature_expansion_ratio": d / d if d > 0 else 1.0,
        "target_mean": float(y.mean()) if len(y) else 0.0,
        "target_std": float(y.std()) if len(y) else 0.0,
        "target_skew": float(skew(y)) if len(y) else 0.0,
        "target_kurtosis": float(kurtosis(y)) if len(y) else 0.0,
    }

    # Work with processed data (should all be numeric now)
    X_num = X_processed
    if X_num.shape[1] == 0: 
        # no numeric features → fill zeros
        meta_features.update({
            "mean_feature_skew": 0.0,
            "mean_feature_kurtosis": 0.0,
            # "zero_var_pct": 1.0,
            "mean_abs_corr": 0.0,
            "max_abs_corr": 0.0,
        })
        
    else:
        # 1) Per-feature mean skew/kurtosis
        # Compute skew/kurtosis across columns means (column-wise)
        means = X_num.mean(axis=0)
        meta_features["mean_feature_skew"]     = float(skew(means))
        meta_features["mean_feature_kurtosis"] = float(kurtosis(means))

        # 2) Zero-variance percentage
        # zero_var = (X_num.var(axis=0) == 0).sum()
        # meta_features["zero_var_pct"] = float(zero_var / X_num.shape[1])

        # 3) Pairwise correlations (only if >1 numeric column)
        if X_num.shape[1] > 1:
            corr = X_num.corr().abs()
            # take upper triangle without diagonal
            iu = np.triu_indices(corr.shape[0], k=1)
            tri_vals = corr.values[iu]
            meta_features["mean_abs_corr"] = float(np.nanmean(tri_vals))
            meta_features["max_abs_corr"]  = float(np.nanmax(tri_vals))
        else:
            meta_features["mean_abs_corr"] = 0.0
            meta_features["max_abs_corr"]  = 0.0

    # 4) Probing Features
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
    
    # Ensure we work with dense arrays for meta-feature computation
    X_train_processed = _to_dense_array(X_train)
    X_test_processed = _to_dense_array(X_test)

    sample_fraction = 0.01  # 1% sample for probing
    # 4.1) Mean value predictor performance (baseline)
    dummy_mean = DummyRegressor(strategy='mean')
    dummy_mean.fit(X_train_processed, y_train)
    y_pred_mean = dummy_mean.predict(X_test_processed)
    
    meta_features['mean_predictor_r2'] = r2_score(y_test, y_pred_mean)
    
    # 4.2) Decision stump performance (depth=1 tree)
    stump = DecisionTreeRegressor(max_depth=1, random_state=42)
    stump.fit(X_train_processed, y_train)
    y_pred_stump = stump.predict(X_test_processed)
    
    meta_features['decision_stump_r2'] = r2_score(y_test, y_pred_stump)
    
    # Relative improvement over mean predictor
    meta_features['stump_vs_mean_r2_ratio'] = (
        meta_features['decision_stump_r2'] / max(meta_features['mean_predictor_r2'], 1e-10)
    )
    
    # 4.3) Simple rule model performance (linear regression)
    simple_rule = LinearRegression()
    simple_rule.fit(X_train_processed, y_train)
    y_pred_rule = simple_rule.predict(X_test_processed)

    meta_features['simple_rule_r2'] = r2_score(y_test, y_pred_rule)
        
    # Relative improvement over mean predictor
    meta_features['rule_vs_mean_r2_ratio'] = (
        meta_features['simple_rule_r2'] / max(meta_features['mean_predictor_r2'], 1e-10)
    )
    
    # 4.4) Performance of algorithms on 1% of data
    if X_train_processed.shape[0] > 100:  # Only if we have enough data
        # Sample 1% of training data
        sample_size = max(int(X_train_processed.shape[0] * sample_fraction), 10)
        sample_indices = np.random.choice(X_train_processed.shape[0], sample_size, replace=False)
        # X_train_processed is now a numpy array after _to_dense_array conversion
        X_sample = X_train_processed[sample_indices]
        y_sample = y_train.iloc[sample_indices]

        for algo_name, algorithm in algorithms_dict.items():
            try:
                # Clone the algorithm to avoid fitting issues
                from sklearn.base import clone
                algo_clone = clone(algorithm)
                
                # Fit on 1% sample
                algo_clone.fit(X_sample, y_sample)
                
                # Predict on test set
                y_pred_algo = algo_clone.predict(X_test_processed)
                
                # Store performance
                algo_r2 = r2_score(y_test, y_pred_algo)
                
                meta_features[f'{algo_name}_1pct_r2'] = algo_r2
                
                # Relative performance vs baselines
                meta_features[f'{algo_name}_vs_mean_r2_ratio'] = (
                    algo_r2 / max(meta_features['mean_predictor_r2'], 1e-10)
                )
                meta_features[f'{algo_name}_vs_stump_r2_ratio'] = (
                    algo_r2 / max(meta_features['decision_stump_r2'], 1e-10)
                )
                
            except Exception as e:
                print(f"Error evaluating {algo_name} on 1% data: {e}")
                meta_features[f'{algo_name}_1pct_r2'] = -1.0
                meta_features[f'{algo_name}_1pct_rmse'] = float('inf')
                meta_features[f'{algo_name}_vs_mean_r2_ratio'] = 0.0
                meta_features[f'{algo_name}_vs_stump_r2_ratio'] = 0.0
    
    # 4.5) Additional derived meta-features
    # meta_features['baseline_difficulty'] = 1 - meta_features['mean_predictor_r2']
    # meta_features['linear_separability'] = meta_features['simple_rule_r2']
    meta_features['tree_advantage'] = (
        meta_features['decision_stump_r2'] - meta_features['simple_rule_r2']
    )

    # 5) Algorithm suitability indicators
    meta_features['tabpfn_suitable'] = 1 if (n < 1000 and d < 100 and n*d < 10000) else 0
    meta_features['svm_suitable'] = 1 if (100 <= n <= 10000 and d <= 1000) else 0
    meta_features['linear_favorable'] = 1 if (meta_features['simple_rule_r2'] > 0.7 and 
                                            meta_features['target_skew'] < 2) else 0

    return meta_features
