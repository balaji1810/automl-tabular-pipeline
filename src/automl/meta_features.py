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

from automl.pre_processor_old import build_preprocessor
from automl.algorithms import algorithms_dict

import warnings
import os

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*LightGBM.*")
warnings.filterwarnings("ignore", message=".*No further splits.*")
warnings.filterwarnings("ignore", message=".*Stopped training.*")
os.environ['LIGHTGBM_VERBOSITY'] = '-1'  # Suppress LightGBM verbose output

# Suppress other common ML warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Additional LightGBM suppression
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)


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
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series")
    
    # CRITICAL: Convert sparse target to dense if needed
    if hasattr(y, 'sparse') or str(y.dtype).startswith('Sparse'):
        y = y.sparse.to_dense() if hasattr(y, 'sparse') else pd.Series(y.values, dtype=float)
    # Preprocess the data in X first
    try:
        preprocessor = build_preprocessor(X)
        try:
            preprocessor.set_output(transform="default")  # Force numpy output
        except AttributeError:
            pass  # Some transformers don't have set_output method
        if hasattr(preprocessor, 'sparse_threshold'):
            preprocessor.sparse_threshold = 0  # Force dense output
        if hasattr(preprocessor, 'transformers'):
            for name, transformer, columns in preprocessor.transformers:
                # Handle Pipeline transformers
                if hasattr(transformer, 'steps'):
                    for step_name, step_transformer in transformer.steps:
                        if hasattr(step_transformer, 'sparse_threshold'):
                            step_transformer.sparse_threshold = 0
                        if hasattr(step_transformer, 'sparse'):
                            step_transformer.sparse = False
                        # Fix StandardScaler centering issue
                        if hasattr(step_transformer, 'with_centering'):
                            step_transformer.with_centering = False
                            # print(f"   - Disabled centering for {step_name}")
                # Handle direct transformers
                else:
                    if hasattr(transformer, 'sparse_threshold'):
                        transformer.sparse_threshold = 0
                    if hasattr(transformer, 'sparse'):
                        transformer.sparse = False
                    if hasattr(transformer, 'with_centering'):
                        transformer.with_centering = False
                        # print(f"   - Disabled centering for {name}")
        
        X_processed = preprocessor.fit_transform(X)
        
    except Exception as e:
        raise ValueError("Preprocessing failed. Check your data format and types.")
    
    # Convert sparse matrix to dense if needed
    if sparse.issparse(X_processed):
        X_processed = X_processed.toarray()
    
    if isinstance(X_processed, np.ndarray):
        X_processed = pd.DataFrame(X_processed)
    elif hasattr(X_processed, 'sparse'):
        X_processed = X_processed.sparse.to_dense()

    if isinstance(X_processed, pd.DataFrame):
        for col in X_processed.columns:
            if hasattr(X_processed[col], 'sparse'):
                X_processed[col] = X_processed[col].sparse.to_dense()

    
    # Basic sizes (using both original and processed dimensions)
    meta_features = {
        "n_samples": n,
        "n_features": d,
        "log_n_samples": np.log(n + 1) if n > 0 else 0.0,
        "log_n_features": np.log(d + 1) if d > 0 else 0.0,
        "feature_ratio": d / n if n else 0.0,
    }
    
    # Target statistics with sparse handling
    try:
        # Ensure y is dense for statistical operations
        y_dense = y.sparse.to_dense() if hasattr(y, 'sparse') else y
        if hasattr(y_dense, 'values'):
            y_values = y_dense.values
        else:
            y_values = np.array(y_dense)
            
        meta_features.update({
            "target_mean": float(np.mean(y_values)) if len(y_values) else 0.0,
            "target_std": float(np.std(y_values)) if len(y_values) else 0.0,
            "target_skew": float(skew(y_values)) if len(y_values) else 0.0,
            "target_kurtosis": float(kurtosis(y_values)) if len(y_values) else 0.0,
        })
    except Exception as target_error:
        meta_features.update({
            "target_mean": 0.0,
            "target_std": 0.0,
            "target_skew": 0.0,
            "target_kurtosis": 0.0,
        })

    # Work with processed data (should all be numeric now)
    X_num = X_processed
    
    # Check if any columns are still sparse
    if isinstance(X_num, pd.DataFrame):
        sparse_columns = []
        for col in X_num.columns:
            if hasattr(X_num[col], 'sparse') or str(X_num[col].dtype).startswith('Sparse'):
                sparse_columns.append(col)
        if sparse_columns:
            for col in sparse_columns:
                X_num[col] = X_num[col].sparse.to_dense() if hasattr(X_num[col], 'sparse') else pd.array(X_num[col], dtype=float)
    
    if X_num.shape[1] == 0: 
        meta_features.update({
            "mean_feature_skew": 0.0,
            "mean_feature_kurtosis": 0.0,
            # "zero_var_pct": 1.0,
            "mean_abs_corr": 0.0,
            "max_abs_corr": 0.0,
        })
        
    else:
        try:
            # Ensure all columns are dense before statistical operations
            if isinstance(X_num, pd.DataFrame):
                for col in X_num.columns:
                    if hasattr(X_num[col], 'sparse') or str(X_num[col].dtype).startswith('Sparse'):
                        X_num[col] = X_num[col].sparse.to_dense() if hasattr(X_num[col], 'sparse') else pd.array(X_num[col].values, dtype=float)
            
            # 1) Per-feature mean skew/kurtosis
            # Compute skew/kurtosis across columns means (column-wise)
            means = X_num.mean(axis=0)
            
            # Additional safety check for means
            if hasattr(means, 'sparse') or (hasattr(means, 'dtype') and str(means.dtype).startswith('Sparse')):
                means = means.sparse.to_dense() if hasattr(means, 'sparse') else pd.Series(means.values, dtype=float)
            
            meta_features["mean_feature_skew"]     = float(skew(means))
            meta_features["mean_feature_kurtosis"] = float(kurtosis(means))

            # 2) Pairwise correlations (only if >1 numeric column)
            if X_num.shape[1] > 1:
                # Force conversion to float64 dense DataFrame for correlation computation
                X_corr = X_num.astype(float) if isinstance(X_num, pd.DataFrame) else pd.DataFrame(X_num, dtype=float)
                corr = X_corr.corr().abs()
                # take upper triangle without diagonal
                iu = np.triu_indices(corr.shape[0], k=1)
                tri_vals = corr.values[iu]
                meta_features["mean_abs_corr"] = float(np.nanmean(tri_vals))
                meta_features["max_abs_corr"]  = float(np.nanmax(tri_vals))
            else:
                meta_features["mean_abs_corr"] = 0.0
                meta_features["max_abs_corr"]  = 0.0
                
        except Exception as stat_error:
            meta_features.update({
                "mean_feature_skew": 0.0,
                "mean_feature_kurtosis": 0.0,
                "mean_abs_corr": 0.0,
                "max_abs_corr": 0.0,
            })

    # 3) Probing Features
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
    
    # Ensure we work with dense arrays for meta-feature computation
    X_train_processed = _to_dense_array(X_train)
    X_test_processed = _to_dense_array(X_test)
    sample_fraction = 0.01  # 1% sample for probing

    # Mean value predictor performance (baseline)
    dummy_mean = DummyRegressor(strategy='mean')
    dummy_mean.fit(X_train_processed, y_train)
    y_pred_mean = dummy_mean.predict(X_test_processed)
    
    meta_features['mean_predictor_r2'] = r2_score(y_test, y_pred_mean)
    
    # Decision stump performance (depth=1 tree)
    stump = DecisionTreeRegressor(max_depth=1, random_state=42)
    try:
        stump.fit(X_train_processed, y_train)
        y_pred_stump = stump.predict(X_test_processed)

        meta_features['decision_stump_r2'] = r2_score(y_test, y_pred_stump)
    except Exception as e:
        meta_features['decision_stump_r2'] = meta_features['mean_predictor_r2']

    # Simple rule model performance (linear regression)
    simple_rule = LinearRegression()
    simple_rule.fit(X_train_processed, y_train)
    y_pred_rule = simple_rule.predict(X_test_processed)

    meta_features['simple_rule_r2'] = r2_score(y_test, y_pred_rule)
        
    
    # Performance of algorithms on 1% of data
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
                
            except Exception as e:
                meta_features[f'{algo_name}_1pct_r2'] = -1.0
                meta_features[f'{algo_name}_1pct_rmse'] = float('inf')
    meta_features['tree_advantage'] = (
        meta_features['decision_stump_r2'] - meta_features['simple_rule_r2']
    )

    # 5) Algorithm suitability indicators
    meta_features['tabpfn_suitable'] = 1 if (n < 1000 and d < 100 and n*d < 10000) else 0
    meta_features['svm_suitable'] = 1 if (100 <= n <= 10000 and d <= 1000) else 0
    meta_features['linear_favorable'] = 1 if (meta_features['simple_rule_r2'] > 0.7 and 
                                            meta_features['target_skew'] < 2) else 0

    return meta_features
