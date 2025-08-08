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
    print("=========== Extracting meta-features ===========")
    print(" X Type:", type(X), "Shape:", X.shape)
    print(" y Type:", type(y), "Shape:", y.shape)
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series")
    
    # CRITICAL: Convert sparse target to dense if needed
    if hasattr(y, 'sparse') or str(y.dtype).startswith('Sparse'):
        print(" • Converting sparse target y to dense")
        y = y.sparse.to_dense() if hasattr(y, 'sparse') else pd.Series(y.values, dtype=float)
    # Preprocess the data in X first
    try:
        print("Building preprocessor...")
        preprocessor = build_preprocessor(X)
        print("Fitting preprocessor...")
        print("X_processed shape before :", X.shape)
        print("Type of X:", type(X))
        try:
            preprocessor.set_output(transform="default")  # Force numpy output
        except AttributeError:
            pass  # Some transformers don't have set_output method
        # ADDITIONAL FIX: Handle the ColumnTransformer's sparse_threshold
        if hasattr(preprocessor, 'sparse_threshold'):
            preprocessor.sparse_threshold = 0  # Force dense output
            print(" • Set sparse_threshold=0 to force dense output")
        if hasattr(preprocessor, 'transformers'):
            print(" • Configuring ColumnTransformer components...")
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
                            print(f"   - Disabled centering for {step_name}")
                # Handle direct transformers
                else:
                    if hasattr(transformer, 'sparse_threshold'):
                        transformer.sparse_threshold = 0
                    if hasattr(transformer, 'sparse'):
                        transformer.sparse = False
                    if hasattr(transformer, 'with_centering'):
                        transformer.with_centering = False
                        print(f"   - Disabled centering for {name}")
        
        X_processed = preprocessor.fit_transform(X)
        
        print(f" • Preprocessor output type: {type(X_processed)}")
        print(f" • Preprocessor output shape: {X_processed.shape}")
        print(f" • Is sparse matrix?: {sparse.issparse(X_processed)}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise ValueError("Preprocessing failed. Check your data format and types.")
    
    # Convert sparse matrix to dense if needed
    if sparse.issparse(X_processed):
        print(" • Converting sparse matrix to dense for meta-features extraction")
        X_processed = X_processed.toarray()
        print(f" • After toarray(): type={type(X_processed)}, shape={X_processed.shape}")
    
    # CRITICAL: Convert to proper DataFrame for consistent handling
    if isinstance(X_processed, np.ndarray):
        print(" • Converting numpy array to DataFrame")
        X_processed = pd.DataFrame(X_processed)
    elif hasattr(X_processed, 'sparse'):
        print(" • Converting sparse DataFrame to dense DataFrame")
        # Handle pandas sparse DataFrame
        X_processed = X_processed.sparse.to_dense()
    
    print(f" • Final X_processed type: {type(X_processed)}")
    print(f" • Final X_processed shape: {X_processed.shape}")
    
    # Additional safety check for sparse columns
    if isinstance(X_processed, pd.DataFrame):
        for col in X_processed.columns:
            if hasattr(X_processed[col], 'sparse'):
                print(f" • Converting sparse column {col} to dense")
                X_processed[col] = X_processed[col].sparse.to_dense()

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
        print(f" • ERROR computing target statistics: {target_error}")
        meta_features.update({
            "target_mean": 0.0,
            "target_std": 0.0,
            "target_skew": 0.0,
            "target_kurtosis": 0.0,
        })

    # Work with processed data (should all be numeric now)
    print(" • Step: Computing feature statistics...")
    X_num = X_processed
    print(f" • X_num type: {type(X_num)}, shape: {X_num.shape}")
    
    # Check if any columns are still sparse
    if isinstance(X_num, pd.DataFrame):
        sparse_columns = []
        for col in X_num.columns:
            if hasattr(X_num[col], 'sparse') or str(X_num[col].dtype).startswith('Sparse'):
                sparse_columns.append(col)
        if sparse_columns:
            print(f" • Found sparse columns: {sparse_columns}")
            for col in sparse_columns:
                X_num[col] = X_num[col].sparse.to_dense() if hasattr(X_num[col], 'sparse') else pd.array(X_num[col], dtype=float)
    
    if X_num.shape[1] == 0: 
        print(" • No features after preprocessing, using default values")
        # no numeric features → fill zeros
        meta_features.update({
            "mean_feature_skew": 0.0,
            "mean_feature_kurtosis": 0.0,
            # "zero_var_pct": 1.0,
            "mean_abs_corr": 0.0,
            "max_abs_corr": 0.0,
        })
        
    else:
        try:
            print(" • Computing feature skew/kurtosis...")
            # Ensure all columns are dense before statistical operations
            if isinstance(X_num, pd.DataFrame):
                for col in X_num.columns:
                    if hasattr(X_num[col], 'sparse') or str(X_num[col].dtype).startswith('Sparse'):
                        print(f" • Converting remaining sparse column {col} to dense")
                        X_num[col] = X_num[col].sparse.to_dense() if hasattr(X_num[col], 'sparse') else pd.array(X_num[col].values, dtype=float)
            
            # 1) Per-feature mean skew/kurtosis
            # Compute skew/kurtosis across columns means (column-wise)
            means = X_num.mean(axis=0)
            
            # Additional safety check for means
            if hasattr(means, 'sparse') or (hasattr(means, 'dtype') and str(means.dtype).startswith('Sparse')):
                means = means.sparse.to_dense() if hasattr(means, 'sparse') else pd.Series(means.values, dtype=float)
            
            meta_features["mean_feature_skew"]     = float(skew(means))
            meta_features["mean_feature_kurtosis"] = float(kurtosis(means))
            print(f" • ✓ Feature statistics computed")

            # 2) Zero-variance percentage
            # zero_var = (X_num.var(axis=0) == 0).sum()
            # meta_features["zero_var_pct"] = float(zero_var / X_num.shape[1])

            # 3) Pairwise correlations (only if >1 numeric column)
            print(f" • Computing correlations for {X_num.shape[1]} features...")
            if X_num.shape[1] > 1:
                # Force conversion to float64 dense DataFrame for correlation computation
                X_corr = X_num.astype(float) if isinstance(X_num, pd.DataFrame) else pd.DataFrame(X_num, dtype=float)
                corr = X_corr.corr().abs()
                # take upper triangle without diagonal
                iu = np.triu_indices(corr.shape[0], k=1)
                tri_vals = corr.values[iu]
                meta_features["mean_abs_corr"] = float(np.nanmean(tri_vals))
                meta_features["max_abs_corr"]  = float(np.nanmax(tri_vals))
                print(f" • ✓ Correlations computed")
            else:
                meta_features["mean_abs_corr"] = 0.0
                meta_features["max_abs_corr"]  = 0.0
                print(" • Only 1 feature, correlations set to 0")
                
        except Exception as stat_error:
            print(f" • ERROR computing feature statistics: {stat_error}")
            print(f" • X_num dtypes: {X_num.dtypes if isinstance(X_num, pd.DataFrame) else 'Not DataFrame'}")
            # Fallback to default values
            meta_features.update({
                "mean_feature_skew": 0.0,
                "mean_feature_kurtosis": 0.0,
                "mean_abs_corr": 0.0,
                "max_abs_corr": 0.0,
            })

    # 4) Probing Features
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
    
    # Ensure we work with dense arrays for meta-feature computation
    X_train_processed = _to_dense_array(X_train)
    X_test_processed = _to_dense_array(X_test)

    print(" X_train Type:", type(X_train_processed), "Shape:", X_train_processed.shape)
    print(" X_test Type:", type(X_test_processed), "Shape:", X_test_processed.shape)

    print(" =============== meta features evaluation =============== ")
    sample_fraction = 0.01  # 1% sample for probing

    print("Evaluating dummy regressor")
    # 4.1) Mean value predictor performance (baseline)
    dummy_mean = DummyRegressor(strategy='mean')
    dummy_mean.fit(X_train_processed, y_train)
    y_pred_mean = dummy_mean.predict(X_test_processed)
    
    meta_features['mean_predictor_r2'] = r2_score(y_test, y_pred_mean)
    
    print("evaluating decision stump ")
    # 4.2) Decision stump performance (depth=1 tree)
    stump = DecisionTreeRegressor(max_depth=1, random_state=42)
    try:
        stump.fit(X_train_processed, y_train)
        y_pred_stump = stump.predict(X_test_processed)

        meta_features['decision_stump_r2'] = r2_score(y_test, y_pred_stump)
    except Exception as e:
        print(f"Error evaluating decision stump: {e}")
        # Use mean predictor performance as fallback (decision stump should at least match dummy)
        meta_features['decision_stump_r2'] = meta_features['mean_predictor_r2']
    # Relative improvement over mean predictor
    # meta_features['stump_vs_mean_r2_ratio'] = (
    #     meta_features['decision_stump_r2'] / max(meta_features['mean_predictor_r2'], 1e-10)
    # )
    
    print("evaluating simple rule model ")
    # 4.3) Simple rule model performance (linear regression)
    simple_rule = LinearRegression()
    simple_rule.fit(X_train_processed, y_train)
    y_pred_rule = simple_rule.predict(X_test_processed)

    meta_features['simple_rule_r2'] = r2_score(y_test, y_pred_rule)
        
    # Relative improvement over mean predictor
    # meta_features['rule_vs_mean_r2_ratio'] = (
    #     meta_features['simple_rule_r2'] / max(meta_features['mean_predictor_r2'], 1e-10)
    # )
    
    print("evaluating algorithms on 1% sample ")
    # 4.4) Performance of algorithms on 1% of data
    # if X_train_processed.shape[0] > 100:  # Only if we have enough data
    
    #TODO
    # Sample 1% of training data
    sample_size = max(int(X_train_processed.shape[0] * sample_fraction), 10)
    sample_indices = np.random.choice(X_train_processed.shape[0], sample_size, replace=True)
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
            # meta_features[f'{algo_name}_vs_mean_r2_ratio'] = (
            #     algo_r2 / max(meta_features['mean_predictor_r2'], 1e-10)
            # )
            # meta_features[f'{algo_name}_vs_stump_r2_ratio'] = (
            #     algo_r2 / max(meta_features['decision_stump_r2'], 1e-10)
            # )
            
        except Exception as e:
            print(f"========= Error evaluating {algo_name} on 1% data: {e} ========== ")
            meta_features[f'{algo_name}_1pct_r2'] = -1.0
            meta_features[f'{algo_name}_1pct_rmse'] = float('inf')
            # meta_features[f'{algo_name}_vs_mean_r2_ratio'] = 0.0
            # meta_features[f'{algo_name}_vs_stump_r2_ratio'] = 0.0

    # 4.5) Additional derived meta-features
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
