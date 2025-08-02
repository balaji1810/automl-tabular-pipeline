from typing import Any
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import torch
from automl.pre_processor import build_preprocessor

class MetaFeatureExtractor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def extract_meta_features(self, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, int], pd.DataFrame]:
        """Compute meta-features for regression datasets."""
        # Basic sizes
        n, d = X.shape
        meta_features = {
            "n_samples": n,
            "n_features": d,
            "log_n_samples": np.log(n + 1) if n > 0 else 0.0, #not needed because we use n_samples directly
            "log_n_features": np.log(d + 1) if d > 0 else 0.0, #not needed because we use n_features directly
            "feature_ratio": d / n if n else 0.0,
            "target_mean": float(y.mean()) if len(y) else 0.0,
            "target_std": float(y.std()) if len(y) else 0.0,
            "target_skew": float(skew(y)) if len(y) else 0.0,
            "target_kurtosis": float(kurtosis(y)) if len(y) else 0.0,
        }

        # Restrict to numeric cols
        X_num = X.select_dtypes(include=[np.number])
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

        preprocessor = build_preprocessor(X)
        X = preprocessor.fit_transform(X)  # type: ignore

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.random_state)

        # Preprocess (impute + encode) X_train and X_test
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        sample_fraction = 0.01  # 1% sample for probing
        # 4.1) Mean value predictor performance (baseline)
        dummy_mean = DummyRegressor(strategy='mean')
        dummy_mean.fit(X_train, y_train)
        y_pred_mean = dummy_mean.predict(X_test)

        meta_features['mean_predictor_r2'] = r2_score(y_test, y_pred_mean)

        # 4.2) Decision stump performance (depth=1 tree)
        stump = DecisionTreeRegressor(max_depth=1, random_state=42)
        stump.fit(X_train, y_train)
        y_pred_stump = stump.predict(X_test)

        meta_features['decision_stump_r2'] = r2_score(y_test, y_pred_stump)

        # Relative improvement over mean predictor
        meta_features['stump_vs_mean_r2_ratio'] = (
            meta_features['decision_stump_r2'] / max(meta_features['mean_predictor_r2'], 1e-10)
        )

        # 4.3) Simple rule model performance (linear regression)
        simple_rule = LinearRegression()
        simple_rule.fit(np.array(X_train), np.array(y_train))
        y_pred_rule = simple_rule.predict(np.array(X_test))

        meta_features['simple_rule_r2'] = r2_score(y_test, y_pred_rule)

        # Relative improvement over mean predictor
        meta_features['rule_vs_mean_r2_ratio'] = (
            meta_features['simple_rule_r2'] / max(meta_features['mean_predictor_r2'], 1e-10)
        )

        # 4.4) Performance of algorithms on 1% of data
        if len(X_train) > 100:  # Only if we have enough data
            # Sample 1% of training data
            sample_size = max(int(len(X_train) * sample_fraction), 10)
            sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train.iloc[sample_indices]
            y_sample = y_train.iloc[sample_indices]

            for algo_name, algorithm in algorithms_dict.items():
                try:
                    # Clone the algorithm to avoid fitting issues
                    from sklearn.base import clone
                    algo_clone = clone(algorithm)

                    # Fit on 1% sample
                    algo_clone.fit(X_sample, y_sample)

                    # Predict on test set
                    y_pred_algo = algo_clone.predict(X_test)

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
        meta_features['baseline_difficulty'] = 1 - meta_features['mean_predictor_r2']
        # meta_features['linear_separability'] = meta_features['simple_rule_r2']
        meta_features['tree_advantage'] = (
            meta_features['decision_stump_r2'] - meta_features['simple_rule_r2']
        )

        # 5) Algorithm suitability indicators
        meta_features['tabpfn_suitable'] = 1 if (n < 1000 and d < 100 and n*d < 10000) else 0
        meta_features['svm_suitable'] = 1 if (100 <= n <= 10000 and d <= 1000) else 0
        meta_features['linear_favorable'] = 1 if (meta_features['simple_rule_r2'] > 0.7 and 
                                                meta_features['target_skew'] < 2) else 0

        return meta_features, X
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the meta-feature extractor on the provided dataset.
        This is a no-op since we don't need to learn anything from the data.
        """
        meta_features, X = self.extract_meta_features(X, y)
        self.meta_features_ = meta_features
        self.X_processed = X
        meta_model = torch.load("meta_model.pth")
        self.best_algorithms = meta_model.predict(meta_features)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted preprocessor.
        Returns the processed DataFrame.
        """
        if not hasattr(self, "X_processed"):
            raise RuntimeError("The model has not been fitted yet.")

        # Here you would apply any transformations to X based on the fitted model
        # For now, we'll just return the processed DataFrame
        return self.X_processed
