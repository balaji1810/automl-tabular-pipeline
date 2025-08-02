from math import floor
from typing import Literal
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

# Suppress specific sklearn warnings about binning
warnings.filterwarnings("ignore", message="Bins whose width are too small*", category=UserWarning)

class FeatureSelector:
    #TODO add params [bool] for alg specific feature engineering:
    def __init__(self, max_features: float = 0.75, select_method: Literal["permutation", "tree"] = "permutation", 
                 add_polynomial_features_xgb: bool = False,   
                 add_binning_features_xgb: bool = False, 
                 add_statistical_features_xgb: bool = False,
                 add_quantile_binning_lgb: bool = False,
                 add_categorical_encodings_lgb: bool = False,
                 add_aggregation_features_lgb: bool = False, 
                 algorithm: str = "xgboost", seed: int = 20):
        self.max_features = max_features
        self.select_method = select_method
        self.algorithm = algorithm  # New parameter for algorithm-specific engineering
        self.seed = seed
        self.X_perm_columns : pd.Index = pd.Index([])
        self.X_tree_columns : pd.Index = pd.Index([])
        self.selected_columns : pd.Index = pd.Index([])
        
        # For algorithm-specific feature engineering
        self.poly_features = None
        self.binning_discretizer = None
        self.target_encoders = {}
        self.frequency_encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self._binning_valid_cols = []  # Store valid columns for binning

        # Bool for algorithm-specific feature engineering
        self.add_polynomial_features_xgb = add_polynomial_features_xgb
        self.add_binning_features_xgb = add_binning_features_xgb
        self.add_statistical_features_xgb = add_statistical_features_xgb
        self.add_quantile_binning_lgb = add_quantile_binning_lgb
        self.add_categorical_encodings_lgb = add_categorical_encodings_lgb
        self.add_aggregation_features_lgb =add_aggregation_features_lgb

    def _add_polynomial_features_xgb(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features for XGBoost"""
        X_poly = X.copy()
        
        if self.poly_features is None:
            # Only use top numerical features to avoid explosion
            top_numerical = self.numerical_cols[:min(3, len(self.numerical_cols))]
            if len(top_numerical) >= 2:
                self.poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                poly_data = self.poly_features.fit_transform(X[top_numerical])
                poly_feature_names = self.poly_features.get_feature_names_out(top_numerical)
                
                # Add polynomial features
                poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=X.index)
                # Only keep interaction terms (not original features)
                interaction_cols = [col for col in poly_df.columns if ' ' in col]
                X_poly = pd.concat([X_poly, poly_df[interaction_cols]], axis=1)
        else:
            # Transform using fitted polynomial features
            top_numerical = self.numerical_cols[:min(3, len(self.numerical_cols))]
            if len(top_numerical) >= 2:
                poly_data = self.poly_features.transform(X[top_numerical])
                poly_feature_names = self.poly_features.get_feature_names_out(top_numerical)
                poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=X.index)
                interaction_cols = [col for col in poly_df.columns if ' ' in col]
                X_poly = pd.concat([X_poly, poly_df[interaction_cols]], axis=1)
        
        return X_poly
    
    def _add_binning_features_xgb(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add binned features for XGBoost"""
        X_binned = X.copy()
        
        if self.binning_discretizer is None and len(self.numerical_cols) > 0:
            # Filter out columns with very small variance to avoid binning warnings
            valid_cols = []
            for col in self.numerical_cols:
                col_std = X[col].std()
                col_range = X[col].max() - X[col].min()
                # Only include columns with sufficient variance and range
                if col_std > 1e-6 and col_range > 1e-6 and X[col].nunique() > 5:
                    valid_cols.append(col)
            
            if len(valid_cols) > 0:
                # Store valid columns for later use in transform
                self._binning_valid_cols = valid_cols
                # Use fewer bins for better stability
                n_bins = min(5, max(3, len(valid_cols)))
                self.binning_discretizer = KBinsDiscretizer(
                    n_bins=n_bins, 
                    encode='ordinal', 
                    strategy='quantile',
                    subsample=None  # Use all data for stable quantiles
                )
                try:
                    binned_data = self.binning_discretizer.fit_transform(X[valid_cols])
                    binned_df = pd.DataFrame(binned_data, 
                                           columns=[f"{col}_binned" for col in valid_cols], 
                                           index=X.index)
                    X_binned = pd.concat([X_binned, binned_df], axis=1)
                except Exception as e:
                    # If binning fails, skip this feature engineering step
                    pass
        elif self.binning_discretizer is not None and len(self.numerical_cols) > 0:
            # Get the original valid columns used during fit
            # Store valid columns during fit for later use
            if hasattr(self, '_binning_valid_cols'):
                valid_cols = self._binning_valid_cols
            else:
                # Fallback: try to determine valid cols from current data
                valid_cols = []
                for col in self.numerical_cols:
                    col_std = X[col].std()
                    col_range = X[col].max() - X[col].min()
                    if col_std > 1e-6 and col_range > 1e-6 and X[col].nunique() > 5:
                        valid_cols.append(col)
                        
            valid_cols = [col for col in valid_cols if col in X.columns]
            if len(valid_cols) > 0:
                try:
                    binned_data = self.binning_discretizer.transform(X[valid_cols])
                    binned_df = pd.DataFrame(binned_data, 
                                           columns=[f"{col}_binned" for col in valid_cols], 
                                           index=X.index)
                    X_binned = pd.concat([X_binned, binned_df], axis=1)
                except Exception as e:
                    # If transform fails, skip this feature engineering step
                    pass
        
        return X_binned
    
    def _add_statistical_features_xgb(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features for XGBoost"""
        X_stats = X.copy()
        
        if len(self.numerical_cols) >= 2:
            # Add statistical aggregations across numerical features
            numerical_data = X[self.numerical_cols]
            
            # Row-wise statistics
            X_stats['num_mean'] = numerical_data.mean(axis=1)
            X_stats['num_std'] = numerical_data.std(axis=1)
            X_stats['num_min'] = numerical_data.min(axis=1)
            X_stats['num_max'] = numerical_data.max(axis=1)
            X_stats['num_range'] = X_stats['num_max'] - X_stats['num_min']
            
            # Ratios of top features
            if len(self.numerical_cols) >= 2:
                X_stats[f'{self.numerical_cols[0]}_div_{self.numerical_cols[1]}'] = (
                    X[self.numerical_cols[0]] / (X[self.numerical_cols[1]] + 1e-8)
                )
        
        return X_stats
    
    def _add_quantile_binning_lgb(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add different binning strategies for LightGBM"""
        X_binned = X.copy()
        
        if len(self.numerical_cols) > 0:
            # Filter valid columns for binning
            valid_cols = []
            for col in self.numerical_cols[:3]:  # Limit to avoid too many features
                col_std = X[col].std()
                col_range = X[col].max() - X[col].min()
                # Only include columns with sufficient variance and range
                if col_std > 1e-6 and col_range > 1e-6 and X[col].nunique() > 4:
                    valid_cols.append(col)
            
            # Quantile-based binning (different from XGBoost uniform binning)
            for col in valid_cols:
                try:
                    # Create quantile bins with duplicate handling
                    _, bin_edges = pd.qcut(X[col], q=4, retbins=True, duplicates='drop')
                    if len(bin_edges) > 2:  # Ensure we have valid bins
                        X_binned[f"{col}_quartile"] = pd.cut(X[col], bins=bin_edges, labels=False, include_lowest=True)
                    
                    # Create percentile-based bins
                    percentiles = [0, 0.1, 0.5, 0.9, 1.0]
                    _, perc_edges = pd.qcut(X[col], q=percentiles, retbins=True, duplicates='drop')
                    if len(perc_edges) > 2:  # Ensure we have valid bins
                        X_binned[f"{col}_percentile"] = pd.cut(X[col], bins=perc_edges, labels=False, include_lowest=True)
                except Exception as e:
                    # If binning fails for this column, skip it
                    continue
        
        return X_binned
    
    def _add_categorical_encodings_lgb(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Add categorical encodings for LightGBM"""
        X_encoded = X.copy()
        
        if len(self.categorical_cols) > 0:
            for col in self.categorical_cols:
                # Target encoding for categorical variables
                if y is not None and col not in self.target_encoders:
                    # Simple target encoding with regularization
                    target_mean = y.mean()
                    col_target_mean = X.groupby(col)[y.name].mean() if hasattr(y, 'name') else X.groupby(col).apply(lambda x: y.iloc[x.index].mean())
                    col_counts = X[col].value_counts()
                    
                    # Regularized target encoding
                    alpha = 10  # Regularization parameter
                    regularized_encoding = (col_target_mean * col_counts + target_mean * alpha) / (col_counts + alpha)
                    self.target_encoders[col] = regularized_encoding.to_dict()
                
                if col in self.target_encoders:
                    X_encoded[f"{col}_target_enc"] = X[col].map(self.target_encoders[col]).fillna(y.mean() if y is not None else 0)
                
                # Frequency encoding
                if col not in self.frequency_encoders:
                    freq_encoding = X[col].value_counts(normalize=True).to_dict()
                    self.frequency_encoders[col] = freq_encoding
                
                X_encoded[f"{col}_freq_enc"] = X[col].map(self.frequency_encoders[col]).fillna(0)
        
        return X_encoded
    
    def _add_aggregation_features_lgb(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add aggregation features for LightGBM"""
        X_agg = X.copy()
        
        if len(self.numerical_cols) >= 2 and len(self.categorical_cols) >= 1:
            # Group-by aggregations
            main_cat = self.categorical_cols[0]
            
            for num_col in self.numerical_cols[:2]:  # Limit to avoid explosion
                # Group statistics by main categorical variable
                group_stats = X.groupby(main_cat)[num_col].agg(['mean', 'std', 'min', 'max'])
                
                # Map back to original dataframe
                X_agg[f"{num_col}_groupby_{main_cat}_mean"] = X[main_cat].map(group_stats['mean'])
                X_agg[f"{num_col}_groupby_{main_cat}_std"] = X[main_cat].map(group_stats['std']).fillna(0)
                
                # Difference from group mean
                X_agg[f"{num_col}_diff_from_group_mean"] = X[num_col] - X_agg[f"{num_col}_groupby_{main_cat}_mean"]
        
        return X_agg

    def _apply_algorithm_specific_features(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Apply algorithm-specific feature engineering"""
        X_engineered = X.copy()
        
        if self.algorithm == "xgboost":
            # XGBoost-specific features
            if self.add_polynomial_features_xgb:
                X_engineered = self._add_polynomial_features_xgb(X_engineered)
            if self.add_binning_features_xgb:
                X_engineered = self._add_binning_features_xgb(X_engineered)
            if self.add_statistical_features_xgb:
                X_engineered = self._add_statistical_features_xgb(X_engineered)
        elif self.algorithm == "lightgbm":
            # LightGBM-specific features
            if self.add_quantile_binning_lgb:
                X_engineered = self._add_quantile_binning_lgb(X_engineered)
            if self.add_categorical_encodings_lgb:
                X_engineered = self._add_categorical_encodings_lgb(X_engineered, y)
            if self.add_aggregation_features_lgb:
                X_engineered = self._add_aggregation_features_lgb(X_engineered)
        
        return X_engineered

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit the feature selector with algorithm-specific feature engineering and selection.
        """
        # Identify column types for feature engineering
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Apply algorithm-specific feature engineering
        X_engineered = self._apply_algorithm_specific_features(X, y)
        
        # Split for feature selection
        X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=self.seed)
        
        # Select model for importance calculation
        if self.algorithm == "lightgbm" and LGBMRegressor is not None:
            reg1 = LGBMRegressor(random_state=self.seed, n_jobs=-1, verbose=-1)
        else:
            reg1 = XGBRegressor(random_state=self.seed, n_jobs=-1)
        
        reg1.fit(X_train, y_train)
        num_features = floor(len(X_train.columns) * self.max_features)

        # Algorithm-specific feature selection
        if self.algorithm == "xgboost" or self.select_method == "tree":
            tree_importance_sorted_idx = np.argsort(reg1.feature_importances_)
            self.X_tree_columns = X_train.columns[tree_importance_sorted_idx][-num_features:]
            self.selected_columns = self.X_tree_columns
        else:  # Use permutation importance for LightGBM or when explicitly requested
            result = permutation_importance(reg1, X_test, y_test, n_repeats=5, random_state=self.seed, n_jobs=-1)
            perm_sorted_idx = result["importances_mean"].argsort()
            self.X_perm_columns = X_train.columns[perm_sorted_idx][-num_features:]
            self.selected_columns = self.X_perm_columns
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame using algorithm-specific feature engineering and selection.
        """
        # Apply algorithm-specific feature engineering
        X_engineered = self._apply_algorithm_specific_features(X)
        
        # Apply feature selection using stored selected columns
        available_features = [col for col in self.selected_columns if col in X_engineered.columns]
        return X_engineered[available_features] if available_features else X_engineered
    
    def set_params(self, **params):
        """
        Set parameters for the feature selector.
        """
        self.X_perm_columns = pd.Index([])
        self.X_tree_columns = pd.Index([])
        self.selected_columns = pd.Index([])
        self.poly_features = None
        self.binning_discretizer = None
        for key, value in params.items():
            setattr(self, key, value)
        return self
    