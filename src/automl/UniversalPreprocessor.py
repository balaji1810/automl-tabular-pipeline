"""
Universal Preprocessing Methods for Both Tree-Based and Other ML Algorithms

This module provides preprocessing strategies that work well across different algorithm types:
- Tree-based: RandomForest, XGBoost, LightGBM, DecisionTree
- Distance-based: SVR, KNN
- Neural: MLPRegressor
- Linear: LinearRegression, Ridge, Lasso
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer,
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')


def detect_column_types_enhanced(X: pd.DataFrame) -> dict:
    """
    Enhanced column type detection with more granular categorization.
    """
    column_info = {
        'numeric_continuous': [],
        'numeric_discrete': [],
        'categorical_nominal': [],
        'categorical_ordinal': [],
        'datetime': [],
        'text': [],
        'binary': [],
        'high_cardinality': []
    }
    
    for col in X.columns:
        col_data = X[col]
        
        # DateTime detection
        if pd.api.types.is_datetime64_any_dtype(col_data):
            column_info['datetime'].append(col)
        
        # Numeric columns
        elif pd.api.types.is_numeric_dtype(col_data):
            unique_vals = col_data.nunique()
            total_vals = len(col_data)
            
            # Binary numeric (0/1, True/False)
            if unique_vals <= 2:
                column_info['binary'].append(col)
            
            # Discrete numeric (integers with low cardinality)
            elif col_data.dtype in ['int64', 'int32'] and unique_vals < 20:
                column_info['numeric_discrete'].append(col)
            
            # Continuous numeric
            else:
                column_info['numeric_continuous'].append(col)
        
        # Categorical columns
        else:
            unique_vals = col_data.nunique()
            
            # High cardinality (might need special handling)
            if unique_vals > 50:
                column_info['high_cardinality'].append(col)
            
            # Binary categorical
            elif unique_vals <= 2:
                column_info['binary'].append(col)
            
            # Regular categorical (assume nominal for now)
            else:
                column_info['categorical_nominal'].append(col)
    
    return column_info


class UniversalImputer(BaseEstimator, TransformerMixin):
    """
    Intelligent imputation that works well for all algorithm types.
    """
    
    def __init__(self, numeric_strategy='median', categorical_strategy='most_frequent', 
                 use_knn_for_numeric=False, knn_neighbors=5):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.use_knn_for_numeric = use_knn_for_numeric
        self.knn_neighbors = knn_neighbors
        self.imputers_ = {}
        self._output_transform = None
    
    def set_output(self, *, transform=None):
        """Set output container - compatibility with sklearn's ColumnTransformer."""
        self._output_transform = transform
        # Propagate to underlying sklearn transformers when they're created
        for imputer in self.imputers_.values():
            if hasattr(imputer, 'set_output'):
                imputer.set_output(transform=transform)
        return self
    
    def fit(self, X, y=None):
        column_info = detect_column_types_enhanced(X)
        
        # Numeric imputation
        numeric_cols = (column_info['numeric_continuous'] + 
                       column_info['numeric_discrete'])
        
        if numeric_cols:
            if self.use_knn_for_numeric and len(numeric_cols) > 1:
                # KNN imputation for numeric (good for preserving relationships)
                self.imputers_['numeric'] = KNNImputer(n_neighbors=self.knn_neighbors)
            else:
                # Simple imputation (more robust)
                self.imputers_['numeric'] = SimpleImputer(strategy=self.numeric_strategy)
            
            # Set output format if specified
            if (hasattr(self, '_output_transform') and self._output_transform and 
                hasattr(self.imputers_['numeric'], 'set_output')):
                if self._output_transform == 'pandas':
                    self.imputers_['numeric'].set_output(transform='pandas')
                elif self._output_transform == 'default':
                    self.imputers_['numeric'].set_output(transform='default')
            
            self.imputers_['numeric'].fit(X[numeric_cols])
            self.numeric_cols_ = numeric_cols
        
        # Categorical imputation
        categorical_cols = (column_info['categorical_nominal'] + 
                          column_info['categorical_ordinal'] +
                          column_info['binary'] +
                          column_info['high_cardinality'])
        
        if categorical_cols:
            self.imputers_['categorical'] = SimpleImputer(strategy=self.categorical_strategy)
            
            # Set output format if specified
            if (hasattr(self, '_output_transform') and self._output_transform and 
                hasattr(self.imputers_['categorical'], 'set_output')):
                if self._output_transform == 'pandas':
                    self.imputers_['categorical'].set_output(transform='pandas')
                elif self._output_transform == 'default':
                    self.imputers_['categorical'].set_output(transform='default')
            
            self.imputers_['categorical'].fit(X[categorical_cols])
            self.categorical_cols_ = categorical_cols
        
        return self
    
    def transform(self, X):
        X_imputed = X.copy()
        
        # Impute numeric columns
        if hasattr(self, 'numeric_cols_') and 'numeric' in self.imputers_:
            imputed_numeric = self.imputers_['numeric'].transform(X[self.numeric_cols_])
            
            # Ensure DataFrame output
            if not isinstance(imputed_numeric, pd.DataFrame):
                imputed_numeric = pd.DataFrame(
                    imputed_numeric, 
                    index=X.index, 
                    columns=self.numeric_cols_
                )
            
            X_imputed[self.numeric_cols_] = imputed_numeric
        
        # Impute categorical columns
        if hasattr(self, 'categorical_cols_') and 'categorical' in self.imputers_:
            imputed_categorical = self.imputers_['categorical'].transform(X[self.categorical_cols_])
            
            # Ensure DataFrame output
            # if not isinstance(imputed_categorical, pd.DataFrame):
            #     imputed_categorical = pd.DataFrame(
            #         imputed_categorical, 
            #         index=X.index, 
            #         columns=self.categorical_cols_
            #     )
            
            X_imputed[self.categorical_cols_] = imputed_categorical
        
        return X_imputed


class UniversalScaler(BaseEstimator, TransformerMixin):
    """
    Scaling strategy that benefits both tree-based and other algorithms.
    """
    
    def __init__(self, method='robust', handle_outliers=True):
        self.method = method
        self.handle_outliers = handle_outliers
        self.scalers_ = {}
        self.outlier_info_ = {}
        self._output_transform = None
    
    def set_output(self, *, transform=None):
        """Set output container - compatibility with sklearn's ColumnTransformer."""
        self._output_transform = transform
        # Propagate to underlying sklearn transformers
        for scaler in self.scalers_.values():
            if hasattr(scaler, 'set_output'):
                if transform == 'pandas':
                    scaler.set_output(transform='pandas')
                elif transform == 'default':
                    scaler.set_output(transform='default')
        return self
    
    def fit(self, X, y=None):
        column_info = detect_column_types_enhanced(X)
        numeric_cols = (column_info['numeric_continuous'] + 
                       column_info['numeric_discrete'])
        
        if not numeric_cols:
            return self
        
        self.numeric_cols_ = numeric_cols
        
        for col in numeric_cols:
            col_data = X[col].dropna()
            
            # Detect outliers
            if self.handle_outliers:
                self.outlier_info_[col] = self._detect_outliers(col_data)
            
            # Choose appropriate scaler
            if self.method == 'robust':
                # Robust to outliers, works well for both tree and non-tree
                self.scalers_[col] = RobustScaler()
            elif self.method == 'standard':
                # Good for linear models, neural networks
                self.scalers_[col] = StandardScaler()
            elif self.method == 'minmax':
                # Good for neural networks, bounded features
                self.scalers_[col] = MinMaxScaler()
            elif self.method == 'quantile':
                # Makes features more uniform, good for all algorithms
                self.scalers_[col] = QuantileTransformer(n_quantiles=100, random_state=42)
            else:
                # Default to robust
                self.scalers_[col] = RobustScaler()
            
            # Fit scaler
            self.scalers_[col].fit(col_data.values.reshape(-1, 1))
            
            # Set output format if specified
            if (hasattr(self, '_output_transform') and self._output_transform and 
                hasattr(self.scalers_[col], 'set_output')):
                if self._output_transform == 'pandas':
                    self.scalers_[col].set_output(transform='pandas')
                elif self._output_transform == 'default':
                    self.scalers_[col].set_output(transform='default')
        
        return self
    
    def transform(self, X):
        if not hasattr(self, 'numeric_cols_'):
            return X
        
        X_scaled = X.copy()
        
        for col in self.numeric_cols_:
            if col in self.scalers_:
                # X_scaled[col] = self.scalers_[col].transform(X[col])
                X_scaled[col] = self.scalers_[col].transform(X[col].values.reshape(-1, 1)).flatten()
        
        return X_scaled
    
    def _detect_outliers(self, data):
        """Detect outliers using IQR method."""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        outlier_ratio = outliers / len(data)
        
        return {
            'count': outliers,
            'ratio': outlier_ratio,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }


class UniversalEncoder(BaseEstimator, TransformerMixin):
    """
    Categorical encoding that works well for all algorithm types.
    """
    
    def __init__(self, high_cardinality_threshold=50, encoding_strategy='auto'):
        self.high_cardinality_threshold = high_cardinality_threshold
        self.encoding_strategy = encoding_strategy
        self.encoders_ = {}
        self.encoding_map_ = {}
        self._output_transform = None
    
    def set_output(self, *, transform=None):
        """Set output container - compatibility with sklearn's ColumnTransformer."""
        self._output_transform = transform
        # Propagate to underlying sklearn transformers
        for encoder in self.encoders_.values():
            if hasattr(encoder, 'set_output'):
                if transform == 'pandas':
                    encoder.set_output(transform='pandas')
                elif transform == 'default':
                    encoder.set_output(transform='default')
        return self
    
    def fit(self, X, y=None):
        column_info = detect_column_types_enhanced(X)
        
        # Get all categorical columns
        categorical_cols = (column_info['categorical_nominal'] + 
                          column_info['categorical_ordinal'] +
                          column_info['binary'] +
                          column_info['high_cardinality'])
        
        if not categorical_cols:
            return self
        
        self.categorical_cols_ = categorical_cols
        
        for col in categorical_cols:
            cardinality = X[col].nunique()
            
            # Decide encoding strategy
            if self.encoding_strategy == 'auto':
                if cardinality <= 2:
                    # Binary: use label encoding (0/1)
                    strategy = 'label'
                elif cardinality <= 10:
                    # Low cardinality: use one-hot (good for all algorithms)
                    strategy = 'onehot'
                elif cardinality <= self.high_cardinality_threshold:
                    # Medium cardinality: use ordinal (tree-friendly, space-efficient)
                    strategy = 'ordinal'
                else:
                    # High cardinality: use target encoding or frequency encoding
                    strategy = 'frequency'  # Could also be 'target' if y is provided
            else:
                strategy = self.encoding_strategy
            
            self.encoding_map_[col] = strategy
            
            # Create appropriate encoder
            if strategy == 'onehot':
                self.encoders_[col] = OneHotEncoder(
                    sparse_output=False, 
                    handle_unknown='ignore',
                    drop='if_binary'  # Drop one category for binary features
                )
                # Set output format if specified
                if (hasattr(self, '_output_transform') and self._output_transform and 
                    hasattr(self.encoders_[col], 'set_output')):
                    if self._output_transform == 'pandas':
                        self.encoders_[col].set_output(transform='pandas')
                    elif self._output_transform == 'default':
                        self.encoders_[col].set_output(transform='default')
                
                self.encoders_[col].fit(X[[col]])
                
            elif strategy == 'ordinal':
                self.encoders_[col] = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                # Set output format if specified
                if (hasattr(self, '_output_transform') and self._output_transform and 
                    hasattr(self.encoders_[col], 'set_output')):
                    if self._output_transform == 'pandas':
                        self.encoders_[col].set_output(transform='pandas')
                    elif self._output_transform == 'default':
                        self.encoders_[col].set_output(transform='default')
                
                self.encoders_[col].fit(X[[col]])
                
            elif strategy == 'label':
                self.encoders_[col] = LabelEncoder()
                self.encoders_[col].fit(X[col].astype(str))
                
            elif strategy == 'frequency':
                # Frequency encoding
                freq_map = X[col].value_counts(normalize=True).to_dict()
                self.encoders_[col] = freq_map
        
        return self
    
    def transform(self, X):
        if not hasattr(self, 'categorical_cols_'):
            return X
        
        X_encoded = X.copy()
        new_columns = []
        columns_to_drop = []
        
        for col in self.categorical_cols_:
            if col not in self.encoders_:
                continue
                
            strategy = self.encoding_map_[col]
            
            if strategy == 'onehot':
                # One-hot encoding
                encoded = self.encoders_[col].transform(X[[col]])
                feature_names = self.encoders_[col].get_feature_names_out([col])
                
                # Add new columns
                for i, feature_name in enumerate(feature_names):
                    X_encoded[feature_name] = encoded[:, i]
                    new_columns.append(feature_name)
                
                columns_to_drop.append(col)
                
            elif strategy in ['ordinal', 'label']:
                # Ordinal or label encoding
                if strategy == 'ordinal':
                    # X_encoded[col] = self.encoders_[col].transform(X[col])
                    X_encoded[col] = self.encoders_[col].transform(X[[col]]).flatten()
                else:  # label
                    # X_encoded[col] = self.encoders_[col].transform(X[col])
                    X_encoded[col] = self.encoders_[col].transform(X[col].astype(str))
                    
            elif strategy == 'frequency':
                # Frequency encoding
                freq_map = self.encoders_[col]
                X_encoded[col] = X[col].map(freq_map).fillna(0)  # Unknown categories get 0 frequency
        
        # Drop original columns that were one-hot encoded
        if columns_to_drop:
            X_encoded = X_encoded.drop(columns=columns_to_drop)
        
        return X_encoded


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering that benefits all algorithm types.
    """
    
    def __init__(self, create_interactions=True, create_polynomials=False, 
                 polynomial_degree=2, max_interaction_features=50):
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.polynomial_degree = polynomial_degree
        self.max_interaction_features = max_interaction_features
        self.interaction_features_ = []
    
    def set_output(self, *, transform=None):
        """Set output container - compatibility with sklearn's ColumnTransformer."""
        return self
    
    def fit(self, X, y=None):
        column_info = detect_column_types_enhanced(X)
        numeric_cols = (column_info['numeric_continuous'] + 
                       column_info['numeric_discrete'])
        
        self.numeric_cols_ = numeric_cols[:10]  # Limit to prevent explosion
        
        if self.create_interactions and len(self.numeric_cols_) > 1:
            # Create pairwise interactions for top features by variance
            if len(self.numeric_cols_) > 5:
                # Select top features by variance
                variances = X[self.numeric_cols_].var().sort_values(ascending=False)
                top_features = variances.head(5).index.tolist()
            else:
                top_features = self.numeric_cols_
            
            # Create interaction pairs
            for i, col1 in enumerate(top_features):
                for col2 in top_features[i+1:]:
                    interaction_name = f"{col1}_x_{col2}"
                    self.interaction_features_.append((col1, col2, interaction_name))
                    
                    if len(self.interaction_features_) >= self.max_interaction_features:
                        break
                if len(self.interaction_features_) >= self.max_interaction_features:
                    break
        
        return self
    
    def transform(self, X):
        X_engineered = X.copy()
        
        # Create interaction features
        for col1, col2, interaction_name in self.interaction_features_:
            if col1 in X.columns and col2 in X.columns:
                X_engineered[interaction_name] = X[col1] * X[col2]
        
        # Create polynomial features (be careful - can explode feature space)
        if self.create_polynomials and hasattr(self, 'numeric_cols_'):
            for col in self.numeric_cols_[:3]:  # Only for top 3 features
                if col in X.columns:
                    for degree in range(2, self.polynomial_degree + 1):
                        poly_name = f"{col}_power_{degree}"
                        X_engineered[poly_name] = X[col] ** degree
        
        return X_engineered


def build_universal_preprocessor(X: pd.DataFrame, preprocessing_strategy='balanced') -> ColumnTransformer:
    """
    Build a comprehensive preprocessor that works well for all algorithm types.
    Uses the custom classes for more intelligent preprocessing.
    
    Args:
        X: Input DataFrame
        preprocessing_strategy: 'conservative', 'balanced', or 'aggressive'
    
    Returns:
        ColumnTransformer with appropriate preprocessing steps
    """
    
    column_info = detect_column_types_enhanced(X)
    
    # Build transformers list using custom Universal classes
    transformers = []
    
    # Numeric columns - use UniversalScaler
    numeric_cols = (column_info['numeric_continuous'] + 
                   column_info['numeric_discrete'])
    
    if numeric_cols:
        if preprocessing_strategy == 'conservative':
            # Minimal preprocessing for tree-based algorithms
            numeric_steps = [
                ('imputer', UniversalImputer(
                    numeric_strategy='median', 
                    categorical_strategy='most_frequent',
                    use_knn_for_numeric=False
                )),
                ('scaler', UniversalScaler(method='robust'))  # Light scaling
            ]
            
        elif preprocessing_strategy == 'aggressive':
            # Heavy preprocessing for neural networks, SVR
            numeric_steps = [
                ('imputer', UniversalImputer(
                    numeric_strategy='median',
                    categorical_strategy='most_frequent', 
                    use_knn_for_numeric=True,  # KNN imputation for better relationships
                    knn_neighbors=5
                )),
                ('scaler', UniversalScaler(method='standard')),  # Strong scaling
                ('normalizer', UniversalScaler(method='quantile'))  # Additional normalization
            ]
            
        else:  # 'balanced'
            # Balanced preprocessing - good compromise for all algorithms
            numeric_steps = [
                ('imputer', UniversalImputer(
                    numeric_strategy='median',
                    categorical_strategy='most_frequent',
                    use_knn_for_numeric=False
                )),
                ('scaler', UniversalScaler(method='robust')),  # Robust scaling
                ('variance_filter', VarianceThreshold(threshold=0.001))  # Remove near-zero variance
            ]
        
        transformers.append(('numeric', Pipeline(numeric_steps), numeric_cols))
    
    # Categorical columns - use UniversalEncoder
    categorical_cols = (column_info['categorical_nominal'] + 
                       column_info['categorical_ordinal'] +
                       column_info['binary'])
    
    if categorical_cols:
        if preprocessing_strategy == 'conservative':
            # Tree-friendly encoding
            categorical_steps = [
                ('encoder', UniversalEncoder(
                    high_cardinality_threshold=50,
                    encoding_strategy='ordinal'  # Space-efficient for trees
                ))
            ]
            
        elif preprocessing_strategy == 'aggressive':
            # Neural network friendly encoding
            categorical_steps = [
                ('encoder', UniversalEncoder(
                    high_cardinality_threshold=20,  # Lower threshold for one-hot
                    encoding_strategy='onehot'  # Prevent ordinal assumptions
                ))
            ]
            
        else:  # 'balanced'
            # Automatic strategy selection
            categorical_steps = [
                ('encoder', UniversalEncoder(
                    high_cardinality_threshold=50,
                    encoding_strategy='auto'  # Let encoder decide
                ))
            ]
        
        transformers.append(('categorical', Pipeline(categorical_steps), categorical_cols))
    
    # High cardinality columns - special handling with UniversalEncoder
    high_card_cols = column_info['high_cardinality']
    if high_card_cols:
        high_card_steps = [
            ('encoder', UniversalEncoder(
                high_cardinality_threshold=100,  # Higher threshold for high-card columns
                encoding_strategy='frequency'  # Use frequency encoding for high cardinality
            ))
        ]
        transformers.append(('high_cardinality', Pipeline(high_card_steps), high_card_cols))
    
    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop datetime and text columns for now
        sparse_threshold=0,
        n_jobs=-1
    )
    
    preprocessor.set_output(transform="pandas")
    return preprocessor