from math import floor
from typing import Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:
    def __init__(self, max_features: float = 0.75, select_method: Literal["permutation", "tree"] = "permutation", seed: int = 20):
        self.max_features = max_features
        self.select_method = select_method
        self.seed = seed
        self.X_perm_columns : pd.Index = pd.Index([])
        self.X_tree_columns : pd.Index = pd.Index([])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the feature selector and store selected features.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
        
        reg1 = RandomForestRegressor(random_state=self.seed, n_jobs=-1)
        reg1.fit(X_train, y_train)
        num_features = floor(len(X_train.columns) * self.max_features)

        if self.select_method == "tree":
            tree_importance_sorted_idx = np.argsort(reg1.feature_importances_)
            self.X_tree_columns = X_train.columns[tree_importance_sorted_idx][-num_features:]
        elif self.select_method == "permutation":
            result = permutation_importance(reg1, X_test, y_test, n_repeats=10, random_state=self.seed, n_jobs=-1)
            perm_sorted_idx = result["importances_mean"].argsort()
            self.X_perm_columns = X_train.columns[perm_sorted_idx][-num_features:]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by selecting features based on importance.
        """
        return X[self.X_perm_columns] if self.select_method == "permutation" else X[self.X_tree_columns]
    
    def set_params(self, **params):
        """
        Set parameters for the feature selector.
        """
        self.y = None
        self.X_perm_columns = pd.Index([])
        self.X_tree_columns = pd.Index([])
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
