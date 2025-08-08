from math import floor
from typing import Literal
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, f_regression, mutual_info_regression, r_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

class FeatureSelector:
    def __init__(
            self, max_features: float = 0.75,
            select_method: Literal[
                "permutation",
                "tree",
                "TruncatedSVD",
                "PCA",
                "KBest",
                "RecursiveFeatureElimination",
                "SequentialFeatureSelector"] = "permutation",
            score_func = "f_regression",
            direction: Literal["forward", "backward"] = "forward",
            seed: int = 20
    ):
        self.max_features = max_features
        self.select_method = select_method
        self.seed = seed
        self.score_func = score_func
        self.direction : Literal["forward", "backward"] = direction
        self.X_columns : pd.Index = pd.Index([])
        self.svd: TruncatedSVD | None = None
        self.pca: PCA | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the feature selector and store selected features.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)

        reg1 = RandomForestRegressor(random_state=self.seed, n_jobs=-1)
        num_features = floor(len(X_train.columns) * self.max_features)

        if self.select_method == "tree":
            reg1.fit(X_train, y_train)
            tree_importance_sorted_idx = np.argsort(reg1.feature_importances_)
            self.X_columns = X_train.columns[tree_importance_sorted_idx][-num_features:]
        elif self.select_method == "permutation":
            reg1.fit(X_train, y_train)
            result = permutation_importance(reg1, X_test, y_test, n_repeats=10, random_state=self.seed, n_jobs=-1)
            perm_sorted_idx = result["importances_mean"].argsort()
            self.X_columns = X_train.columns[perm_sorted_idx][-num_features:]
        elif self.select_method == "TruncatedSVD":
            self.svd = TruncatedSVD(n_components=num_features, random_state=self.seed)
            self.svd.fit(X_train)
            self.X_columns = pd.Index([f"svd_{i}" for i in range(num_features)])
        elif self.select_method == "PCA":
            self.pca = PCA(n_components=num_features, random_state=self.seed)
            self.pca.fit(X_train)
            self.X_columns = pd.Index([f"pca_{i}" for i in range(num_features)])
        elif self.select_method == "KBest":
            score_function = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression,
                "r_regression": r_regression
            }
            selector = SelectKBest(score_func=score_function[self.score_func], k=num_features)
            selector.fit(X_train, y_train)
            self.X_columns = X_train.columns[selector.get_support(indices=True)]
        elif self.select_method == "RecursiveFeatureElimination":
            rfe_selector = RFE(estimator=reg1, n_features_to_select=num_features)
            rfe_selector.fit(X_train, y_train)
            self.X_columns = X_train.columns[rfe_selector.get_support(indices=True)]
        elif self.select_method == "SequentialFeatureSelector":
            reg2 = RandomForestRegressor(random_state=self.seed, n_jobs=-1)
            sfs = SequentialFeatureSelector(estimator=reg2, n_features_to_select=num_features, direction=self.direction, n_jobs=-1)
            sfs.fit(X_train, y_train)
            self.X_columns = X_train.columns[sfs.get_support(indices=True)]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by selecting features based on importance.
        """
        if self.select_method == "TruncatedSVD":
            return pd.DataFrame(self.svd.transform(X), columns=self.X_columns, index=X.index)
        elif self.select_method == "PCA":
            return pd.DataFrame(self.pca.transform(X), columns=self.X_columns, index=X.index)
        return X[self.X_columns]
    
    def set_params(self, **params):
        """
        Set parameters for the feature selector.
        """
        self.X_columns = pd.Index([])
        self.svd = None
        self.pca = None
        self.direction = "forward"
        for key, value in params.items():
            setattr(self, key, value)
        return self
    