from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from tabpfn import TabPFNRegressor

from automl.FeatureSelector import FeatureSelector


# XGBRegressor
param = {
    "model__objective": "reg:squarederror",
    "model__n_estimators": trial.suggest_int("budget", 50, 500),  
    "model__booster": trial.suggest_categorical("model__booster", ["gbtree", "gblinear"]),
    "model__lambda": trial.suggest_float("model__lambda", 1e-8, 1.0, log=True),
    "model__alpha": trial.suggest_float("model__alpha", 1e-8, 1.0, log=True),
    "model__subsample": trial.suggest_float("model__subsample", 0.2, 1.0),
    "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.2, 1.0),
    "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),
    
    "featureselector__select_method": trial.suggest_categorical("featureselector__select_method", ["permutation", "tree"]),
    "featureselector__add_polynomial_features_xgb": trial.suggest_categorical("featureselector__add_polynomial_features_xgb", [True, False]),
    "featureselector__add_binning_features_xgb": trial.suggest_categorical("featureselector__add_binning_features_xgb", [True, False]),
    "featureselector__add_statistical_features_xgb": trial.suggest_categorical("featureselector__add_statistical_features_xgb", [True, False])
}
    
if param["model__booster"] in ["gbtree", "dart"]:
    param["model__max_depth"] = trial.suggest_int("model__max_depth", 3, 9, step=2)
    param["model__min_child_weight"] = trial.suggest_int("model__min_child_weight", 2, 10)
    param["model__eta"] = trial.suggest_float("model__eta", 1e-8, 1.0, log=True)
    param["model__gamma"] = trial.suggest_float("model__gamma", 1e-8, 1.0, log=True)
    param["model__grow_policy"] = trial.suggest_categorical("model__grow_policy", ["depthwise", "lossguide"])

if param["model__booster"] == "dart":
    param["model__sample_type"] = trial.suggest_categorical("model__sample_type", ["uniform", "weighted"])
    param["model__normalize_type"] = trial.suggest_categorical("model__normalize_type", ["tree", "forest"])
    param["model__rate_drop"] = trial.suggest_float("model__rate_drop", 1e-8, 1.0, log=True)
    param["model__skip_drop"] = trial.suggest_float("model__skip_drop", 1e-8, 1.0, log=True)


#lgbm params
param = {
    "model__objective": "regression",
    "model__n_estimators": trial.suggest_int("budget", 50, 500), 
    "model__boosting_type": trial.suggest_categorical("model__boosting_type", ["gbdt", "dart"]),
    "model__num_leaves": trial.suggest_int("model__num_leaves", 10, 300),
    "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
    "model__feature_fraction": trial.suggest_float("model__feature_fraction", 0.4, 1.0),
    "model__bagging_fraction": trial.suggest_float("model__bagging_fraction", 0.4, 1.0),
    "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 10.0, log=True),
    "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 10.0, log=True),
    
    
    "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.7, 1.0),
    "featureselector__select_method": trial.suggest_categorical("featureselector__select_method", ["permutation", "tree"]),
    "featureselector__add_polynomial_features_lgb": trial.suggest_categorical("featureselector__add_polynomial_features_lgb", [True, False]),
    "featureselector__add_binning_features_lgb": trial.suggest_categorical("featureselector__add_binning_features_lgb", [True, False]),
    "featureselector__add_statistical_features_lgb": trial.suggest_categorical("featureselector__add_statistical_features_lgb", [True, False]),
}

# MLPRegressor params
param = {
    "model__hidden_layer_sizes": trial.suggest_categorical("model__hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 50)]),
    "model__activation": trial.suggest_categorical("model__activation", ["relu", "tanh"]),
    "model__alpha": trial.suggest_float("model__alpha", 1e-5, 1.0, log=True),
    "model__learning_rate_init": trial.suggest_float("model__learning_rate_init", 1e-4, 1e-2, log=True),
    "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),
}

# RandomForestRegressor params
RandomForestRegressor()
param = {
    "model__n_estimators": trail.suggest_int(50, 500),
    # "model__max_depth": trial.suggest_int("model__max_depth", 3, 20),
    "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 10),
    "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 4),
    "model__max_features": trial.suggest_categorical("model__max_features", [1, "sqrt", "log2"]),
    "model__bootstrap": trial.suggest_categorical("model__bootstrap", [True, False]),
    "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),
}

# GradientBoostingRegressor params
GradientBoostingRegressor()
param = {
    "model__n_estimators": trial.suggest_int("budget", 50, 500),  # Use budget as n_estimators
    "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
    "model__max_depth": trial.suggest_int("model__max_depth", 1, 500),
    "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 100),
    "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 100),
    "model__max_features": trial.suggest_categorical("model__max_features", [None, "sqrt", "log2"]),
    "model__alpha": trial.suggest_float("model__alpha", 0.01, 0.99),
    "model__max_leaf_nodes": trial.suggest_int("model__max_leaf_nodes", 2, 1000, log=True),
    "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),   
}

# HistGradientBoostingRegressor params

LinearRegression
param = {
    "model__fit_intercept": trial.suggest_categorical("model__fit_intercept", [True, False]),
    "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),
}

BayesianRidge

params = {
    "model__max_iter": trial.suggest_int("model__max_iter", 100, 1000),
    "model__tol": trial.suggest_float("model__tol", 1e-5, 1e-2, log=True),
    "model__alpha_1": trial.suggest_float("model__alpha_1", 1e-6, 1.0, log=True),
    "model__alpha_2": trial.suggest_float("model__alpha_2", 1e-6, 1.0, log=True),
    "model__lambda_1": trial.suggest_float("model__lambda_1", 1e-6, 1.0, log=True),
    "model__lambda_2": trial.suggest_float("model__lambda_2", 1e-6, 1.0, log=True),
    "model__compute_score": trial.suggest_categorical("model__compute_score", [True, False]),
    "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),
}

DecisionTreeRegressor
param = {
    "model__splitter": trial.suggest_categorical("model__splitter", ["best", "random"]),
    "model__criterion": trial.suggest_categorical("model__criterion", ["squared_error", "friedman_mse", "absolute_error"]),
    "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 10),
    "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 4),
    "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),
}

SVR
param = {
    "model__kernel": trial.suggest_categorical("model__kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"]),
    "model__degree": trial.suggest_int("model__degree", 2, 5),
    "model__gamma": trial.suggest_categorical("model__gamma", ["scale", "auto"]),
    "model__C": trial.suggest_float("model__C", 1e-3, 1e3, log=True),
    "model__epsilon": trial.suggest_float("model__epsilon", 1e-3, 1.0, log=True),
    "model__degree": trial.suggest_int("model__degree", 2, 5),
    "model__shrinking": trial.suggest_categorical("model__shrinking", [True, False]),
    ""
    "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),
}

TabPFNRegressor