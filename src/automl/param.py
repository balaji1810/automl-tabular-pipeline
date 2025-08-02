from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from tabpfn import TabPFNRegressor

# xgb pramas
param = {
            "model__objective": "reg:squarederror",
            "model__n_estimators": budget,  # Use budget as n_estimators
            "model__booster": trial.suggest_categorical("model__booster", ["gbtree", "gblinear"]),
            "model__lambda": trial.suggest_float("model__lambda", 1e-8, 1.0, log=True),
            "model__alpha": trial.suggest_float("model__alpha", 1e-8, 1.0, log=True),
            "model__subsample": trial.suggest_float("model__subsample", 0.2, 1.0),
            "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.2, 1.0),
            "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),
            
            "featureselector__select_method": trial.suggest_categorical("featureselector__select_method", ["permutation", "tree"]),
            "featureselector__add_polynomial_features_xgb": trial.suggest_categorical("featureselector__add_polynomial_features_xgb", [True, False]),
            "featureselector__add_binning_features_xgb": trial.suggest_categorical("featureselector__add_binning_features_xgb", [True, False]),
            "featureselector__add_statistical_features_xgb": trial.suggest_categorical("featureselector__add_statistical_features_xgb", [True, False]),
        }
   
#lgbm params
param = {
    "model__objective": "regression",
    "model__n_estimators": budget,  # Use budget as n_estimators
    "model__boosting_type": trial.suggest_categorical("model__boosting_type", ["gbdt", "dart"]),
    "model__num_leaves": trial.suggest_int("model__num_leaves", 10, 300),
    "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
    "model__feature_fraction": trial.suggest_float("model__feature_fraction", 0.4, 1.0),
    "model__bagging_fraction": trial.suggest_float("model__bagging_fraction", 0.4, 1.0),
    "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 10.0, log=True),
    "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 10.0, log=True),
    "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.7, 1.0),
}

# MLPRegressor params
param = { MLPRegressor() }

# RandomForestRegressor params
RandomForestRegressor()

# GradientBoostingRegressor params

# HistGradientBoostingRegressor params

LinearRegression params

BayesianRidge