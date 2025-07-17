import optuna

import pandas as pd
from sklearn.pipeline import Pipeline

def hyperparam_search_optuna(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Pipeline:
    """
    Run a NEPS Experiment to maximize `scoring` over sklearn Pipeline.
    """
    def objective(trial):

        param = {
            "model__objective": "reg:squarederror",
            "model__booster": trial.suggest_categorical("model__booster", ["gbtree", "gblinear", "dart"]),
            "model__lambda": trial.suggest_float("model__lambda", 1e-8, 1.0, log=True),
            "model__alpha": trial.suggest_float("model__alpha", 1e-8, 1.0, log=True),
            "model__subsample": trial.suggest_float("model__subsample", 0.2, 1.0),
            "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.2, 1.0),
            "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),
            "featureselector__select_method": trial.suggest_categorical("featureselector__select_method", ["permutation", "tree"]),
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
        
        pipeline.set_params(**param)
        pipeline.fit(X=X_train, y=y_train)
        return float(pipeline.score(X_val, y_val))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    pipeline.set_params(**study.best_trial.params)
    return pipeline