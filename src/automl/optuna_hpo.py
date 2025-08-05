import optuna
from optuna.samplers import TPESampler

import pandas as pd
from sklearn.pipeline import Pipeline
from automl.param import fetch_params


def hyperparam_search_optuna(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    timeout: int
) -> Pipeline:
    """
    Run a optimization to maximize `scoring` over sklearn Pipeline.
    """
    def objective(trial : optuna.Trial, model_name: str) -> float:

        param = fetch_params(trial, model_name)
        pipeline.set_params(**param)

        pipeline.fit(X=X_train, y=y_train)
        score = float(pipeline.score(X_val, y_val))
        
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler()
    )
    study.optimize(lambda trials: objective(trials, model_name), timeout=timeout)

    best_params = study.best_trial.params
    pipeline.set_params(**best_params)
    return pipeline