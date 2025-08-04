import optuna
try:
    from optuna.integration import BoTorchSampler
except ImportError:
    # Fallback to TPE sampler if BoTorch is not available
    BoTorchSampler = optuna.samplers.TPESampler

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
    Run a BOHB optimization to maximize `scoring` over sklearn Pipeline with multi-fidelity optimization.
    """
    def objective(trial : optuna.Trial, model_name: str) -> float:

        param = fetch_params(trial, model_name)
        
        # pipeline_params = {k: v for k, v in param.items() if k != "budget"}
        pipeline.set_params(**param)
        
        # Standard training with budget as n_estimators
        pipeline.fit(X=X_train, y=y_train)
        score = float(pipeline.score(X_val, y_val))
        
        return score

    # True BOHB: Bayesian Optimization + HyperBand
    study = optuna.create_study(
        direction="maximize",
        sampler=BoTorchSampler()
    )
    study.optimize(lambda trials: objective(trials, model_name), timeout=timeout)

    best_params = study.best_trial.params
    pipeline.set_params(**best_params)
    return pipeline