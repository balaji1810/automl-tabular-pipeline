import optuna
from pyparsing import Opt
try:
    from optuna.integration import BoTorchSampler
except ImportError:
    # Fallback to TPE sampler if BoTorch is not available
    BoTorchSampler = optuna.samplers.TPESampler

import pandas as pd
from sklearn.pipeline import Pipeline

def fetch_trials(trial : optuna.Trial, model_name : str, ) -> pd.DataFrame:
    """
    Fetch all trials from the Optuna study and return as a DataFrame.
    """
    
    return trials.reset_index(drop=True)


def hyperparam_search_optuna(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str
) -> Pipeline:
    """
    Run a BOHB optimization to maximize `scoring` over sklearn Pipeline with multi-fidelity optimization.
    """
    def objective(trial : optuna.Trial, model_name: str) -> float:

        param = fetch_trials(trial, model_name)
        
        # pipeline_params = {k: v for k, v in param.items() if k != "budget"}
        pipeline.set_params(**pipeline_params)
        
        # Standard training with budget as n_estimators
        pipeline.fit(X=X_train, y=y_train)
        score = float(pipeline.score(X_val, y_val))

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return score

    # True BOHB: Bayesian Optimization + HyperBand
    study = optuna.create_study(
        direction="maximize",
        sampler=BoTorchSampler(),  # Bayesian Optimization
        pruner=optuna.pruners.HyperbandPruner(  # True HyperBand
            min_resource=50,  # Minimum budget
            max_resource=500,  # Maximum budget
            reduction_factor=3  # Resource reduction factor
        )
    )
    study.optimize(lambda trials: objective(trials, model_name), n_trials=100)

    # Set best parameters (excluding budget as it's used for training control)
    best_params = {k: v for k, v in study.best_trial.params.items() if k != "budget"}
    pipeline.set_params(**best_params)
    return pipeline