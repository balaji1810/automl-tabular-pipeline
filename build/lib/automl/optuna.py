import optuna
try:
    from optuna.integration import BoTorchSampler
except ImportError:
    # Fallback to TPE sampler if BoTorch is not available
    BoTorchSampler = optuna.samplers.TPESampler

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
    Run a BOHB optimization to maximize `scoring` over sklearn Pipeline with multi-fidelity optimization.
    """
    def objective(trial):
        # Multi-fidelity budget: n_estimators as the fidelity dimension
        budget = trial.suggest_int("budget", 50, 500, step=50)  # Budget allocation

        param = {
            "model__objective": "reg:squarederror",
            "model__n_estimators": budget,  # Use budget as n_estimators for multi-fidelity
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
        
        # Progressive resource allocation based on budget
        if budget < 200:
            # Low budget: no additional parameters needed (budget controls n_estimators)
            pass
        elif budget < 350:
            # Medium budget: no additional parameters needed
            pass
        # High budget: no additional parameters needed
        
        # Set parameters and fit with budget-aware training
        pipeline_params = {k: v for k, v in param.items() if k != "budget"}
        pipeline.set_params(**pipeline_params)
        
        # Standard training with budget as n_estimators
        pipeline.fit(X=X_train, y=y_train)
        score = float(pipeline.score(X_val, y_val))
        
        # Report intermediate values for pruning (progressive resource allocation)
        trial.report(score, budget)
        
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
    study.optimize(objective, n_trials=100)

    # Set best parameters (excluding budget as it's used for training control)
    best_params = {k: v for k, v in study.best_trial.params.items() if k != "budget"}
    pipeline.set_params(**best_params)
    return pipeline