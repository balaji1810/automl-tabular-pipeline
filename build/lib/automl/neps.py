import neps
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import random
import torch

def set_seed(seed: int):
    """
    Set the seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def hyperparam_search_neps(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_space: dict,
    random_state: int = 20,
) -> Pipeline:
    """
    Run a NEPS Experiment to maximize `scoring` over sklearn Pipeline.
    """
    def evaluate(**config) -> float:
        pipeline.set_params(**config)
        pipeline.fit(X=X_train, y=y_train)
        return -float(pipeline.score(X_val, y_val))
    set_seed(random_state)
    neps.run(
        evaluate_pipeline=evaluate,
        root_directory='neps/',
        pipeline_space=param_space,
        post_run_summary=True,
        overwrite_working_directory=True,
        max_evaluations_total=50,
        # ignore_errors=True
    )
    full, short = neps.status('neps/')

    # set the best hyperparameters
    pipeline.set_params(**short[[*param_space]].to_dict())
    return pipeline
