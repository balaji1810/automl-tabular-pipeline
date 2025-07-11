"""AutoML class for regression tasks.
"""
from __future__ import annotations

# from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging

import neps
from sklearn.pipeline import Pipeline
from torch import seed
from xgboost import XGBRegressor
from automl.neps import hyperparam_search_neps
from automl.pre_processor import build_numeric_preprocessor

logger = logging.getLogger(__name__)

METRICS = {"r2": r2_score}

class AutoML:

    def __init__(
        self,
        seed: int,
        metric: str = "r2",
    ) -> None:
        self.seed = seed
        self.metric = METRICS[metric]
        self._model: Pipeline | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> AutoML:

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            random_state=self.seed,
            test_size=0.2,
        )
        preprocessor = build_numeric_preprocessor()

        model_pipeline = Pipeline([
            ("preproc", preprocessor),
            ("model", XGBRegressor(random_state=42))
        ])

        pipeline_space = {
            'model__n_estimators' : neps.Integer(50, 500),
            'model__max_depth' : neps.Integer(4, 10),
            'model__gamma' : neps.Float(0, 3),
            'model__learning_rate' : neps.Float(0, 1),
            'model__tree_method' : neps.Categorical(['auto', 'exact', 'approx', 'hist']),
            'model__reg_lambda': neps.Float(0, 5),
            'model__reg_alpha' : neps.Float(0, 2),
            'model__grow_policy' : neps.Categorical(['depthwise', 'lossguide']),
            'model__min_child_weight' : neps.Float(0, 2)
        }

        model_pipeline = hyperparam_search_neps(model_pipeline, X_train, y_train, X_val, y_val, pipeline_space, self.seed)

        model = model_pipeline.fit(X_train, y_train)
        self._model = model

        val_preds = model.predict(X_val)
        val_score = self.metric(y_val, val_preds)
        logger.info(f"Validation score: {val_score}")

        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise ValueError("Model not fitted")

        return self._model.predict(X)
