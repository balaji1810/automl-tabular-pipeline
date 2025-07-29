"""AutoML class for regression tasks.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging

import optuna
import neps
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from automl.FeatureSelector import FeatureSelector
from automl.neps import hyperparam_search_neps
from automl.optuna import hyperparam_search_optuna
from automl.pre_processor import build_preprocessor

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
        preprocessor = build_preprocessor(X)

        model_pipeline = Pipeline([
            ("preproc", preprocessor),
            ("featureselector", FeatureSelector(seed=self.seed)),
            ("model", RandomForestRegressor())
        ])

        model_pipeline = hyperparam_search_optuna(model_pipeline, X_train, y_train, X_val, y_val, self.seed)
        model = model_pipeline.fit(X_train, y_train)
        self._model = model


        val_preds = model.predict(X_val)
        val_score = self.metric(y_val, val_preds)
        logger.info(f"________Validation score: {val_score}")

        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise ValueError("Model not fitted")

        return self._model.predict(X)
