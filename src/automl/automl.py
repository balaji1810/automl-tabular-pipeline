"""AutoML class for regression tasks.
"""
from __future__ import annotations

# from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging

import neps
from sklearn.pipeline import Pipeline
# from torch import seed
# from xgboost import XGBRegressor
from automl.FeatureSelector import FeatureSelector
from automl.neps import hyperparam_search_neps
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
            ("model", RandomForestRegressor(random_state=self.seed, n_jobs=-1)),
        ])

        pipeline_space = {
            'model__n_estimators' : neps.Integer(90, 200, prior=100, prior_confidence='medium'),
            'model__criterion' : neps.Categorical(["squared_error", "friedman_mse"], prior='squared_error', prior_confidence='high'),
            'model__max_features' : neps.Categorical(['sqrt', 'log2', 1.0], prior=1.0),
            'model__min_samples_split' : neps.Integer(2, 5, prior=2),
            # 'model__bootstrap' : neps.Categorical([True, False], prior=True),
            'featureselector__max_features': neps.Float(0.6, 1.0, prior=0.75, prior_confidence='medium'),
            'featureselector__select_method': neps.Categorical(['permutation', 'tree'], prior='permutation', prior_confidence='high'),
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
