"""AutoML class for regression tasks.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging

import neps
from sklearn.pipeline import Pipeline
import torch
from xgboost import XGBRegressor
from automl.meta_features import extract_meta_features
from automl.FeatureSelector import FeatureSelector
from automl.neps import hyperparam_search_neps
from automl.optuna import hyperparam_search_optuna
from automl.pre_processor import build_preprocessor
from constants import algorithms_dict

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
        self._model: list[Pipeline] = []
        self.val_preds: np.ndarray | None = None
        self.val_score: float | None = None

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
            # stratify=y if y.nunique() < 10 else None
        )
        preprocessor = build_preprocessor(X)
        meta_features = extract_meta_features(X, y)
        logger.info(f"Extracted meta-features")
        meta_model = torch.load("meta_model.pth")
        meta_model_predictions = meta_model.predict(meta_features)

        logger.info(f"Meta-model predictions: {meta_model_predictions}")

        for model_name in meta_model_predictions:
            model = algorithms_dict[model_name]
            logger.info(f"Selected model: {model_name}")

            model_pipeline = Pipeline([
                ("preproc", preprocessor),
                ("featureselector", FeatureSelector(seed=self.seed)),
                ("model", model)
            ])

            model_pipeline = hyperparam_search_optuna(model_pipeline, X_train, y_train, X_val, y_val, model_name)

            model = model_pipeline.fit(X_train, y_train)
            self._model.append(model)
            self.val_preds(model.predict(X_val))
            self.val_score = self.metric(y_val, self.val_preds)
            logger.info(f"Validation score for {model_name}: {val_score}")

        val_preds = [model.predict(X_val) for model in self._model]
        val_score = self.metric(y_val, val_preds)
        logger.info(f"Validation score: {val_score}")

        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise ValueError("Model not fitted")

        return self._model.predict(X)