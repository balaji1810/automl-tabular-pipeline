"""AutoML class for regression tasks.
"""
from __future__ import annotations

# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging
import os

from sklearn.pipeline import Pipeline
import torch
# from xgboost import XGBRegressor
from automl.meta_features import extract_meta_features
from automl.FeatureSelector_new import FeatureSelector
# from automl.neps import hyperparam_search_neps
from automl.optuna_util import hyperparam_search_optuna
from automl.pre_processor import build_preprocessor
from automl.constants import algorithms_dict
from automl.meta_trainer import load_ranking_meta_model, predict_algorithm_rankings

logger = logging.getLogger(__name__)

METRICS = {"r2": r2_score}

class AutoML:

    def __init__(
        self,
        seed: int = 10,
        timeout: int = 60,
        metric: str = "r2",
    ) -> None:
        self.seed = seed
        self.timeout = timeout
        self.metric = METRICS[metric]
        self.models: list[Pipeline] = []
        self.val_preds: dict[str, np.ndarray] = {}
        self.val_score: dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> AutoML:

        preprocessor = build_preprocessor(X)
        meta_features = extract_meta_features(X, y)
        logger.info(f"Extracted meta-features")

        meta_features = pd.DataFrame([meta_features])
        meta_model, checkpoint = load_ranking_meta_model("src/automl/meta_model_uci.pth")
        meta_model_predictions = predict_algorithm_rankings(meta_model, checkpoint, meta_features)
        
        meta_model_predictions = meta_model_predictions.apply(
            lambda row: pd.Series(
                row[row.apply(lambda x: isinstance(x, (int, float)))].sort_values().values,
                index=row[row.apply(lambda x: isinstance(x, (int, float)))].sort_values().index
            ),
            axis=1
        )
        selected_algorithms = meta_model_predictions.iloc[:,:4] # Keep top 4 algorithms
        
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            random_state=self.seed,
            test_size=0.2,
        )
        
        eval_time = int(self.timeout / len(selected_algorithms))

        for model_name in selected_algorithms:
            model = algorithms_dict[model_name]
            if model_name == "LinearRegression" or model_name == "BayesianRidge" or model_name == "SVR":
                pass
            else :
                model.set_params(random_state=self.seed)
            logger.info(f"Selected model: {model_name}")

            model_pipeline = Pipeline([
                ("preproc", preprocessor),
                ("featureselector", FeatureSelector(seed=self.seed)),
                ("model", model)
            ])

            model_pipeline = hyperparam_search_optuna(model_pipeline, X_train, y_train, X_val, y_val, model_name, eval_time)

            model = model_pipeline.fit(X_train, y_train)
            self.models.append(model)
            self.val_preds[model_name] = model.predict(X_val)
            self.val_score[model_name] = self.metric(y_val, self.val_preds[model_name])
            logger.info(f"Validation score for {model_name}: {self.val_score[model_name]}")
        logger.info("All models fitted")

        self.best_model_name = max(self.val_score, key=self.val_score.get)
        self.best_model = next(model for model in self.models if model.named_steps['model'].__class__.__name__ == self.best_model_name)
        logger.info(f"Best model found: {self.best_model_name} with score: {self.val_score[self.best_model_name]}")

        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict using the best model."""
        if not self.models:
            raise ValueError("Model has not been fitted yet. Call fit() before predict().")
        predictions = self.best_model.predict(X)
        return predictions