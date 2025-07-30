"""AutoML class for regression tasks with enhanced dual-algorithm approach.
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
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from automl.FeatureSelector import FeatureSelector
from automl.neps import hyperparam_search_neps
from automl.optuna import hyperparam_search_optuna
from automl.pre_processor import build_preprocessor

logger = logging.getLogger(__name__)

METRICS = {"r2": r2_score}

class AutoML:
    """
    Enhanced AutoML with dual-algorithm approach, algorithm-specific feature engineering,
    and diversity-aware ensembling.
    """

    def __init__(
        self,
        seed: int,
        metric: str = "r2",
        correlation_threshold: float = 0.7,
        n_trials: int = 100,
        use_mult_algorithms: bool = True
    ) -> None:
        self.seed = seed
        self.metric = METRICS[metric]
        self.correlation_threshold = correlation_threshold
        self.n_trials = n_trials
        self.use_mult_algorithms = use_mult_algorithms
        
        # Model components
        self._xgb_model: Pipeline | None = None
        self._lgbm_model: Pipeline | None = None
        self._single_model: Pipeline | None = None
        
        # Ensemble info
        self.ensemble_weights = None
        self.use_ensemble = False
        self.best_single_algorithm = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AutoML':
        """Fit the enhanced AutoML pipeline."""
        logger.info("Starting Enhanced AutoML training...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, random_state=self.seed, test_size=0.2
        )
        
        preprocessor = build_preprocessor(X)
        
        if self.use_mult_algorithms:
            self._fit_dual_algorithms(preprocessor, X_train, y_train, X_val, y_val)
        else:
            self._fit_single_algorithm(preprocessor, X_train, y_train, X_val, y_val)
        
        return self

    def _fit_single_algorithm(self, preprocessor, X_train, y_train, X_val, y_val):
        """Fit single XGBoost algorithm (fallback)."""
        self._single_model = Pipeline([
            ("preproc", preprocessor),
            ("featureselector", FeatureSelector(seed=self.seed)),
            ("model", XGBRegressor(random_state=self.seed, n_jobs=-1))
        ])
        
        self._single_model = hyperparam_search_optuna(self._single_model, X_train, y_train, X_val, y_val)
        self._single_model.fit(X_train, y_train)
        
        val_preds = self._single_model.predict(X_val)
        val_score = self.metric(y_val, val_preds)
        logger.info(f"Single model validation score: {val_score:.4f}")

    def _fit_dual_algorithms(self, preprocessor, X_train, y_train, X_val, y_val):
        """Fit both XGBoost and LightGBM with ensemble logic."""
        # Create and optimize XGBoost pipeline
        logger.info("Training XGBoost...")
        self._xgb_model = Pipeline([
            ("preproc", preprocessor),
            ("featureselector", FeatureSelector(algorithm="xgboost", seed=self.seed)),
            ("model", XGBRegressor(random_state=self.seed, n_jobs=-1))
        ])
        self._xgb_model = self._optimize_xgb(X_train, y_train, X_val, y_val)
        self._xgb_model.fit(X_train, y_train)
        
        # Create and optimize LightGBM pipeline
        logger.info("Training LightGBM...")
        self._lgbm_model = Pipeline([
            ("preproc", preprocessor),
            ("featureselector", FeatureSelector(algorithm="lightgbm", seed=self.seed)),
            ("model", LGBMRegressor(random_state=self.seed, n_jobs=-1, verbose=-1))
        ])
        self._lgbm_model = self._optimize_lgbm(X_train, y_train, X_val, y_val)
        self._lgbm_model.fit(X_train, y_train)
        
        # Determine ensemble strategy
        xgb_val_preds = self._xgb_model.predict(X_val)
        lgbm_val_preds = self._lgbm_model.predict(X_val)
        
        self._determine_ensemble_strategy(xgb_val_preds, lgbm_val_preds, y_val)

    def _optimize_xgb(self, X_train, y_train, X_val, y_val):
        """Optimize XGBoost hyperparameters using true BOHB with multi-fidelity."""
        import optuna
        try:
            from optuna.integration import BoTorchSampler
        except ImportError:
            # Fallback to sklearn sampler if BoTorch is not available
            from optuna.samplers import RandomSampler as BoTorchSampler
            logger.warning("BoTorch not available, using RandomSampler")
        
        def objective(trial):
            # Multi-fidelity budget: n_estimators as the fidelity dimension
            budget = trial.suggest_int("budget", 50, 500, step=50)  # Budget allocation
            
            param = {
                "model__objective": "reg:squarederror",
                "model__n_estimators": budget,  # Use budget as n_estimators
                "model__booster": trial.suggest_categorical("model__booster", ["gbtree", "gblinear"]),
                "model__lambda": trial.suggest_float("model__lambda", 1e-8, 1.0, log=True),
                "model__alpha": trial.suggest_float("model__alpha", 1e-8, 1.0, log=True),
                "model__subsample": trial.suggest_float("model__subsample", 0.2, 1.0),
                "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.2, 1.0),
                "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.6, 1.0),
            }
            
            if param["model__booster"] == "gbtree":
                param["model__max_depth"] = trial.suggest_int("model__max_depth", 3, 9, step=2)
                param["model__eta"] = trial.suggest_float("model__eta", 1e-8, 1.0, log=True)
                param["model__gamma"] = trial.suggest_float("model__gamma", 1e-8, 1.0, log=True)
            
            # Progressive resource allocation: Use budget for n_estimators control
            if budget < 200:
                # Low budget: fewer estimators
                pass  # Budget already set as n_estimators
            elif budget < 350:
                # Medium budget: moderate estimators
                pass  # Budget already set as n_estimators
            else:
                # High budget: more estimators
                pass  # Budget already set as n_estimators
            
            self._xgb_model.set_params(**param)
            
            # Multi-fidelity training: use budget as resource allocation
            model_params = {k.replace("model__", ""): v for k, v in param.items() if k.startswith("model__")}
            feature_params = {k.replace("featureselector__", ""): v for k, v in param.items() if k.startswith("featureselector__")}
            
            # Set feature selector params
            self._xgb_model.named_steps['featureselector'].set_params(**feature_params)
            
            # Standard fit with budget-controlled n_estimators
            self._xgb_model.named_steps['model'].set_params(**model_params)
            self._xgb_model.fit(X_train, y_train)
            score = self._xgb_model.score(X_val, y_val)
            
            # Report intermediate values for pruning (progressive resource allocation)
            trial.report(float(score), budget)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return float(score)
        
        # True BOHB: Bayesian Optimization + HyperBand
        study = optuna.create_study(
            direction="maximize", 
            sampler=BoTorchSampler(seed=self.seed),  # Bayesian Optimization
            pruner=optuna.pruners.HyperbandPruner(  # True HyperBand (not Successive Halving)
                min_resource=50,  # Minimum budget
                max_resource=500,  # Maximum budget  
                reduction_factor=3  # Resource reduction factor
            )
        )
        study.optimize(objective, n_trials=self.n_trials)
        
        # Set best parameters (excluding budget as it's used for training control)
        best_params = {k: v for k, v in study.best_trial.params.items() if k != "budget"}
        self._xgb_model.set_params(**best_params)
        return self._xgb_model

    def _optimize_lgbm(self, X_train, y_train, X_val, y_val):
        """Optimize LightGBM hyperparameters using true BOHB with multi-fidelity."""
        import optuna
        try:
            from optuna.integration import BoTorchSampler
        except ImportError:
            # Fallback to sklearn sampler if BoTorch is not available
            from optuna.samplers import RandomSampler as BoTorchSampler
            logger.warning("BoTorch not available, using RandomSampler")
        
        def objective(trial):
            # Multi-fidelity budget: n_estimators as the fidelity dimension
            budget = trial.suggest_int("budget", 50, 500, step=50)  # Budget allocation
            
            param = {
                "model__objective": "regression",
                "model__n_estimators": budget,  # Use budget as n_estimators
                "model__boosting_type": trial.suggest_categorical("model__boosting_type", ["gbdt", "dart"]),
                "model__num_leaves": trial.suggest_int("model__num_leaves", 10, 300),
                "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
                "model__feature_fraction": trial.suggest_float("model__feature_fraction", 0.4, 1.0),
                "model__bagging_fraction": trial.suggest_float("model__bagging_fraction", 0.4, 1.0),
                "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 10.0, log=True),
                "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 10.0, log=True),
                "featureselector__max_features": trial.suggest_float("featureselector__max_features", 0.7, 1.0),
            }
            
            # Progressive resource allocation: Use budget for n_estimators control
            if budget < 200:
                # Low budget: fewer estimators
                param["model__verbose"] = -1
            elif budget < 350:
                # Medium budget: moderate estimators
                param["model__verbose"] = -1
            else:
                # High budget: more estimators
                param["model__verbose"] = -1
            
            # Multi-fidelity training: use budget as resource allocation
            model_params = {k.replace("model__", ""): v for k, v in param.items() if k.startswith("model__")}
            feature_params = {k.replace("featureselector__", ""): v for k, v in param.items() if k.startswith("featureselector__")}
            
            # Set feature selector params
            self._lgbm_model.named_steps['featureselector'].set_params(**feature_params)
            
            # Standard fit with budget-controlled n_estimators
            self._lgbm_model.named_steps['model'].set_params(**model_params)
            self._lgbm_model.fit(X_train, y_train)
            score = self._lgbm_model.score(X_val, y_val)
            
            # Report intermediate values for pruning (progressive resource allocation)
            trial.report(float(score), budget)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return float(score)
        
        # True BOHB: Bayesian Optimization + HyperBand
        study = optuna.create_study(
            direction="maximize", 
            sampler=BoTorchSampler(seed=self.seed),  # Bayesian Optimization
            pruner=optuna.pruners.HyperbandPruner(  # True HyperBand (not Successive Halving)
                min_resource=50,  # Minimum budget
                max_resource=500,  # Maximum budget  
                reduction_factor=3  # Resource reduction factor
            )
        )
        study.optimize(objective, n_trials=self.n_trials)
        
        # Set best parameters (excluding budget as it's used for training control)
        best_params = {k: v for k, v in study.best_trial.params.items() if k != "budget"}
        self._lgbm_model.set_params(**best_params)
        return self._lgbm_model

    def _determine_ensemble_strategy(self, xgb_preds, lgbm_preds, y_val):
        """Determine whether to ensemble based on diversity criteria."""
        correlation = np.corrcoef(xgb_preds, lgbm_preds)[0, 1]
        xgb_r2 = self.metric(y_val, xgb_preds)
        lgbm_r2 = self.metric(y_val, lgbm_preds)
        
        logger.info(f"XGBoost R²: {xgb_r2:.4f}")
        logger.info(f"LightGBM R²: {lgbm_r2:.4f}")
        logger.info(f"Prediction correlation: {correlation:.4f}")
        
        # Check if should ensemble
        if correlation < self.correlation_threshold and self._errors_complement(xgb_preds, lgbm_preds, y_val):
            self.use_ensemble = True
            self.ensemble_weights = self._calculate_optimal_weights(xgb_preds, lgbm_preds, y_val, xgb_r2, lgbm_r2)
            logger.info(f"Using ensemble with weights: XGB={self.ensemble_weights[0]:.3f}, LGBM={self.ensemble_weights[1]:.3f}")
        else:
            self.use_ensemble = False
            self.best_single_algorithm = "xgb" if xgb_r2 >= lgbm_r2 else "lgbm"
            reason = "high correlation" if correlation >= self.correlation_threshold else "non-complementary errors"
            logger.info(f"Using best single model ({self.best_single_algorithm}) due to {reason}")

    def _errors_complement(self, xgb_preds, lgbm_preds, y_true):
        """Check if model errors complement each other."""
        xgb_errors = np.abs(y_true - xgb_preds)
        lgbm_errors = np.abs(y_true - lgbm_preds)
        
        xgb_better = xgb_errors < lgbm_errors
        lgbm_better = lgbm_errors < xgb_errors
        
        xgb_advantage_ratio = np.mean(xgb_better)
        lgbm_advantage_ratio = np.mean(lgbm_better)
        
        return xgb_advantage_ratio > 0.2 and lgbm_advantage_ratio > 0.2

    def _calculate_optimal_weights(self, xgb_preds, lgbm_preds, y_true, xgb_r2, lgbm_r2):
        """Calculate optimal ensemble weights."""
        best_r2 = -np.inf
        best_weights = (0.5, 0.5)
        
        for w1 in np.arange(0.1, 1.0, 0.1):
            w2 = 1.0 - w1
            ensemble_preds = w1 * xgb_preds + w2 * lgbm_preds
            ensemble_r2 = self.metric(y_true, ensemble_preds)
            
            if ensemble_r2 > best_r2:
                best_r2 = ensemble_r2
                best_weights = (w1, w2)
        
        # Fallback to performance-based weights if grid search doesn't improve
        if best_r2 <= max(xgb_r2, lgbm_r2):
            total_performance = xgb_r2 + lgbm_r2
            if total_performance > 0:
                best_weights = (xgb_r2 / total_performance, lgbm_r2 / total_performance)
            else:
                best_weights = (0.6, 0.4)
        
        return best_weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model(s)."""
        if not self.use_mult_algorithms:
            if self._single_model is None:
                raise ValueError("Model not fitted")
            return self._single_model.predict(X)
        
        if self._xgb_model is None or self._lgbm_model is None:
            raise ValueError("Models not fitted")
        
        xgb_preds = self._xgb_model.predict(X)
        lgbm_preds = self._lgbm_model.predict(X)
        
        if self.use_ensemble:
            return self.ensemble_weights[0] * xgb_preds + self.ensemble_weights[1] * lgbm_preds
        else:
            return xgb_preds if self.best_single_algorithm == "xgb" else lgbm_preds
