"""
Complete AutoML System with Variable-Length Meta-Learning

This system combines the variable-length meta-learner with hyperparameter optimization
to provide a complete AutoML solution that can suggest algorithms by directly analyzing datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import warnings
warnings.filterwarnings('ignore')

# Algorithm imports
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# ML utilities
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Hyperparameter optimization
import optuna
import joblib

# Local imports
from .variable_meta_learner import DatasetMetaLearner
from .dataset_evaluator import build_preprocessor, get_algorithm_portfolio


class VariableMetaAutoML:
    """
    AutoML system using variable-length dataset meta-learning for algorithm selection.
    
    This system:
    1. Uses a neural network that can handle datasets with variable features
    2. Learns dataset representations directly from data samples
    3. Suggests algorithms based on entire dataset analysis
    4. Provides hyperparameter optimization and ensemble options
    """
    
    def __init__(self, 
                 meta_model_path: Optional[str] = None,
                 optimize_hyperparams: bool = True,
                 n_trials: int = 50,
                 use_ensemble: bool = False,
                 ensemble_top_k: int = 3,
                 random_state: int = 42):
        """
        Initialize the Variable Meta-AutoML system.
        
        Args:
            meta_model_path: Path to pre-trained variable-length meta-learner
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of trials for hyperparameter optimization
            use_ensemble: Whether to create ensemble from top-k algorithms
            ensemble_top_k: Number of top algorithms to include in ensemble
            random_state: Random state for reproducibility
        """
        self.meta_model_path = meta_model_path
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.use_ensemble = use_ensemble
        self.ensemble_top_k = ensemble_top_k
        self.random_state = random_state
        
        # Initialize components
        self.meta_learner = None
        self.algorithm_instances = get_algorithm_portfolio()
        self.fitted_models = {}
        self.preprocessor = None
        self.final_model = None
        self.training_results = {}
        
        # Load meta-learner
        self._initialize_meta_learner()
    
    def _initialize_meta_learner(self):
        """Initialize the variable-length meta-learner."""
        if self.meta_model_path and Path(self.meta_model_path).exists():
            try:
                self.meta_learner = DatasetMetaLearner()
                self.meta_learner.load(self.meta_model_path)
                print(f"Loaded pre-trained variable-length meta-learner from {self.meta_model_path}")
            except Exception as e:
                print(f"Failed to load meta-learner: {e}")
                print("Meta-learning will be disabled.")
                self.meta_learner = None
        else:
            print("No pre-trained meta-learner found. Will use RandomForestRegressor as default.")
            self.meta_learner = None
    
    def analyze_dataset(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Analyze dataset using variable-length meta-learner.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary with dataset analysis and recommendations
        """
        analysis = {}
        
        # Basic dataset info
        analysis['n_samples'] = X.shape[0]
        analysis['n_features'] = X.shape[1]
        analysis['feature_types'] = {
            'numerical': len(X.select_dtypes(include=[np.number]).columns),
            'categorical': len(X.select_dtypes(include=['object', 'category']).columns)
        }
        analysis['missing_values'] = X.isnull().sum().sum()
        analysis['target_stats'] = {
            'mean': float(y.mean()),
            'std': float(y.std()),
            'min': float(y.min()),
            'max': float(y.max())
        }
        
        # Meta-learning predictions
        if self.meta_learner:
            try:
                recommended_algorithm, algorithm_ranking = self.meta_learner.predict(X, y)
                
                analysis['recommended_algorithm'] = recommended_algorithm
                analysis['algorithm_ranking'] = algorithm_ranking
                analysis['meta_learning'] = 'variable_length_neural'
                
                print(f"Dataset Analysis:")
                print(f"  Shape: {X.shape}")
                print(f"  Missing values: {analysis['missing_values']}")
                print(f"  Feature types: {analysis['feature_types']}")
                print(f"  Recommended algorithm: {recommended_algorithm}")
                print("  Algorithm ranking (top 5):")
                for i, (alg, score) in enumerate(algorithm_ranking[:5]):
                    print(f"    {i+1}. {alg}: {score:.4f}")
                    
            except Exception as e:
                print(f"Error in meta-learning prediction: {e}")
                analysis['recommended_algorithm'] = "RandomForestRegressor"
                analysis['algorithm_ranking'] = []
                analysis['meta_learning'] = 'failed'
        else:
            analysis['recommended_algorithm'] = "RandomForestRegressor"  # Default fallback
            analysis['algorithm_ranking'] = []
            analysis['meta_learning'] = 'disabled'
            print("Meta-learner not available. Using RandomForestRegressor as default.")
        
        return analysis
    
    def _get_algorithm_hyperparams(self, algorithm_name: str) -> Dict[str, Any]:
        """Get hyperparameter search space for a given algorithm."""
        hyperparams = {
            'RandomForestRegressor': {
                'n_estimators': (50, 200),
                'max_depth': (3, 20),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 5)
            },
            'XGBRegressor': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0)
            },
            'LGBMRegressor': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'num_leaves': (10, 100)
            },
            'GradientBoostingRegressor': {
                'n_estimators': (50, 200),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.3)
            },
            'HistGradientBoostingRegressor': {
                'max_iter': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3)
            },
            'MLPRegressor': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'alpha': (1e-5, 1e-1),
                'learning_rate_init': (1e-4, 1e-2)
            },
            'SVR': {
                'C': (0.1, 100),
                'gamma': (1e-4, 1),
                'epsilon': (0.01, 1)
            }
        }
        return hyperparams.get(algorithm_name, {})
    
    def _optimize_hyperparameters(self, algorithm_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters for a given algorithm."""
        hyperparams = self._get_algorithm_hyperparams(algorithm_name)
        
        if not hyperparams:
            print(f"No hyperparameter optimization defined for {algorithm_name}")
            return {}
        
        def objective(trial):
            params = {}
            for param, space in hyperparams.items():
                if isinstance(space, tuple) and len(space) == 2:
                    if isinstance(space[0], int):
                        params[param] = trial.suggest_int(param, space[0], space[1])
                    else:
                        params[param] = trial.suggest_float(param, space[0], space[1])
                elif isinstance(space, list):
                    params[param] = trial.suggest_categorical(param, space)
            
            # Add random state if algorithm supports it
            if 'random_state' in self.algorithm_instances[algorithm_name].get_params():
                params['random_state'] = self.random_state
            
            # Create algorithm instance with suggested parameters
            algorithm_class = type(self.algorithm_instances[algorithm_name])
            model = algorithm_class(**params)
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            
            # Cross-validation score
            try:
                scores = cross_val_score(pipeline, X, y, cv=3, scoring='r2', n_jobs=1)
                return scores.mean()
            except Exception as e:
                print(f"Error in CV for {algorithm_name}: {e}")
                return -1.0
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize', 
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        try:
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            print(f"  Best score: {study.best_value:.4f}")
            return study.best_params
        except Exception as e:
            print(f"  Optimization failed: {e}")
            return {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Fit the Variable Meta-AutoML system.
        
        Args:
            X: Feature matrix
            y: Target values
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training results and model performance
        """
        start_time = time.time()
        
        print("Starting Variable Meta-AutoML training...")
        print(f"Dataset shape: {X.shape}")
        
        # Analyze dataset using variable-length meta-learner
        analysis = self.analyze_dataset(X, y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state
        )
        
        # Build preprocessor
        self.preprocessor = build_preprocessor(X)
        
        # Get algorithm recommendations
        recommended_algorithm = analysis['recommended_algorithm']
        
        if self.use_ensemble:
            # Train top-k algorithms for ensemble
            algorithm_ranking = analysis.get('algorithm_ranking', [])
            if algorithm_ranking:
                top_algorithms = [alg for alg, _ in algorithm_ranking[:self.ensemble_top_k]]
            else:
                top_algorithms = [recommended_algorithm, "XGBRegressor", "LGBMRegressor"][:self.ensemble_top_k]
            
            print(f"\nTraining ensemble with algorithms: {top_algorithms}")
            
            ensemble_models = {}
            ensemble_weights = []
            
            for algorithm_name in top_algorithms:
                if algorithm_name in self.algorithm_instances:
                    print(f"\nTraining {algorithm_name}...")
                    model = self._train_single_algorithm(algorithm_name, X_train, y_train)
                    
                    # Evaluate on validation set
                    val_score = r2_score(y_val, model.predict(X_val))
                    ensemble_models[algorithm_name] = model
                    ensemble_weights.append(max(0, val_score))  # Ensure non-negative weights
                    
                    print(f"  Validation R²: {val_score:.4f}")
            
            # Normalize weights
            total_weight = sum(ensemble_weights)
            if total_weight > 0:
                ensemble_weights = [w / total_weight for w in ensemble_weights]
            else:
                ensemble_weights = [1.0 / len(ensemble_models) for _ in ensemble_models]
            
            # Create ensemble
            self.final_model = EnsembleModel(ensemble_models, ensemble_weights)
            
        else:
            # Train single recommended algorithm
            print(f"\nTraining recommended algorithm: {recommended_algorithm}")
            self.final_model = self._train_single_algorithm(recommended_algorithm, X_train, y_train)
        
        # Final evaluation
        train_score = r2_score(y_train, self.final_model.predict(X_train))
        val_score = r2_score(y_val, self.final_model.predict(X_val))
        
        # Training results
        training_time = time.time() - start_time
        self.training_results = {
            'dataset_analysis': analysis,
            'train_r2': train_score,
            'validation_r2': val_score,
            'training_time': training_time,
            'final_algorithm': recommended_algorithm if not self.use_ensemble else 'Ensemble',
            'use_ensemble': self.use_ensemble,
            'meta_learning_type': 'variable_length_neural'
        }
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Training R²: {train_score:.4f}")
        print(f"Validation R²: {val_score:.4f}")
        
        return self.training_results
    
    def _train_single_algorithm(self, algorithm_name: str, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """Train a single algorithm with optional hyperparameter optimization."""
        if algorithm_name not in self.algorithm_instances:
            print(f"Algorithm {algorithm_name} not found in portfolio. Using RandomForestRegressor.")
            algorithm_name = "RandomForestRegressor"
        
        # Get base algorithm
        base_algorithm = self.algorithm_instances[algorithm_name]
        
        # Optimize hyperparameters if requested
        if self.optimize_hyperparams:
            print(f"  Optimizing hyperparameters...")
            best_params = self._optimize_hyperparameters(algorithm_name, X, y)
            if best_params:
                # Create new instance with optimized parameters
                algorithm_class = type(base_algorithm)
                base_params = base_algorithm.get_params()
                base_params.update(best_params)
                base_algorithm = algorithm_class(**base_params)
        
        # Create and train pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', base_algorithm)
        ])
        
        pipeline.fit(X, y)
        return pipeline
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.final_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.final_model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the model on test data."""
        predictions = self.predict(X)
        
        return {
            'r2_score': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions))
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.final_model is None:
            return {"status": "not_trained"}
        
        info = {
            "training_results": self.training_results,
            "use_ensemble": self.use_ensemble,
            "model_type": type(self.final_model).__name__,
            "meta_learning": "variable_length_neural"
        }
        
        if hasattr(self.final_model, 'named_steps'):
            info["algorithm"] = type(self.final_model.named_steps['model']).__name__
        
        return info
    
    def save(self, filepath: str):
        """Save the trained AutoML system."""
        model_data = {
            'final_model': self.final_model,
            'preprocessor': self.preprocessor,
            'training_results': self.training_results,
            'meta_model_path': self.meta_model_path,
            'config': {
                'optimize_hyperparams': self.optimize_hyperparams,
                'n_trials': self.n_trials,
                'use_ensemble': self.use_ensemble,
                'ensemble_top_k': self.ensemble_top_k,
                'random_state': self.random_state
            }
        }
        joblib.dump(model_data, filepath)
        print(f"Variable Meta-AutoML system saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a trained AutoML system."""
        model_data = joblib.load(filepath)
        
        # Create instance
        config = model_data['config']
        automl = cls(
            meta_model_path=model_data.get('meta_model_path'),
            optimize_hyperparams=config['optimize_hyperparams'],
            n_trials=config['n_trials'],
            use_ensemble=config['use_ensemble'],
            ensemble_top_k=config['ensemble_top_k'],
            random_state=config['random_state']
        )
        
        # Restore state
        automl.final_model = model_data['final_model']
        automl.preprocessor = model_data['preprocessor']
        automl.training_results = model_data['training_results']
        
        print(f"Variable Meta-AutoML system loaded from {filepath}")
        return automl


class EnsembleModel:
    """Weighted ensemble of multiple models."""
    
    def __init__(self, models: Dict[str, Any], weights: List[float]):
        self.models = models
        self.weights = weights
        self.model_names = list(models.keys())
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        
        for model_name, weight in zip(self.model_names, self.weights):
            pred = self.models[model_name].predict(X)
            predictions.append(weight * pred)
        
        return np.sum(predictions, axis=0)
    
    def __repr__(self):
        return f"EnsembleModel(models={self.model_names}, weights={[f'{w:.3f}' for w in self.weights]})"


def demo_complete_system():
    """Demonstrate the complete variable meta-AutoML system."""
    from sklearn.datasets import make_regression
    from pathlib import Path
    
    print("Complete Variable Meta-AutoML System Demo")
    print("=" * 50)
    
    # Check if we have the required components
    meta_model_path = "variable_length_meta_model.pth"
    
    if not Path(meta_model_path).exists():
        print(f"Pre-trained meta-model not found at: {meta_model_path}")
        print("The system will work without meta-learning (using default algorithms).")
        meta_model_path = None
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different configurations
    configs = [
        {
            "name": "Single Algorithm (No Optimization)",
            "config": {
                "meta_model_path": meta_model_path,
                "optimize_hyperparams": False,
                "use_ensemble": False
            }
        },
        {
            "name": "Single Algorithm (With Optimization)", 
            "config": {
                "meta_model_path": meta_model_path,
                "optimize_hyperparams": True,
                "use_ensemble": False,
                "n_trials": 20
            }
        },
        {
            "name": "Ensemble (Top 3)",
            "config": {
                "meta_model_path": meta_model_path,
                "optimize_hyperparams": True,
                "use_ensemble": True,
                "ensemble_top_k": 3,
                "n_trials": 15
            }
        }
    ]
    
    results = {}
    
    for config_info in configs:
        name = config_info["name"]
        config = config_info["config"]
        
        print(f"\n{'-'*50}")
        print(f"Testing: {name}")
        print(f"{'-'*50}")
        
        try:
            # Create Variable Meta-AutoML system
            automl = VariableMetaAutoML(**config, random_state=42)
            
            # Train
            start_time = time.time()
            training_results = automl.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate
            test_metrics = automl.evaluate(X_test, y_test)
            
            results[name] = {
                "training_time": training_time,
                "train_r2": training_results["train_r2"],
                "validation_r2": training_results["validation_r2"],
                "test_r2": test_metrics["r2_score"],
                "test_rmse": test_metrics["rmse"],
                "recommended_algorithm": training_results["dataset_analysis"]["recommended_algorithm"],
                "meta_learning": training_results["dataset_analysis"].get("meta_learning", "unknown")
            }
            
            print(f"Training time: {training_time:.2f}s")
            print(f"Test R²: {test_metrics['r2_score']:.4f}")
            print(f"Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"Recommended algorithm: {training_results['dataset_analysis']['recommended_algorithm']}")
            print(f"Meta-learning: {training_results['dataset_analysis'].get('meta_learning', 'unknown')}")
            
        except Exception as e:
            print(f"Error with configuration {name}: {e}")
            results[name] = None
    
    # Print comparison summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"{'Configuration':<35} {'Test R²':<10} {'Test RMSE':<12} {'Time (s)':<10} {'Meta-Learning':<15}")
    print("-" * 85)
    
    for name, result in results.items():
        if result:
            ml_type = result.get('meta_learning', 'N/A')[:14]
            print(f"{name:<35} {result['test_r2']:<10.4f} {result['test_rmse']:<12.2f} {result['training_time']:<10.1f} {ml_type:<15}")
        else:
            print(f"{name:<35} {'Failed':<10} {'Failed':<12} {'Failed':<10} {'Failed':<15}")


if __name__ == "__main__":
    demo_complete_system()
