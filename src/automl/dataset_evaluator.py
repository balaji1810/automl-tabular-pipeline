"""
Dataset Evaluator for Meta-Learning

This module evaluates all datasets in the data folder with the portfolio of algorithms
and creates a training dataset for the meta-learning model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
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
# from tabpfn import TabPFNRegressor  # Optional

# ML utilities
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def get_algorithm_portfolio():
    """Define the portfolio of algorithms to evaluate."""
    return {
        "MLPRegressor": MLPRegressor(random_state=42, max_iter=300),
        "XGBRegressor": XGBRegressor(random_state=42, n_jobs=-1),
        "LGBMRegressor": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "RandomForestRegressor": RandomForestRegressor(random_state=42, n_jobs=-1),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "BayesianRidge": BayesianRidge(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(),
        # "TabPFNRegressor": TabPFNRegressor()  # Uncomment if available
    }


def detect_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Detect numerical and categorical columns."""
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline."""
    num_cols, cat_cols = detect_column_types(X)
    
    transformers = []
    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", RobustScaler())
        ])
        transformers.append(("numerical", num_pipeline, num_cols))
    
    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("categorical", cat_pipeline, cat_cols))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0
    )
    
    return preprocessor


def load_dataset(dataset_path: str, fold: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load a specific dataset and fold."""
    base_path = Path(dataset_path) / str(fold)
    
    X_train = pd.read_parquet(base_path / "X_train.parquet")
    y_train = pd.read_parquet(base_path / "y_train.parquet").iloc[:, 0]
    X_test = pd.read_parquet(base_path / "X_test.parquet")
    y_test = pd.read_parquet(base_path / "y_test.parquet").iloc[:, 0]
    
    return X_train, y_train, X_test, y_test


def evaluate_algorithm_on_dataset(algorithm, algorithm_name: str, X_train: pd.DataFrame, 
                                y_train: pd.Series, X_test: pd.DataFrame, 
                                y_test: pd.Series, preprocessor: ColumnTransformer) -> Dict[str, Any]:
    """Evaluate a single algorithm on a dataset."""
    try:
        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", algorithm)
        ])
        
        # Fit and predict
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            "r2_score": float(r2),
            "rmse": float(rmse),
            "status": "success"
        }
        
    except Exception as e:
        print(f"  Error with {algorithm_name}: {str(e)}")
        return {
            "r2_score": -1.0,
            "rmse": float('inf'),
            "status": "failed"
        }


def find_all_datasets(data_folder: str) -> List[str]:
    """Find all dataset folders in the data directory."""
    data_path = Path(data_folder)
    datasets = []
    
    for item in data_path.iterdir():
        if item.is_dir():
            # Check if it has fold structure
            if any((item / str(i)).exists() for i in range(1, 11)):
                datasets.append(item.name)
    
    return sorted(datasets)


def evaluate_all_datasets(data_folder: str = "data", max_folds: int = 10) -> pd.DataFrame:
    """
    Evaluate all datasets with all algorithms and create meta-learning dataset.
    
    Returns a DataFrame with columns:
    - dataset_name: Name of the dataset
    - fold: Fold number
    - dataset_path: Path to X_train.parquet and y_train.parquet
    - best_algorithm: Algorithm that performed best
    - algorithm_rankings: JSON string with all algorithm scores and rankings
    """
    
    datasets = find_all_datasets(data_folder)
    algorithms = get_algorithm_portfolio()
    
    print(f"Found {len(datasets)} datasets: {datasets}")
    print(f"Evaluating with {len(algorithms)} algorithms: {list(algorithms.keys())}")
    
    results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_path = Path(data_folder) / dataset_name
        
        # Check available folds
        available_folds = []
        for fold in range(1, max_folds + 1):
            fold_path = dataset_path / str(fold)
            if (fold_path / "X_train.parquet").exists() and (fold_path / "y_train.parquet").exists():
                available_folds.append(fold)
        
        print(f"Available folds: {available_folds}")
        
        for fold in available_folds:
            print(f"\n--- Fold {fold} ---")
            
            try:
                # Load data
                X_train, y_train, X_test, y_test = load_dataset(str(dataset_path), fold)
                print(f"Data shape: X_train={X_train.shape}, y_train={y_train.shape}")
                
                # Build preprocessor
                preprocessor = build_preprocessor(pd.concat([X_train, X_test], ignore_index=True))
                
                # Evaluate all algorithms
                algorithm_scores = {}
                
                for algorithm_name, algorithm in algorithms.items():
                    print(f"  Evaluating {algorithm_name}...", end=" ")
                    
                    results_dict = evaluate_algorithm_on_dataset(
                        algorithm, algorithm_name, X_train, y_train, X_test, y_test, preprocessor
                    )
                    
                    algorithm_scores[algorithm_name] = results_dict
                    print(f"R² = {results_dict['r2_score']:.4f}")
                
                # Find best algorithm and create rankings
                valid_scores = {name: scores['r2_score'] for name, scores in algorithm_scores.items() 
                              if scores['status'] == 'success' and scores['r2_score'] > -1}
                
                if valid_scores:
                    # Sort by R² score (descending)
                    sorted_algorithms = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)
                    best_algorithm = sorted_algorithms[0][0]
                    
                    # Create rankings (1 = best)
                    rankings = {}
                    for rank, (alg_name, score) in enumerate(sorted_algorithms, 1):
                        rankings[alg_name] = {
                            "rank": rank,
                            "r2_score": score,
                            "rmse": algorithm_scores[alg_name]['rmse']
                        }
                    
                    # Add failed algorithms with worst rank
                    worst_rank = len(sorted_algorithms) + 1
                    for alg_name, scores in algorithm_scores.items():
                        if scores['status'] == 'failed':
                            rankings[alg_name] = {
                                "rank": worst_rank,
                                "r2_score": scores['r2_score'],
                                "rmse": scores['rmse']
                            }
                    
                    print(f"  Best algorithm: {best_algorithm} (R² = {valid_scores[best_algorithm]:.4f})")
                    
                    # Create record
                    record = {
                        "dataset_name": dataset_name,
                        "fold": fold,
                        "dataset_path": str(dataset_path / str(fold)),
                        "X_train_path": str(dataset_path / str(fold) / "X_train.parquet"),
                        "y_train_path": str(dataset_path / str(fold) / "y_train.parquet"),
                        "X_test_path": str(dataset_path / str(fold) / "X_test.parquet"),
                        "y_test_path": str(dataset_path / str(fold) / "y_test.parquet"),
                        "best_algorithm": best_algorithm,
                        "best_r2_score": valid_scores[best_algorithm],
                        "n_samples": len(X_train),
                        "n_features": X_train.shape[1],
                        "algorithm_rankings": json.dumps(rankings, indent=2)
                    }
                    
                    results.append(record)
                    
                else:
                    print(f"  Warning: No valid algorithm results for {dataset_name}, fold {fold}")
                    
            except Exception as e:
                print(f"  Error processing {dataset_name}, fold {fold}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if not df.empty:
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total dataset-fold combinations evaluated: {len(df)}")
        print(f"Unique datasets: {df['dataset_name'].nunique()}")
        print(f"Algorithm performance summary:")
        
        # Count best algorithm occurrences
        best_algo_counts = df['best_algorithm'].value_counts()
        for algo, count in best_algo_counts.items():
            print(f"  {algo}: {count} times best ({count/len(df)*100:.1f}%)")
    
    return df


def save_meta_dataset(df: pd.DataFrame, output_path: str = "meta_learning_dataset.csv"):
    """Save the meta-learning dataset."""
    df.to_csv(output_path, index=False)
    print(f"\nMeta-learning dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


def main():
    """Main function to create meta-learning dataset."""
    print("Creating Meta-Learning Dataset for Algorithm Selection")
    print("=" * 60)
    
    # Evaluate all datasets
    meta_df = evaluate_all_datasets(data_folder="data", max_folds=10)
    
    if not meta_df.empty:
        # Save the dataset
        save_meta_dataset(meta_df, "src/automl/meta_learning_dataset.csv")
        
        # Display sample
        print(f"\nSample of meta-learning dataset:")
        print(meta_df[['dataset_name', 'fold', 'best_algorithm', 'best_r2_score', 'n_samples', 'n_features']].head(10))
        
    else:
        print("No valid evaluation results. Please check your data folder and datasets.")


if __name__ == "__main__":
    main()
