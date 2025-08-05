from __future__ import annotations
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from automl.algorithms import algorithms
import openml
from scipy.stats import spearmanr
import joblib

import argparse
import warnings
import logging


predict_split_flag = False  # Flag to control whether to split predictions or not

def parse_args():
    parser = argparse.ArgumentParser(description="Train or load meta-model for algorithm selection.")
    parser.add_argument('--model-path', type=str, default='meta_model.pth',
                        help='Path to save/load the meta-model')
    parser.add_argument('--split', action='store_true',
                        help='Whether to make predictions using the trained model')

    return parser.parse_args()


def train_meta_model(dataset_df: pd.DataFrame, 
                     save_path: str = "meta_model.pth",
                     test_size: float = 0.2,
                     random_state: int = 42):
    """
    Train a meta-model to predict algorithm rankings from meta-features using XGBoost.
    
    Args:
        dataset_df: DataFrame containing meta-features and algorithm rankings
        save_path: Path to save the trained model
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Trained model package with metadata
    """
    
    # Separate meta-features (X) from algorithm rankings (y)
    rank_columns = [col for col in dataset_df.columns if col.endswith('_rank')]
    meta_feature_columns = [col for col in dataset_df.columns if not col.endswith('_rank')]
    
    if len(rank_columns) == 0:
        raise ValueError("No ranking columns found. Expected columns ending with '_rank'")

    X = dataset_df[meta_feature_columns].copy()
    y = dataset_df[rank_columns].copy()
    
    print(f"Training XGBoost meta-model for algorithm ranking prediction")
    print(f"Meta-features: {X.shape[1]} features")
    print(f"Algorithm rankings: {y.shape[1]} algorithms")
    print(f"Dataset size: {X.shape[0]} datasets")
    print(f"Algorithm names: {[col.replace('_rank', '') for col in rank_columns]}")
    
    # Handle missing values if any
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    # Normalize meta-features (important for neural networks, helpful for XGBoost too)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    # Split data
    if len(X) > 50:  # Only split if we have enough data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y.values, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y.values, y.values
        print("Warning: Using all data for both training and testing due to small sample size")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train separate XGBoost classifier for each algorithm
    algorithm_names = [col.replace('_rank', '') for col in rank_columns]
    num_algorithms = len(algorithm_names)
    num_ranks = len(rank_columns)  # Number of possible ranks
    
    # Handle class label mapping for XGBoost
    # XGBoost requires consecutive class labels starting from 0
    def map_ranks_to_consecutive_classes(ranks):
        """Map arbitrary rank values to consecutive class labels 0, 1, 2, ..."""
        unique_ranks = sorted(np.unique(ranks))
        rank_to_class = {rank: i for i, rank in enumerate(unique_ranks)}
        return np.array([rank_to_class[rank] for rank in ranks]), len(unique_ranks)
    
    # Train individual XGBoost models for each algorithm
    models = {}
    train_predictions = {}
    test_predictions = {}
    class_mappers = {}  # Store mapping info for each algorithm
    
    print("Training XGBoost models for each algorithm...")
    for i, algo_name in enumerate(algorithm_names):
        print(f"  Training model for {algo_name}...")
        
        # Get this algorithm's rank data
        algo_y_train_original = y_train[:, i]
        algo_y_test_original = y_test[:, i]
        
        # Map ranks to consecutive classes for XGBoost
        algo_y_train_mapped, n_classes_train = map_ranks_to_consecutive_classes(algo_y_train_original)
        algo_y_test_mapped, n_classes_test = map_ranks_to_consecutive_classes(algo_y_test_original)
        
        # Store mapping info
        unique_train_ranks = sorted(np.unique(algo_y_train_original))
        unique_test_ranks = sorted(np.unique(algo_y_test_original))
        all_unique_ranks = sorted(np.unique(np.concatenate([algo_y_train_original, algo_y_test_original])))
        
        class_mappers[algo_name] = {
            'rank_to_class': {rank: i for i, rank in enumerate(all_unique_ranks)},
            'class_to_rank': {i: rank for i, rank in enumerate(all_unique_ranks)},
            'n_classes': len(all_unique_ranks)
        }
        
        # Re-map both train and test using the combined mapping
        combined_mapper = class_mappers[algo_name]['rank_to_class']
        algo_y_train_mapped = np.array([combined_mapper[rank] for rank in algo_y_train_original])
        algo_y_test_mapped = np.array([combined_mapper[rank] for rank in algo_y_test_original])
        
        # XGBoost Classifier for this algorithm's rank prediction
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='mlogloss',
            verbosity=0  # Reduce output
        )
        
        model.fit(X_train, algo_y_train_mapped)
        models[algo_name] = model
        
        # Get predictions and convert back to original ranks
        train_pred_classes = model.predict(X_train)
        test_pred_classes = model.predict(X_test)
        
        # Convert predicted classes back to original rank values
        class_to_rank = class_mappers[algo_name]['class_to_rank']
        train_pred_ranks = np.array([class_to_rank[cls] for cls in train_pred_classes])
        test_pred_ranks = np.array([class_to_rank[cls] for cls in test_pred_classes])
        
        train_predictions[algo_name] = train_pred_ranks
        test_predictions[algo_name] = test_pred_ranks
        
        # Calculate accuracy for this algorithm
        train_acc = np.mean(train_pred_ranks == algo_y_train_original)
        test_acc = np.mean(test_pred_ranks == algo_y_test_original)
        print(f"    {algo_name} - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
        print(f"      Unique ranks: {all_unique_ranks}, Mapped to classes: {list(range(len(all_unique_ranks)))}")
    
    # Combine predictions for overall evaluation
    train_pred_matrix = np.column_stack([train_predictions[name] for name in algorithm_names])
    test_pred_matrix = np.column_stack([test_predictions[name] for name in algorithm_names])
    
    print(f"Training prediction matrix shape: {train_pred_matrix.shape}")
    print(f"Test prediction matrix shape: {test_pred_matrix.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Calculate ranking correlations (Spearman's rank correlation)
    train_correlations = []
    test_correlations = []
    
    # Fixed correlation calculation
    for i in range(len(y_train)):
        # Get the true and predicted ranks for dataset i
        true_ranks = y_train[i]  # Shape: (num_algorithms,)
        pred_ranks = train_pred_matrix[i]  # Shape: (num_algorithms,)
        
        # Ensure both are 1D arrays
        if hasattr(true_ranks, 'flatten'):
            true_ranks = true_ranks.flatten()
        if hasattr(pred_ranks, 'flatten'):
            pred_ranks = pred_ranks.flatten()
        
        # Calculate Spearman correlation
        try:
            if len(true_ranks) == len(pred_ranks) and len(true_ranks) > 1:
                corr, _ = spearmanr(true_ranks, pred_ranks)
                if not np.isnan(corr) and not np.isinf(corr):
                    train_correlations.append(corr)
        except Exception as e:
            print(f"Warning: Spearman correlation failed for train sample {i}: {e}")
            print(f"  true_ranks shape: {true_ranks.shape if hasattr(true_ranks, 'shape') else len(true_ranks)}")
            print(f"  pred_ranks shape: {pred_ranks.shape if hasattr(pred_ranks, 'shape') else len(pred_ranks)}")
    
    for i in range(len(y_test)):
        # Get the true and predicted ranks for dataset i
        true_ranks = y_test[i]  # Shape: (num_algorithms,)
        pred_ranks = test_pred_matrix[i]  # Shape: (num_algorithms,)
        
        # Ensure both are 1D arrays
        if hasattr(true_ranks, 'flatten'):
            true_ranks = true_ranks.flatten()
        if hasattr(pred_ranks, 'flatten'):
            pred_ranks = pred_ranks.flatten()
        
        # Calculate Spearman correlation
        try:
            if len(true_ranks) == len(pred_ranks) and len(true_ranks) > 1:
                corr, _ = spearmanr(true_ranks, pred_ranks)
                if not np.isnan(corr) and not np.isinf(corr):
                    test_correlations.append(corr)
        except Exception as e:
            print(f"Warning: Spearman correlation failed for test sample {i}: {e}")
            print(f"  true_ranks shape: {true_ranks.shape if hasattr(true_ranks, 'shape') else len(true_ranks)}")
            print(f"  pred_ranks shape: {pred_ranks.shape if hasattr(pred_ranks, 'shape') else len(pred_ranks)}")
    
    avg_train_corr = np.mean(train_correlations) if train_correlations else 0
    avg_test_corr = np.mean(test_correlations) if test_correlations else 0
    
    print(f"Calculated {len(train_correlations)} valid train correlations")
    print(f"Calculated {len(test_correlations)} valid test correlations")
        
    # Calculate top-1 accuracy (best algorithm prediction)
    train_top1_correct = 0
    test_top1_correct = 0
    
    for i in range(len(y_train)):
        true_best = np.argmin(y_train[i])  # Rank 1 = best (minimum rank value)
        pred_best = np.argmin(train_pred_matrix[i])
        if true_best == pred_best:
            train_top1_correct += 1
    
    for i in range(len(y_test)):
        true_best = np.argmin(y_test[i])
        pred_best = np.argmin(test_pred_matrix[i])
        if true_best == pred_best:
            test_top1_correct += 1
    
    train_top1_accuracy = train_top1_correct / len(y_train)
    test_top1_accuracy = test_top1_correct / len(y_test)
    
    print(f"\nXGBoost Training Results:")
    print(f"Train Spearman Correlation: {avg_train_corr:.4f}")
    print(f"Test Spearman Correlation: {avg_test_corr:.4f}")
    print(f"Train Top-1 Accuracy: {train_top1_accuracy:.4f}")
    print(f"Test Top-1 Accuracy: {test_top1_accuracy:.4f}")
    
    # Save model package using joblib
    model_package = {
        'models': models,
        'class_mappers': class_mappers,  # Add class mappers for prediction
        'preprocessing': {
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'feature_names': list(X.columns),
        },
        'algorithm_info': {
            'algorithm_names': algorithm_names,
            'rank_columns': rank_columns,
        },
        'training_info': {
            'train_spearman_corr': avg_train_corr,
            'test_spearman_corr': avg_test_corr,
            'train_top1_accuracy': train_top1_accuracy,
            'test_top1_accuracy': test_top1_accuracy,
            'num_algorithms': num_algorithms,
            'num_ranks': num_ranks,
        },
        'metadata': {
            'num_datasets': len(dataset_df),
            'num_features': X.shape[1],
            'num_algorithms': num_algorithms,
            'model_type': 'XGBoost',
        }
    }
    
    joblib.dump(model_package, save_path)
    print(f"\nXGBoost meta-model saved to {save_path}")
    
    return model_package


def load_ranking_meta_model(model_path: str):
    """Load a trained XGBoost ranking meta-model."""
    checkpoint = joblib.load(model_path)
    
    print(f"Loaded XGBoost ranking model:")
    print(f"  Model type: {checkpoint['metadata']['model_type']}")
    print(f"  Number of algorithms: {checkpoint['metadata']['num_algorithms']}")
    print(f"  Number of features: {checkpoint['metadata']['num_features']}")
    print(f"  Test Spearman correlation: {checkpoint['training_info']['test_spearman_corr']:.4f}")
    print(f"  Test Top-1 accuracy: {checkpoint['training_info']['test_top1_accuracy']:.4f}")
    print(f"  Number of datasets: {checkpoint['metadata']['num_datasets']}")
    
    return checkpoint['models'], checkpoint


def predict_algorithm_rankings(models, checkpoint, meta_features_df, actual_rankings_df=None):
    """
    Predict algorithm rankings for new datasets using trained XGBoost models.
    Returns discrete integer ranks from 1 to N.
    
    Args:
        models: Dictionary of trained XGBoost models (one per algorithm)
        checkpoint: Model checkpoint containing preprocessing info
        meta_features_df: DataFrame with meta-features for new datasets
        actual_rankings_df: Optional DataFrame with actual rankings for comparison
    
    Returns:
        DataFrame with predicted rankings for each algorithm
    """
    
    # Preprocess features using saved scaler
    scaler_mean = checkpoint['preprocessing']['scaler_mean']
    scaler_scale = checkpoint['preprocessing']['scaler_scale']
    feature_names = checkpoint['preprocessing']['feature_names']
    
    # Ensure features are in correct order and handle missing values
    X = meta_features_df[feature_names].fillna(pd.Series(scaler_mean, index=feature_names))
    X_scaled = (X.values - scaler_mean) / scaler_scale
    
    # Predict rankings for each algorithm
    algorithm_names = checkpoint['algorithm_info']['algorithm_names']
    class_mappers = checkpoint['class_mappers']
    predicted_ranks = []
    
    for algo_name in algorithm_names:
        model = models[algo_name]
        # Predict classes (0-based consecutive)
        predicted_classes = model.predict(X_scaled)
        
        # Convert classes back to original rank values
        class_to_rank = class_mappers[algo_name]['class_to_rank']
        algo_predictions = np.array([class_to_rank[cls] for cls in predicted_classes])
        predicted_ranks.append(algo_predictions)
    
    # Transpose to get [num_datasets, num_algorithms] shape
    predicted_ranks = np.column_stack(predicted_ranks)
    
    # Convert to DataFrame with algorithm names
    predictions_df = pd.DataFrame(
        predicted_ranks, 
        columns=[f"{name}_predicted_rank" for name in algorithm_names],
        index=meta_features_df.index
    )
    
    # Add algorithm recommendation (best = rank 1)
    best_algorithms = []
    for i in range(len(predictions_df)):
        row_predictions = predictions_df.iloc[i]
        # Find algorithm with rank 1 (best)
        best_algo_indices = np.where(row_predictions.values == 1)[0]
        if len(best_algo_indices) > 0:
            best_algo = algorithm_names[best_algo_indices[0]]
        else:
            # If no rank 1 (shouldn't happen), pick the lowest rank
            best_algo_idx = np.argmin(row_predictions.values.astype(float))
            best_algo = algorithm_names[best_algo_idx]
        best_algorithms.append(best_algo)
    
    predictions_df['recommended_algorithm'] = best_algorithms

    # If actual rankings are provided, add them to the dataframe
    if actual_rankings_df is not None:
        # Ensure indices match
        actual_rankings_aligned = actual_rankings_df.reindex(meta_features_df.index)
        
        # Add actual rankings with "_actual_rank" suffix
        for col in actual_rankings_aligned.columns:
            if col.endswith('_rank'):
                algo_name = col.replace('_rank', '')
                predictions_df[f"{algo_name}_actual_rank"] = actual_rankings_aligned[col]
        
        # Add actual best algorithm
        actual_best_algorithms = []
        for i in range(len(actual_rankings_aligned)):
            row_actual = actual_rankings_aligned.iloc[i]
            # Find algorithm with rank 1 (best) in actual rankings
            best_actual_indices = np.where(row_actual.values == 1)[0]
            if len(best_actual_indices) > 0:
                best_actual_col = actual_rankings_aligned.columns[best_actual_indices[0]]
                best_actual_algo = best_actual_col.replace('_rank', '')
            else:
                # If no rank 1, pick the lowest rank
                best_actual_col = row_actual.idxmin()
                best_actual_algo = best_actual_col.replace('_rank', '')
            actual_best_algorithms.append(best_actual_algo)
        
        predictions_df['actual_best_algorithm'] = actual_best_algorithms

    # Save to CSV with enhanced information
    predictions_df.to_csv("meta_preds.csv", index=True)
    
    if actual_rankings_df is not None:
        print(f"Predictions saved with actual rankings! Both predicted and actual ranks included.")
        
        # Also create a comparison summary
        comparison_summary = []
        for i in range(len(predictions_df)):
            row = predictions_df.iloc[i]
            predicted_best = row['recommended_algorithm']
            actual_best = row['actual_best_algorithm'] if 'actual_best_algorithm' in row else 'N/A'
            match = predicted_best == actual_best
            
            comparison_summary.append({
                'dataset_index': i,
                'predicted_best': predicted_best,
                'actual_best': actual_best,
                'correct_prediction': match
            })
        
        comparison_df = pd.DataFrame(comparison_summary)
        comparison_df.to_csv("prediction_comparison.csv", index=False)
        
        accuracy = comparison_df['correct_prediction'].mean()
        print(f"Top-1 accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
    else:
        print(f"Predictions saved! All ranks are discrete integers from 1 to {len(algorithm_names)}")
    
    return predictions_df


     
def load_openml_datasets(
        NumberOfFeatures: tuple[int, int], 
        NumberOfInstances: tuple[int, int],
        NumberOfInstancesWithMissingValues: tuple[int, int],  
        NumberOfNumericFeatures: tuple[int, int], 
        NumberOfSymbolicFeatures: tuple[int, int],
        max_datasets: int = 10) -> list:
    """Load datasets from OpenML based on specified criteria.
    Returns a list of loaded datasets.
    """
    datasets = openml.datasets.list_datasets(output_format='dataframe')
    datasets = datasets[(datasets['NumberOfClasses'] == 0)]

    filtered = datasets[
        (datasets['NumberOfFeatures'] <= NumberOfFeatures[1]) &
        (datasets['NumberOfFeatures'] >= NumberOfFeatures[0]) &
        (datasets['NumberOfInstances'] <= NumberOfInstances[1]) &
        (datasets['NumberOfInstances'] >= NumberOfInstances[0]) &
        (datasets['NumberOfInstancesWithMissingValues'] <= NumberOfInstancesWithMissingValues[1]) &
        (datasets['NumberOfInstancesWithMissingValues'] >= NumberOfInstancesWithMissingValues[0]) &
        (datasets['NumberOfNumericFeatures'] <= NumberOfNumericFeatures[1]) &
        (datasets['NumberOfNumericFeatures'] >= NumberOfNumericFeatures[0]) &
        (datasets['NumberOfSymbolicFeatures'] <= NumberOfSymbolicFeatures[1]) &
        (datasets['NumberOfSymbolicFeatures'] >= NumberOfSymbolicFeatures[0])]

    loaded_datasets = []
    filtered_ids = filtered['did'].tolist()[:max_datasets]

    for did in filtered_ids:
        dataset = openml.datasets.get_dataset(did)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        loaded_datasets.append({
            "X": X,
            "y": y,
            "categorical_indicator": categorical_indicator,
            "attribute_names": attribute_names,
            "dataset_id": did
            })

    print(f"Loaded {len(loaded_datasets)} datasets from OpenML")
    print("Dataset IDs:", filtered_ids)

    return loaded_datasets
    
       
def algorithms_eval(algorithms: list, datasets: list):
    """
    Evaluate a list of algorithms on a list of datasets (from load_openml_dataset).
    Returns a list of records with meta-features and algorithm performances.
    """
    from sklearn.model_selection import train_test_split

    records = []

    for i, ds in enumerate(datasets):
        print(f"→ Processing dataset {i+1}/{len(datasets)}")
        X, y = ds["X"], ds["y"]
        print(f"   • Dataset shape: {X.shape}, target shape: {y.shape}")

        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(y):
            print("   • Target is not numeric, skipping dataset")
            continue
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        print(f"   • Train shape: {X_train.head()}, Test shape: {X_test.head()}")
        
        # Build preprocessor on all data (to avoid leakage, you can fit only on train)
        from automl.pre_processor_old import build_preprocessor
        preprocessor = build_preprocessor(X)
        print("============= Preprocessor built inside meta_trainer.py =============")
        # 1) extract meta-features
        from meta_features import extract_meta_features
        meta = extract_meta_features(X, y)

        print("============== Meta-features extracted ==============")

        # 2) evaluate each algorithm
        scores = {}
        for Algo in algorithms:
            name = Algo.__class__.__name__
            print(f"   • Training {name}...", end=" ", flush=True)
            try:
                model = Pipeline([
                    ("preproc", preprocessor),
                    ("model", Algo)
                ])
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
            except Exception as e:
                print(f"[Error: {e}]")
                dummy = DummyRegressor()
                dummy.fit(X_train, y_train)
                r2 = r2_score(y_test, dummy.predict(X_test))

            scores[name] = float(r2)
            print(f"R²={r2:.4f}")

        # 3) compute ranks (1 = best)
        sorted_names = sorted(scores, key=lambda k: -scores[k])

        # 4) assemble record
        record = {
            # "dataset_index": i,
            # "dataset_id": ds["dataset_id"],
            **meta,
            **{f"{n}_rank": i+1 for i, n in enumerate(sorted_names)},
            # **{f"{n}_r2": scores[n] for n in scores}, # full evaluation scores
        }
        records.append(record)

    # 5) save to CSV
    df = pd.DataFrame(records)
    df.to_csv("meta_Y.csv", index=False)
    print("\nSaved meta-dataset to meta_records.csv")
    
    return records
    


def main():
    # Suppress LightGBM warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
    logging.getLogger('lightgbm').setLevel(logging.ERROR)
    args = parse_args()
    
    predict_split_flag = args.split
    
    print(f"Split flag set to: {predict_split_flag}")
    # Example usage of the meta-learning pipeline
    
    # Step 1: Generate meta-learning dataset
    openml_datasets = load_openml_datasets(
            NumberOfFeatures=(1, 1023),
            NumberOfInstances=(1, 100),
            NumberOfInstancesWithMissingValues=(0, 10000),
            NumberOfNumericFeatures=(1, 10000),
            NumberOfSymbolicFeatures=(1, 10000),
            max_datasets=500
        )
    records = algorithms_eval(algorithms=algorithms, datasets=openml_datasets)
    
    # Step 2: Train meta-model (if we have enough data)
    if len(records) > 0:
        meta_df = pd.DataFrame(records)
        print(f"\nMeta-learning dataset shape: {meta_df.shape}")
        print(f"Columns: {list(meta_df.columns)}")
        
        # Train meta-model
        print("\n" + "="*50)
        print("TRAINING META-MODEL")
        print("="*50)
        if predict_split_flag:
            print("Using split predictions for meta-model training")
            # Split predictions into train/test sets
            train_df, test_df = train_test_split(meta_df, test_size=0.2, random_state=42)
            model_package = train_meta_model(train_df, save_path=args.model_path)
        else:
            print("Using full predictions for meta-model training")
            model_package = train_meta_model(meta_df, save_path=args.model_path)
        
        # Step 3: Load model and make predictions
        print("\n" + "="*50)
        print("LOADING AND TESTING META-MODEL")
        print("="*50)
        
        # Load the model
        models, checkpoint = load_ranking_meta_model(args.model_path)

        rank_columns = [col for col in meta_df.columns if col.endswith('_rank')]
        meta_feature_columns = [col for col in meta_df.columns if not col.endswith('_rank')]

        if predict_split_flag:
            # Use the test set for predictions
            test_meta_features = test_df[meta_feature_columns]
            test_actual_rankings = test_df[rank_columns]
        else:
            test_meta_features = meta_df[meta_feature_columns].head(2)  # Take first 2 datasets
            test_actual_rankings = meta_df[rank_columns].head(2)  # Actual rankings for comparison

        predictions = predict_algorithm_rankings(models, checkpoint, test_meta_features, test_actual_rankings)
        
        # print(f"\nPredictions for {len(test_meta_features)} datasets:")
        # print(predictions)

        
    else:
        print("No datasets processed successfully. Cannot train meta-model.")
    
    # Example of how to use the model for new datasets:
    # print("\n" + "="*50)
    # print("EXAMPLE: Using meta-model for new dataset")
    # print("="*50)
    # print("""
    # # To use the trained model for algorithm selection on a new dataset:
    
    # # 1. Extract meta-features from your new dataset
    # from automl.meta_features import extract_meta_features
    # meta_features = extract_meta_features(X_new, y_new)
    # meta_features_df = pd.DataFrame([meta_features])
    
    # # 2. Load the trained XGBoost model
    # models, checkpoint = load_ranking_meta_model('meta_model.pth')
    
    # # 3. Predict algorithm rankings
    # predictions = predict_algorithm_rankings(models, checkpoint, meta_features_df)
    
    # # 4. Get the recommended algorithm
    # best_algorithm = predictions['recommended_algorithm'].iloc[0]
    # print(f"Recommended algorithm: {best_algorithm}")
    
    # # 5. Get full ranking
    # ranking_cols = [col for col in predictions.columns if col.endswith('_predicted_rank')]
    # full_ranking = predictions[ranking_cols].iloc[0].sort_values()
    # print(f"Full algorithm ranking: {full_ranking}")
    # """)
        

if __name__ == "__main__":
    main()
