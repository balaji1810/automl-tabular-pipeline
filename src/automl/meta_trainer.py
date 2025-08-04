from __future__ import annotations
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from automl.constants import algorithms
import openml
from scipy.stats import spearmanr
import joblib

import argparse
import warnings
import logging
import os

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*LightGBM.*")
warnings.filterwarnings("ignore", message=".*No further splits.*")
warnings.filterwarnings("ignore", message=".*Stopped training.*")
os.environ['LIGHTGBM_VERBOSITY'] = '-1'  # Suppress LightGBM verbose output

# Suppress other common ML warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Additional LightGBM suppression
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)


predict_split_flag = False  # Flag to control whether to split predictions or not

def parse_args():
    parser = argparse.ArgumentParser(description="Train or load meta-model for algorithm selection.")
    parser.add_argument('--model-path', type=str, default='meta_model.pth',
                        help='Path to save/load the meta-model')
    parser.add_argument('--split', action='store_true',
                        help='Whether to make predictions using the trained model')

    return parser.parse_args()


class MultiHeadRankingNetwork(nn.Module):
    """Multi-Head Ranking Network for algorithm ranking prediction."""
    
    def __init__(self, input_size, num_algorithms, dropout=0.3):
        super(MultiHeadRankingNetwork, self).__init__()
        
        self.num_algorithms = num_algorithms
        
        # Shared Feature Extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 64), # TODO: changed from 128 to 64
            nn.LayerNorm(64),  # Changed from BatchNorm1d to LayerNorm to handle small batches
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 24), # TODO: changed from 64 to 24
            nn.LayerNorm(24),  # Changed from BatchNorm1d to LayerNorm to handle small batches
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # nn.Linear(64, 32), # TODO: commenting out to reduce complexity
            # nn.BatchNorm1d(32),
            # nn.ReLU()
        )
        
        # Algorithm-Specific Heads (one per algorithm)
        self.algorithm_heads = nn.ModuleList([
            nn.Linear(24, 1) for _ in range(num_algorithms) # TODO: Reduced from 32->1 to 24->1
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Extract shared features
        shared_features = self.shared_layers(x)
        
        # Get raw scores from each algorithm head
        raw_scores = []
        for head in self.algorithm_heads:
            score = head(shared_features)
            raw_scores.append(score)
        
        # Concatenate scores: [batch_size, num_algorithms]
        raw_scores = torch.cat(raw_scores, dim=1)
        
        return raw_scores
    
    def scores_to_ranks(self, raw_scores, temperature=1.0):
        """Convert raw scores to ranks using differentiable ranking."""
        # Use negative scores so higher scores get lower ranks (rank 1 = best)
        # Apply temperature for softer ranking
        neg_scores = -raw_scores / temperature
        
        # Use argsort to get ranking indices
        # argsort gives indices that would sort the array
        # argsort of argsort gives ranks (0-based)
        ranks = torch.argsort(torch.argsort(neg_scores, dim=1, descending=False), dim=1) + 1
        
        return ranks.float()


def ranking_mse_loss(predicted_scores, true_ranks):
    """Primary Loss: Ranking-Aware MSE."""
    # Convert scores to ranks
    predicted_ranks = torch.argsort(torch.argsort(-predicted_scores, dim=1), dim=1) + 1
    return F.mse_loss(predicted_ranks.float(), true_ranks.float())


def pairwise_ranking_loss(predicted_scores, true_ranks):
    """Secondary Loss: Pairwise Ranking Loss."""
    batch_size, n_algos = predicted_scores.shape
    loss = 0.0
    count = 0
    
    for i in range(n_algos):
        for j in range(i + 1, n_algos):
            # If algorithm i should rank better than j (lower rank number)
            should_be_better = (true_ranks[:, i] < true_ranks[:, j]).float()
            
            # Score difference (higher score should mean better rank)
            score_diff = predicted_scores[:, i] - predicted_scores[:, j]
            
            # Hinge loss: penalize when score ordering doesn't match rank ordering
            loss += F.relu(1.0 - should_be_better * score_diff + (1 - should_be_better) * score_diff).mean()
            count += 1
    
    return loss / count if count > 0 else torch.tensor(0.0)


def combined_ranking_loss(predicted_scores, true_ranks, alpha=0.7):
    """Combined ranking loss function."""
    primary_loss = ranking_mse_loss(predicted_scores, true_ranks)
    secondary_loss = pairwise_ranking_loss(predicted_scores, true_ranks)
    
    return alpha * primary_loss + (1 - alpha) * secondary_loss


def train_meta_model(dataset_df: pd.DataFrame, 
                     save_path: str = "meta_model.pth",
                     test_size: float = 0.2,
                     random_state: int = 42):
    """
    Train a Multi-Head Ranking Network to predict algorithm rankings from meta-features.
    
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
    
    print(f"Training Multi-Head Ranking Network for algorithm ranking prediction")
    print(f"Meta-features: {X.shape[1]} features")
    print(f"Algorithm rankings: {y.shape[1]} algorithms")
    print(f"Dataset size: {X.shape[0]} datasets")
    print(f"Algorithm names: {[col.replace('_rank', '') for col in rank_columns]}")
    
    # Handle missing values if any
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    # Normalize meta-features (important for neural networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    # Split data
    if len(X) > 10:  # Only split if we have enough data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y.values, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y.values, y.values
        print("Warning: Using all data for both training and testing due to small sample size")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Set device (CPU optimized)
    device = torch.device("cpu")
    torch.set_num_threads(4)  # Optimize for CPU
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # train_loader = DataLoader(train_dataset, batch_size=min(32, len(X_train)//2 + 1), shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # TODO: changed from 32 to 64 for better convergence
    
    # Initialize model
    input_size = X_train.shape[1]
    num_algorithms = y_train.shape[1]
    algorithm_names = [col.replace('_rank', '') for col in rank_columns]
    
    model = MultiHeadRankingNetwork(input_size, num_algorithms).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.15) # TODO: changed weight_decay from 0.01 to 0.15 and learning rate from 0.001 to 0.005
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.7) # TODO: changed patience from 20 to 15 and factor from 0.5 to 0.7
    
    # Training loop
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 50 # TODO: changed this from 100 to 50 for stronger early stopping
    train_losses = []
    
    print("Starting training...")
    for epoch in range(1000):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            raw_scores = model(batch_X)
            
            # Calculate combined ranking loss
            loss = combined_ranking_loss(raw_scores, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Get predictions
        train_scores = model(X_train_tensor)
        test_scores = model(X_test_tensor)
        
        # Convert scores to ranks
        train_predicted_ranks = model.scores_to_ranks(train_scores).cpu().numpy()
        test_predicted_ranks = model.scores_to_ranks(test_scores).cpu().numpy()
        
        # Calculate metrics
        train_loss = combined_ranking_loss(train_scores, y_train_tensor).item()
        test_loss = combined_ranking_loss(test_scores, y_test_tensor).item()
        
        # Calculate ranking correlations
        train_correlations = []
        test_correlations = []
        
        for i in range(len(y_train)):
            corr, _ = spearmanr(y_train[i], train_predicted_ranks[i])
            if not np.isnan(corr):
                train_correlations.append(corr)
        
        for i in range(len(y_test)):
            corr, _ = spearmanr(y_test[i], test_predicted_ranks[i])
            if not np.isnan(corr):
                test_correlations.append(corr)
        
        avg_train_corr = np.mean(train_correlations) if train_correlations else 0
        avg_test_corr = np.mean(test_correlations) if test_correlations else 0
        
        # Calculate top-1 accuracy
        train_top1_correct = 0
        test_top1_correct = 0
        
        for i in range(len(y_train)):
            true_best = np.argmin(y_train[i])  # Rank 1 = best
            pred_best = np.argmin(train_predicted_ranks[i])
            if true_best == pred_best:
                train_top1_correct += 1
        
        for i in range(len(y_test)):
            true_best = np.argmin(y_test[i])
            pred_best = np.argmin(test_predicted_ranks[i])
            if true_best == pred_best:
                test_top1_correct += 1
        
        train_top1_accuracy = train_top1_correct / len(y_train)
        test_top1_accuracy = test_top1_correct / len(y_test)
        
        print(f"\nMulti-Head Ranking Network Training Results:")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Train Spearman Correlation: {avg_train_corr:.4f}")
        print(f"Test Spearman Correlation: {avg_test_corr:.4f}")
        print(f"Train Top-1 Accuracy: {train_top1_accuracy:.4f}")
        print(f"Test Top-1 Accuracy: {test_top1_accuracy:.4f}")
    
    # Save model package
    model_package = {
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_size': input_size,
            'num_algorithms': num_algorithms,
            'model_class': 'MultiHeadRankingNetwork'
        },
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
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_spearman_corr': avg_train_corr,
            'test_spearman_corr': avg_test_corr,
            'train_top1_accuracy': train_top1_accuracy,
            'test_top1_accuracy': test_top1_accuracy,
            'num_epochs': epoch + 1,
            'best_loss': best_loss,
        },
        'metadata': {
            'num_datasets': len(dataset_df),
            'num_features': input_size,
            'num_algorithms': num_algorithms,
            'model_type': 'MultiHeadRankingNetwork',
        }
    }
    
    torch.save((model, model_package), save_path)
    print(f"\nMulti-Head Ranking Network saved to {save_path}")
    
    return model_package


def load_ranking_meta_model(model_path: str):
    """Load a trained Multi-Head Ranking Network."""
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Rebuild model architecture
    arch = checkpoint['model_architecture']
    model = MultiHeadRankingNetwork(
        arch['input_size'], 
        arch['num_algorithms']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded Multi-Head Ranking Network:")
    print(f"  Model type: {checkpoint['metadata']['model_type']}")
    print(f"  Input size: {arch['input_size']}")
    print(f"  Number of algorithms: {arch['num_algorithms']}")
    print(f"  Number of features: {checkpoint['metadata']['num_features']}")
    print(f"  Test Spearman correlation: {checkpoint['training_info']['test_spearman_corr']:.4f}")
    print(f"  Test Top-1 accuracy: {checkpoint['training_info']['test_top1_accuracy']:.4f}")
    print(f"  Number of datasets: {checkpoint['metadata']['num_datasets']}")
    
    return model, checkpoint


def predict_algorithm_rankings(model, checkpoint, meta_features_df, actual_rankings_df=None):
    """
    Predict algorithm rankings for new datasets using trained Multi-Head Ranking Network.
    Returns discrete integer ranks from 1 to N.
    
    Args:
        model: Trained MultiHeadRankingNetwork model
        checkpoint: Model checkpoint containing preprocessing info
        meta_features_df: DataFrame with meta-features for new datasets
        actual_rankings_df: Optional DataFrame with actual rankings for comparison
    
    Returns:
        DataFrame with predicted rankings for each algorithm
    """
    device = torch.device("cpu")
    
    # Preprocess features using saved scaler
    scaler_mean = checkpoint['preprocessing']['scaler_mean']
    scaler_scale = checkpoint['preprocessing']['scaler_scale']
    feature_names = checkpoint['preprocessing']['feature_names']
    
    # Ensure features are in correct order and handle missing values
    X = meta_features_df[feature_names]
    # .fillna(pd.Series(scaler_mean, index=feature_names))
    X_scaled = (X.values - scaler_mean) / scaler_scale
    
    # Predict rankings
    algorithm_names = checkpoint['algorithm_info']['algorithm_names']
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Get raw scores from the model
        raw_scores = model(X_tensor)
        
        # Convert scores to ranks
        predicted_ranks = model.scores_to_ranks(raw_scores).cpu().numpy()
    
    # Create organized DataFrame with actual and predicted ranks side by side
    predictions_df = pd.DataFrame(index=meta_features_df.index)
    
    # Add algorithm recommendation (best = rank 1)
    best_algorithms = []
    for i in range(len(predicted_ranks)):
        row_predictions = predicted_ranks[i]
        # Find algorithm with rank 1 (best)
        best_algo_indices = np.where(row_predictions == 1)[0]
        if len(best_algo_indices) > 0:
            best_algo = algorithm_names[best_algo_indices[0]]
        else:
            # If no rank 1 (shouldn't happen), pick the lowest rank
            best_algo_idx = np.argmin(row_predictions.astype(float))
            best_algo = algorithm_names[best_algo_idx]
        best_algorithms.append(best_algo)

    # If actual rankings are provided, organize them side by side with predictions
    if actual_rankings_df is not None:
        # Ensure indices match
        actual_rankings_aligned = actual_rankings_df.reindex(meta_features_df.index)
        
        # Add columns for each algorithm: actual_rank, predicted_rank (side by side)
        for i, algo_name in enumerate(algorithm_names):
            # Add actual rank column
            actual_col = f"{algo_name}_rank"
            if actual_col in actual_rankings_aligned.columns:
                predictions_df[f"{algo_name}_actual_rank"] = actual_rankings_aligned[actual_col].astype(int)
            
            # Add predicted rank column (right next to actual) - convert to integers
            predictions_df[f"{algo_name}"] = predicted_ranks[:, i].astype(int)
        
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
        predictions_df['recommended_algorithm'] = best_algorithms
    else:
        # If no actual rankings, just add predicted ranks - convert to integers
        for i, algo_name in enumerate(algorithm_names):
            predictions_df[f"{algo_name}"] = predicted_ranks[:, i].astype(int)
        predictions_df['recommended_algorithm'] = best_algorithms

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
    datasets = datasets[~datasets['format'].isin(['Sparse_ARFF', 'sparse_arff'])]  # Exclude sparse formats

    filtered = datasets[
        (datasets['NumberOfFeatures'] >= NumberOfFeatures[0]) &
        (datasets['NumberOfFeatures'] <= NumberOfFeatures[1]) &
        (datasets['NumberOfInstances'] >= NumberOfInstances[0]) &
        (datasets['NumberOfInstances'] <= NumberOfInstances[1]) &
        (datasets['NumberOfInstancesWithMissingValues'] >= NumberOfInstancesWithMissingValues[0]) &
        (datasets['NumberOfInstancesWithMissingValues'] <= NumberOfInstancesWithMissingValues[1]) &
        (datasets['NumberOfNumericFeatures'] >= NumberOfNumericFeatures[0]) &
        (datasets['NumberOfNumericFeatures'] <= NumberOfNumericFeatures[1]) &
        (datasets['NumberOfSymbolicFeatures'] >= NumberOfSymbolicFeatures[0]) &
        (datasets['NumberOfSymbolicFeatures'] <= NumberOfSymbolicFeatures[1])]

    loaded_datasets = []
    filtered_ids = filtered['did'].tolist()[:max_datasets]

    # filtered_ids = [3967]
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
        dataset_id = ds["dataset_id"]
        
        X, y = ds["X"], ds["y"]
        print(f"→ Processing dataset {i+1}/{len(datasets)} (ID: {dataset_id}) Dataset shape: {X.shape}, target shape: {y.shape}")

        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(y):
            # print("   • Target is not numeric, skipping dataset")
            continue
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # print(f"   • Train shape: {X_train.head()}, Test shape: {X_test.head()}")
        
        # Build preprocessor on all data (to avoid leakage, you can fit only on train)
        from pre_processor import build_preprocessor
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
    df.to_csv("meta_features.csv", index=False)
    
    # # Save updated failing datasets list
    # failing_df = pd.DataFrame({'failing_dataset_ids': failing_datasets})
    # failing_df.to_csv("failing_datasets_list.csv", index=False)
    
    # print(f"\nDataset Processing Summary:")
    # print(f"• Total datasets attempted: {len(datasets)}")
    # print(f"• Successfully processed: {len(records)}")
    # print(f"• Failed to process: {len(datasets) - len(records)}")
    # print(f"• Success rate: {len(records)/len(datasets)*100:.1f}%")
    # print(f"• Updated failing datasets list saved to: failing_datasets_list.csv")
    # print(f"• Total failing datasets: {len(failing_datasets)}")
    
    return records
    


def main():
    args = parse_args()
    
    predict_split_flag = args.split
    
    print(f"Split flag set to: {predict_split_flag}")
    # Example usage of the meta-learning pipeline
    
    # Step 1: Generate meta-learning dataset
    openml_datasets = load_openml_datasets(
            NumberOfFeatures=(10, 5000),
            NumberOfInstances=(10, 50000),
            NumberOfInstancesWithMissingValues=(0, 20000),
            NumberOfNumericFeatures=(1, 15000),
            NumberOfSymbolicFeatures=(1, 15000),
            max_datasets=50
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
