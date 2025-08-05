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
from constants_gpu import algorithms
import openml
from scipy.stats import spearmanr
import joblib

import argparse
import warnings
import logging
import os
import psutil
import time
import signal
import sys
import pickle

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

# CPU Monitoring for cluster safety
class CPUMonitor:
    def __init__(self, max_cpu_percent=85, check_interval=10):
        self.max_cpu_percent = max_cpu_percent
        self.check_interval = check_interval
        self.monitoring = False
        self.start_time = time.time()
        
    def start_monitoring(self):
        """Start CPU monitoring in background"""
        self.monitoring = True
        print(f"Started CPU monitoring (max: {self.max_cpu_percent}%, check every {self.check_interval}s)")
        
    def check_cpu_usage(self):
        """Check if CPU usage is too high"""
        if not self.monitoring:
            return True
            
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > self.max_cpu_percent:
                print(f"\n⚠️  HIGH CPU USAGE DETECTED: {cpu_percent:.1f}%")
                print(f"⚠️  Memory usage: {memory_percent:.1f}%")
                print(f"⚠️  Stopping training to prevent cluster overload...")
                return False
                
            # Log every few minutes
            elapsed = time.time() - self.start_time
            if elapsed % 300 < self.check_interval:  # Every 5 minutes
                print(f"💻 System Status - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")
                
            return True
        except Exception as e:
            print(f"Warning: CPU monitoring failed: {e}")
            return True  # Continue if monitoring fails

# Global CPU monitor
cpu_monitor = CPUMonitor()

def signal_handler(signum, frame):
    """Handle interruption signals gracefully"""
    print(f"\n🛑 Received signal {signum}. Gracefully shutting down...")
    cpu_monitor.monitoring = False
    sys.exit(0)

# Set up signal handlers for cluster environments
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def parse_args():
    parser = argparse.ArgumentParser(description="Train or load meta-model for algorithm selection.")
    parser.add_argument('--model-path', type=str, default='meta_model.pth',
                        help='Path to save/load the meta-model')
    parser.add_argument('--split', action='store_true',
                        help='Whether to make predictions using the trained model')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='Use GPU for training if available')
    parser.add_argument('--max-cpu', type=int, default=85,
                        help='Maximum CPU usage percentage before stopping (for cluster safety)')
    parser.add_argument('--skip-tabpfn', action='store_true',
                        help='Skip TabPFN algorithm to avoid CPU overload')
    parser.add_argument('--cache-path', type=str, default='datasets_cache.pkl',
                        help='Path to save/load cached datasets')
    parser.add_argument('--force-reload', action='store_true',
                        help='Force reload datasets from OpenML (ignore cache)')
    parser.add_argument('--datasets-only', action='store_true',
                        help='Only download and cache datasets, skip algorithm evaluation')
    parser.add_argument('--max-datasets', type=int, default=600,
                        help='Maximum number of datasets to load')
    return parser.parse_args()


class MultiHeadRankingNetwork(nn.Module):
    """Multi-Head Ranking Network for algorithm ranking prediction."""
    
    def __init__(self, input_size, num_algorithms, dropout=0.3):
        super(MultiHeadRankingNetwork, self).__init__()
        
        self.num_algorithms = num_algorithms
        
        # Larger network for better GPU utilization
        hidden_size_1 = 256  # Increased from 64
        hidden_size_2 = 128  # Increased from 24
        hidden_size_3 = 64   # Additional layer
        
        # Shared Feature Extractor with BatchNorm for GPU
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),  # Better for GPU than LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.BatchNorm1d(hidden_size_3),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Algorithm-Specific Heads (one per algorithm)
        self.algorithm_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size_3, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(num_algorithms)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for better GPU training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
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
                     test_size: float = 0.1,
                     random_state: int = 42,
                     use_gpu: bool = True):
    """
    Train a Multi-Head Ranking Network to predict algorithm rankings from meta-features.
    
    Args:
        dataset_df: DataFrame containing meta-features and algorithm rankings
        save_path: Path to save the trained model
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        use_gpu: Whether to use GPU if available
    
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
    
    # GPU/CPU device selection
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        batch_size = 128  # Larger batch size for GPU
        
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
        torch.set_num_threads(4)  # Optimize for CPU
        batch_size = 32
    
    # Convert to PyTorch tensors - keep on CPU for DataLoader, move to device in training loop
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Create data loaders - only pin memory if using GPU and tensors are on CPU
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            pin_memory=(device.type == 'cuda'), num_workers=0)
    
    # Initialize model
    input_size = X_train.shape[1]
    num_algorithms = y_train.shape[1]
    algorithm_names = [col.replace('_rank', '') for col in rank_columns]
    
    model = MultiHeadRankingNetwork(input_size, num_algorithms).to(device)
    
    # Optimizer and scheduler (optimized for GPU/CPU)
    if device.type == 'cuda':
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)  # Higher LR for GPU
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        max_patience = 30
    else:
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.15)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.7)
        max_patience = 50
    
    # Training loop with CPU monitoring
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    print("Starting training...")
    for epoch in range(1000):
        # Check CPU usage periodically for cluster safety
        if epoch % 10 == 0 and not cpu_monitor.check_cpu_usage():
            print("🛑 Stopping training due to high CPU usage")
            break
            
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # Move batch to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
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
        
        # GPU memory cleanup
        if device.type == 'cuda' and epoch % 100 == 0:
            torch.cuda.empty_cache()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Move train tensors to device for evaluation
        X_train_tensor_eval = X_train_tensor.to(device)
        y_train_tensor_eval = y_train_tensor.to(device)
        
        # Get predictions
        train_scores = model(X_train_tensor_eval)
        test_scores = model(X_test_tensor)
        
        # Convert scores to ranks
        train_predicted_ranks = model.scores_to_ranks(train_scores).cpu().numpy()
        test_predicted_ranks = model.scores_to_ranks(test_scores).cpu().numpy()
        
        # Calculate metrics
        train_loss = combined_ranking_loss(train_scores, y_train_tensor_eval).item()
        test_loss = combined_ranking_loss(test_scores, y_test_tensor).item()
        
        # Calculate ranking correlations
        train_correlations = []
        test_correlations = []
        
        # Use CPU numpy arrays for correlation calculations
        y_train_np = y_train_tensor_eval.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
        
        for i in range(len(y_train_np)):
            corr, _ = spearmanr(y_train_np[i], train_predicted_ranks[i])
            if not np.isnan(corr):
                train_correlations.append(corr)
        
        for i in range(len(y_test_np)):
            corr, _ = spearmanr(y_test_np[i], test_predicted_ranks[i])
            if not np.isnan(corr):
                test_correlations.append(corr)
        
        avg_train_corr = np.mean(train_correlations) if train_correlations else 0
        avg_test_corr = np.mean(test_correlations) if test_correlations else 0
        
        # Calculate top-1 accuracy
        train_top1_correct = 0
        test_top1_correct = 0
        
        for i in range(len(y_train_np)):
            true_best = np.argmin(y_train_np[i])  # Rank 1 = best
            pred_best = np.argmin(train_predicted_ranks[i])
            if true_best == pred_best:
                train_top1_correct += 1
        
        for i in range(len(y_test_np)):
            true_best = np.argmin(y_test_np[i])
            pred_best = np.argmin(test_predicted_ranks[i])
            if true_best == pred_best:
                test_top1_correct += 1
        
        train_top1_accuracy = train_top1_correct / len(y_train_np)
        test_top1_accuracy = test_top1_correct / len(y_test_np)
        
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
    
    torch.save(model_package, save_path)
    print(f"\nMulti-Head Ranking Network saved to {save_path}")
    
    return model_package


def load_ranking_meta_model(model_path: str, use_gpu: bool = True):
    """Load a trained Multi-Head Ranking Network."""
    
    # Device selection
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Loading model on GPU")
    else:
        device = torch.device("cpu")
        print("💻 Loading model on CPU")
    
    # Load the saved tuple (model, model_package)
    saved_data = torch.load(model_path, map_location=device, weights_only=False)
    model, checkpoint = saved_data  # Unpack the tuple
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    print(f"Loaded Multi-Head Ranking Network:")
    print(f"  Model type: {checkpoint['metadata']['model_type']}")
    print(f"  Input size: {checkpoint['model_architecture']['input_size']}")
    print(f"  Number of algorithms: {checkpoint['model_architecture']['num_algorithms']}")
    print(f"  Number of features: {checkpoint['metadata']['num_features']}")
    print(f"  Test Spearman correlation: {checkpoint['training_info']['test_spearman_corr']:.4f}")
    print(f"  Test Top-1 accuracy: {checkpoint['training_info']['test_top1_accuracy']:.4f}")
    print(f"  Number of datasets: {checkpoint['metadata']['num_datasets']}")
    
    return model, checkpoint


def predict_algorithm_rankings(model, checkpoint, meta_features_df, actual_rankings_df=None, use_gpu=True):
    """
    Predict algorithm rankings for new datasets using trained Multi-Head Ranking Network.
    Returns discrete integer ranks from 1 to N.
    
    Args:
        model: Trained MultiHeadRankingNetwork model
        checkpoint: Model checkpoint containing preprocessing info
        meta_features_df: DataFrame with meta-features for new datasets
        actual_rankings_df: Optional DataFrame with actual rankings for comparison
        use_gpu: Whether to use GPU if available
    
    Returns:
        DataFrame with predicted rankings for each algorithm
    """
    # Device selection
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Move model to appropriate device
    model = model.to(device)
    
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


def save_datasets_to_disk(datasets: list, save_path: str = "datasets_cache.pkl"):
    """Save loaded datasets to disk for reuse."""
    print(f"💾 Saving {len(datasets)} datasets to {save_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(datasets, f)
    
    # Calculate file size
    file_size = os.path.getsize(save_path) / (1024**2)  # MB
    print(f"✅ Datasets saved successfully! File size: {file_size:.1f} MB")
    return save_path


def load_datasets_from_disk(load_path: str = "datasets_cache.pkl"):
    """Load pre-saved datasets from disk."""
    if not os.path.exists(load_path):
        print(f"❌ Dataset cache not found at {load_path}")
        return None
    
    print(f"📂 Loading datasets from {load_path}...")
    
    with open(load_path, 'rb') as f:
        datasets = pickle.load(f)
    
    print(f"✅ Loaded {len(datasets)} datasets from cache")
    return datasets


def fetch_and_cache_openml_datasets(
        NumberOfFeatures: tuple[int, int], 
        NumberOfInstances: tuple[int, int],
        NumberOfInstancesWithMissingValues: tuple[int, int],  
        NumberOfNumericFeatures: tuple[int, int], 
        NumberOfSymbolicFeatures: tuple[int, int],
        max_datasets: int = 10,
        cache_path: str = "datasets_cache.pkl") -> list:
    """Fetch datasets from OpenML and cache them locally."""
    print("🌐 Fetching datasets from OpenML...")
    
    loaded_datasets = []
    # Use the hardcoded filtered IDs (same as original function)
    filtered_ids = [191, 194, 195, 197, 200, 201, 203, 204, 206, 207, 208, 213, 215, 216, 217, 222, 223, 224, 227, 229, 231, 287, 294, 296, 298, 299, 301, 308, 315, 344, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 433, 434, 436, 437, 439, 440, 441, 442, 487, 498, 503, 505, 506, 510, 511, 516, 521, 524, 527, 530, 531, 534, 540, 542, 543, 546, 552, 558, 560, 562, 564, 566, 567, 568, 569, 570, 573, 574, 575, 576, 577, 578, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 595, 598, 600, 602, 603, 604, 605, 606, 607, 608, 610, 614, 615, 616, 618, 619, 620, 621, 622, 623, 625, 626, 627, 629, 630, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 650, 651, 652, 653, 654, 655, 657, 658, 666, 674, 691, 695, 696, 702, 1028, 1051, 1058, 1070, 1072, 1076, 1089, 1090, 1091, 1097, 1103, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1206, 1207, 1208, 1210, 1213, 1228, 1245, 1414, 4532, 4544, 4545, 4549, 40505, 40916, 41021, 41065, 41514, 41515, 41516, 41517, 41518, 41519, 41523, 41524, 41525, 41540, 41700, 41702, 41704, 41928, 41938, 41943, 41968, 41969, 41980, 42092, 42130, 42131, 42165, 42175, 42176, 42183, 42207, 42208, 42224, 42225, 42351, 42352, 42363, 42366, 42367, 42368, 42369, 42464, 42495, 42559, 42563, 42570, 42571, 42635, 42636, 42688, 42705, 42712, 42713, 42720, 42721, 42724, 42729, 42730, 42731, 42821, 42822, 42876, 42889, 42900, 42931, 43070, 43071, 43090, 43093, 43121, 43123, 43127, 43144, 43174, 43180, 43190, 43192, 43194, 43198, 43250, 43252, 43256, 43379, 43403, 43440, 43441, 43442, 43452, 43465, 43549, 43606, 43617, 43660, 43665, 43672, 43675, 43682, 43686, 43714, 43747, 43812, 43871, 43873, 43874, 43888, 43889, 43926, 43927, 43928, 43939, 43943, 43944, 43954, 43959, 43965, 43967, 43978, 43981, 43982, 43983, 43984, 43986, 43987, 43989, 43990, 43991, 43992, 43994, 43995, 43997, 44001, 44002, 44005, 44006, 44007, 44008, 44009, 44011, 44012, 44014, 44018, 44019, 44022, 44023, 44027, 44029, 44039, 44042, 44043, 44044, 44045, 44046, 44047, 44048, 44049, 44050, 44051, 44052, 44053, 44054, 44057, 44058, 44059, 44060, 44061, 44062, 44063, 44064, 44065, 44066, 44067, 44069, 44132, 44133, 44134, 44135, 44136, 44137, 44139, 44143, 44144, 44147, 44148, 44150, 44179, 44180, 44187, 44191, 44192, 44194, 44201, 44202, 44203, 44212, 44223, 44231, 44235, 44236, 44243, 44252, 44269, 44270, 44280, 44296, 44310, 44311, 44793, 44794, 44961, 44962, 44963, 44964, 44965, 44966, 44967, 44969, 44971, 44972, 44973, 44974, 44975, 44976, 44978, 44979, 44981, 44983, 44986, 44988, 44989, 44990, 44993, 44994, 44998, 45012, 45013, 45014, 45015, 45016, 45017, 45029, 45030, 45031, 45032, 45040, 45041, 45044, 45046, 45052, 45053, 45054, 45055, 45056, 45057, 45061, 45062, 45064, 45065, 45066, 45070, 45071, 45073, 45074, 45075, 45076, 45078, 45079, 45080, 45081, 45082, 45108, 45536, 45537, 45538, 45539, 45540, 45559, 45580, 45586, 45592, 45647, 45720, 45742, 45743, 45744, 45761, 46132, 46133, 46135, 46138, 46139, 46141, 46142, 46144, 46148, 46153, 46156, 46159, 46162, 46169, 46265, 46266, 46267, 46268, 46269, 46271, 46272, 46273, 46274, 46275, 46283, 46285, 46286, 46288, 46293, 46294, 46297, 46299, 46300, 46301, 46307, 46316, 46320, 46328, 46337, 46340, 46356, 46357, 46501, 46533, 46538, 46555, 46585, 46586, 46587, 46588, 46589, 46591, 46592, 46594, 46598, 46604, 46607, 46610, 46612, 46613, 46615, 46618, 46619, 46620, 46634, 46635, 46638, 46641, 46642, 46643, 46644, 46645, 46646, 46657, 46658, 46659, 46668, 46672, 46674, 46681, 46710, 46722, 46723, 46726, 46728, 46731, 46735, 46738, 46747, 46748, 46753, 46754, 46756, 46757, 46763, 46783, 46883, 46884, 46890, 46892, 46897, 46914, 46923, 46928, 46942, 46949, 46953, 46957, 46961, 46964, 46985, 46989, 46996, 46998, 47001, 47002]

    # Limit to max_datasets
    filtered_ids = filtered_ids[:max_datasets]
    
    print(f"📊 Fetching {len(filtered_ids)} datasets from OpenML...")
    
    for i, did in enumerate(filtered_ids):
        try:
            print(f"  Fetching dataset {i+1}/{len(filtered_ids)} (ID: {did})...", end=" ")
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
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print(f"✅ Successfully fetched {len(loaded_datasets)} datasets from OpenML")
    
    # Save to cache for future use
    if loaded_datasets:
        save_datasets_to_disk(loaded_datasets, cache_path)
    
    print(f"Loaded {len(loaded_datasets)} datasets from OpenML")
    print("Dataset IDs:", [ds['dataset_id'] for ds in loaded_datasets])
    
    return loaded_datasets


     
def load_openml_datasets(
        NumberOfFeatures: tuple[int, int], 
        NumberOfInstances: tuple[int, int],
        NumberOfInstancesWithMissingValues: tuple[int, int],  
        NumberOfNumericFeatures: tuple[int, int], 
        NumberOfSymbolicFeatures: tuple[int, int],
        max_datasets: int = 10,
        cache_path: str = "datasets_cache.pkl",
        force_reload: bool = False) -> list:
    """
    Load datasets from OpenML with caching support.
    Returns a list of loaded datasets.
    
    Args:
        NumberOfFeatures: Range of number of features
        NumberOfInstances: Range of number of instances
        NumberOfInstancesWithMissingValues: Range of instances with missing values
        NumberOfNumericFeatures: Range of numeric features
        NumberOfSymbolicFeatures: Range of symbolic features
        max_datasets: Maximum number of datasets to load
        cache_path: Path to save/load cached datasets
        force_reload: If True, ignore cache and reload from OpenML
    
    Returns:
        List of dataset dictionaries with X, y, categorical_indicator, attribute_names, dataset_id
    """
    
    # Try to load from cache first (unless force_reload is True)
    if not force_reload and os.path.exists(cache_path):
        print(f"📂 Found dataset cache at {cache_path}")
        cached_datasets = load_datasets_from_disk(cache_path)
        if cached_datasets is not None:
            print(f"🎯 Using {len(cached_datasets)} cached datasets")
            return cached_datasets[:max_datasets]  # Respect max_datasets limit
    
    print("🌐 Loading datasets from OpenML...")
    
    # Original OpenML loading code - use the fetch_and_cache function
    loaded_datasets = fetch_and_cache_openml_datasets(
        NumberOfFeatures=NumberOfFeatures,
        NumberOfInstances=NumberOfInstances,
        NumberOfInstancesWithMissingValues=NumberOfInstancesWithMissingValues,
        NumberOfNumericFeatures=NumberOfNumericFeatures,
        NumberOfSymbolicFeatures=NumberOfSymbolicFeatures,
        max_datasets=max_datasets,
        cache_path=cache_path
    )

    return loaded_datasets


def load_synthetic_datasets(max_datasets: int = 10) -> list:
    """Generate synthetic datasets for testing when OpenML is not available."""
    from sklearn.datasets import make_regression
    import numpy as np
    
    datasets = []
    np.random.seed(42)
    
    print(f"🔬 Generating {max_datasets} synthetic datasets...")
    
    for i in range(max_datasets):
        # Vary dataset characteristics
        n_samples = np.random.randint(100, 2000)
        n_features = np.random.randint(5, 50)
        noise = np.random.uniform(0.01, 0.3)
        
        # Generate regression dataset
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=42 + i
        )
        
        # Convert to DataFrame
        feature_names = [f"feature_{j}" for j in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")
        
        datasets.append({
            "X": X_df,
            "y": y_series,
            "categorical_indicator": [False] * n_features,
            "attribute_names": feature_names,
            "dataset_id": f"synthetic_{i}"
        })
    
    print(f"✅ Generated {len(datasets)} synthetic datasets")
    return datasets
    
       
def algorithms_eval(algorithms: list, datasets: list, use_gpu: bool = True, skip_tabpfn: bool = False):
    """
    Evaluate a list of algorithms on a list of datasets (from load_openml_dataset).
    Returns a list of records with meta-features and algorithm performances.
    
    Args:
        algorithms: List of algorithm instances to evaluate
        datasets: List of datasets from load_openml_datasets
        use_gpu: Whether to use GPU for compatible algorithms
        skip_tabpfn: Whether to skip TabPFN to avoid CPU overload
    """
    from sklearn.model_selection import train_test_split

    records = []
    
    # Filter algorithms based on options
    if skip_tabpfn:
        algorithms = [algo for algo in algorithms if 'TabPFN' not in algo.__class__.__name__]
        print(f"⚠️  Skipping TabPFN due to --skip-tabpfn flag")
    
    # Configure GPU-enabled algorithms
    gpu_algorithms = []
    for algo in algorithms:
        algo_copy = type(algo)()  # Create new instance
        
        # Configure for GPU if available and requested
        if use_gpu and torch.cuda.is_available():
            if hasattr(algo_copy, 'device'):
                # TabPFN and other PyTorch models expect 'cuda' not 'gpu'
                if 'TabPFN' in algo_copy.__class__.__name__:
                    algo_copy.device = 'cuda'
                else:
                    algo_copy.device = 'gpu'
                print(f"🚀 Configured {algo_copy.__class__.__name__} for GPU")
            elif hasattr(algo_copy, 'tree_method'):  # XGBoost
                algo_copy.tree_method = 'gpu_hist'
                algo_copy.gpu_id = 0
                print(f"🚀 Configured {algo_copy.__class__.__name__} for GPU")
            elif 'LightGBM' in algo_copy.__class__.__name__:
                algo_copy.device = 'gpu'
                algo_copy.gpu_platform_id = 0
                algo_copy.gpu_device_id = 0
                print(f"🚀 Configured {algo_copy.__class__.__name__} for GPU")
        
        gpu_algorithms.append(algo_copy)
    
    print(f"🔍 Evaluating {len(gpu_algorithms)} algorithms on {len(datasets)} datasets")

    for i, ds in enumerate(datasets):
        # Check CPU usage before processing each dataset
        if not cpu_monitor.check_cpu_usage():
            print("🛑 Stopping dataset processing due to high CPU usage")
            break
            
        dataset_id = ds["dataset_id"]
        
        X, y = ds["X"], ds["y"]
        print(f"→ Processing dataset {i+1}/{len(datasets)} (ID: {dataset_id}) Dataset shape: {X.shape}, target shape: {y.shape}")

        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(y):
            continue
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Build preprocessor on all data (to avoid leakage, you can fit only on train)
        try:
            from automl.pre_processor import build_preprocessor
            preprocessor = build_preprocessor(X)
            print("============= Preprocessor built inside meta_trainer.py =============")
        except ImportError:
            from pre_processor import build_preprocessor
            preprocessor = build_preprocessor(X)
            print("============= Preprocessor built inside meta_trainer.py =============")
        
        # 1) extract meta-features
        try:
            from automl.meta_features import extract_meta_features
            meta = extract_meta_features(X, y)
        except ImportError:
            from meta_features import extract_meta_features
            meta = extract_meta_features(X, y)

        print("============== Meta-features extracted ==============")

        # 2) evaluate each algorithm with timeout and CPU monitoring
        scores = {}
        for Algo in gpu_algorithms:
            # Check CPU before each algorithm
            if not cpu_monitor.check_cpu_usage():
                print(f"⚠️  Skipping remaining algorithms due to high CPU usage")
                break
                
            name = Algo.__class__.__name__
            print(f"   • Training {name}...", end=" ", flush=True)
            
            start_time = time.time()
            try:
                model = Pipeline([
                    ("preproc", preprocessor),
                    ("model", Algo)
                ])
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                
                elapsed_time = time.time() - start_time
                if elapsed_time > 300:  # 5 minutes timeout warning
                    print(f"⚠️  {name} took {elapsed_time:.1f}s (long training time)")
                    
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
            **meta,
            **{f"{n}_rank": i+1 for i, n in enumerate(sorted_names)},
        }
        records.append(record)

    # 5) save to CSV
    df = pd.DataFrame(records)
    df.to_csv("meta_features.csv", index=False)
    
    return records
    


def main():
    args = parse_args()
    
    # Initialize CPU monitoring for cluster safety
    cpu_monitor.max_cpu_percent = args.max_cpu
    cpu_monitor.start_monitoring()
    
    predict_split_flag = args.split
    use_gpu = args.use_gpu
    skip_tabpfn = args.skip_tabpfn
    
    # Check GPU availability
    if use_gpu and torch.cuda.is_available():
        print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 GPU not available or disabled, using CPU")
        use_gpu = False
    
    print(f"⚙️  Configuration:")
    print(f"   • Split flag: {predict_split_flag}")
    print(f"   • Use GPU: {use_gpu}")
    print(f"   • Skip TabPFN: {skip_tabpfn}")
    print(f"   • Max CPU usage: {args.max_cpu}%")
    print(f"   • Cache path: {args.cache_path}")
    print(f"   • Force reload: {args.force_reload}")
    print(f"   • Max datasets: {args.max_datasets}")
    print(f"   • Datasets only: {args.datasets_only}")
    
    # Step 1: Load/cache datasets
    print("\n" + "="*50)
    print("LOADING DATASETS")
    print("="*50)
    
    try:
        openml_datasets = load_openml_datasets(
            NumberOfFeatures=(10, 500000000),
            NumberOfInstances=(10, 5000000000),
            NumberOfInstancesWithMissingValues=(0, 2000000),
            NumberOfNumericFeatures=(1, 1500000000),
            NumberOfSymbolicFeatures=(1, 1500000000),
            max_datasets=args.max_datasets,
            cache_path=args.cache_path,
            force_reload=args.force_reload
        )
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        print("🔄 Falling back to synthetic datasets for testing...")
        # Fallback to synthetic datasets
        # try:
        #     openml_datasets = load_synthetic_datasets(max_datasets=min(args.max_datasets, 20))
        # except Exception as e2:
        #     print(f"❌ Synthetic dataset generation also failed: {e2}")
        #     print("🛑 Cannot proceed without datasets")
        #     return
    
    print(f"📊 Total datasets available: {len(openml_datasets)}")
    
    # If only caching datasets, exit here
    if args.datasets_only:
        print("🎯 Dataset caching complete. Exiting.")
        return
    
    # Step 2: Run algorithm evaluation on cached datasets
    print("\n" + "="*50)
    print("EVALUATING ALGORITHMS")
    print("="*50)
    
    # You can now work with a subset for testing if needed
    test_datasets = openml_datasets[:20] if len(openml_datasets) > 20 else openml_datasets
    print(f"🧪 Using {len(test_datasets)} datasets for algorithm evaluation")
    raise Exception("Skipping algorithm evaluation for now")
    records = algorithms_eval(algorithms=algorithms, datasets=test_datasets, 
                             use_gpu=use_gpu, skip_tabpfn=skip_tabpfn)

    # get records from meta_features.csv
    
    # try:
    #     records = pd.read_csv("meta_features.csv").to_dict(orient='records')
    #     records = pd.DataFrame(records)
    #     print(f"Loaded {len(records)} records from meta_features.csv")
    # except FileNotFoundError:
    #     print("No meta_features.csv found. Please run the dataset evaluation first.")
    
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
            model_package = train_meta_model(train_df, save_path=args.model_path, use_gpu=use_gpu)
        else:
            print("Using full predictions for meta-model training")
            model_package = train_meta_model(meta_df, save_path=args.model_path, use_gpu=use_gpu)
        
        # # Step 3: Load model and make predictions
        # print("\n" + "="*50)
        # print("LOADING AND TESTING META-MODEL")
        # print("="*50)
        
        # # Load the model
        # models, checkpoint = load_ranking_meta_model(args.model_path, use_gpu=use_gpu)

        # rank_columns = [col for col in meta_df.columns if col.endswith('_rank')]
        # meta_feature_columns = [col for col in meta_df.columns if not col.endswith('_rank')]

        # if predict_split_flag:
        #     # Use the test set for predictions
        #     test_meta_features = test_df[meta_feature_columns]
        #     test_actual_rankings = test_df[rank_columns]
        # else:
        #     test_meta_features = meta_df[meta_feature_columns].head(2)  # Take first 2 datasets
        #     test_actual_rankings = meta_df[rank_columns].head(2)  # Actual rankings for comparison

        # predictions = predict_algorithm_rankings(models, checkpoint, test_meta_features, test_actual_rankings, use_gpu=use_gpu)
        
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
