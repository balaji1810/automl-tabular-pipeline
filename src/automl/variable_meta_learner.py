"""
Variable-Length Dataset Meta-Learning Model

This implements a meta-learning model that can take entire datasets as input
(with variable features) and suggest the best algorithm. Based on the paper
"Learning dataset representation for automatic machine learning algorithm selection".
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DatasetEncoder(nn.Module):
    """
    Neural network that encodes variable-length datasets into fixed-size representations.
    Uses attention mechanism to handle variable number of features.
    """
    
    def __init__(self, hidden_dim: int = 128, output_dim: int = 64, max_features: int = 1000):
        super(DatasetEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_features = max_features
        
        # Feature embedding layer
        self.feature_embedding = nn.Linear(1, hidden_dim)
        
        # Positional encoding for features
        self.positional_encoding = nn.Parameter(torch.randn(max_features, hidden_dim))
        
        # Transformer encoder for handling variable-length features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Attention pooling to get fixed-size representation
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Final projection layers
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Target encoding layers
        self.target_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # mean, std, min, max
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Combined representation
        self.final_encoder = nn.Sequential(
            nn.Linear(output_dim + hidden_dim // 4, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, features: torch.Tensor, feature_mask: torch.Tensor, 
                target_stats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the dataset encoder.
        
        Args:
            features: [batch_size, max_features, feature_samples] - sampled feature values
            feature_mask: [batch_size, max_features] - mask for valid features
            target_stats: [batch_size, 4] - target statistics (mean, std, min, max)
        
        Returns:
            encoded_dataset: [batch_size, output_dim] - fixed-size dataset representation
        """
        batch_size, max_features, feature_samples = features.shape
        
        # Embed features: average pooling over samples for each feature
        feature_means = features.mean(dim=2, keepdim=True)  # [batch_size, max_features, 1]
        embedded_features = self.feature_embedding(feature_means).squeeze(-2)  # [batch_size, max_features, hidden_dim]
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:max_features].unsqueeze(0)  # [1, max_features, hidden_dim]
        embedded_features = embedded_features + pos_encoding
        
        # Apply mask for padding
        embedded_features = embedded_features * feature_mask.unsqueeze(-1)
        
        # Transformer encoding
        encoded_features = self.transformer(embedded_features)  # [batch_size, max_features, hidden_dim]
        
        # Attention pooling to get fixed-size representation
        # Use the first feature as query for attention pooling
        query = encoded_features[:, :1, :]  # [batch_size, 1, hidden_dim]
        pooled_features, _ = self.attention_pooling(query, encoded_features, encoded_features)
        pooled_features = pooled_features.squeeze(1)  # [batch_size, hidden_dim]
        
        # Project to output dimension
        feature_representation = self.projection(pooled_features)  # [batch_size, output_dim]
        
        # Encode target statistics
        target_representation = self.target_encoder(target_stats)  # [batch_size, hidden_dim//4]
        
        # Combine feature and target representations
        combined = torch.cat([feature_representation, target_representation], dim=1)
        final_representation = self.final_encoder(combined)
        
        return final_representation


class AlgorithmPredictor(nn.Module):
    """
    Neural network that predicts algorithm rankings from dataset representations.
    """
    
    def __init__(self, input_dim: int, num_algorithms: int, hidden_dim: int = 128):
        super(AlgorithmPredictor, self).__init__()
        self.num_algorithms = num_algorithms
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_algorithms)
        )
    
    def forward(self, dataset_representation: torch.Tensor) -> torch.Tensor:
        """
        Predict algorithm scores from dataset representation.
        
        Args:
            dataset_representation: [batch_size, input_dim]
        
        Returns:
            algorithm_scores: [batch_size, num_algorithms] - higher scores = better algorithms
        """
        return self.predictor(dataset_representation)


class MetaLearningModel(nn.Module):
    """
    Complete meta-learning model combining dataset encoder and algorithm predictor.
    """
    
    def __init__(self, num_algorithms: int, encoder_hidden_dim: int = 128, 
                 encoder_output_dim: int = 64, predictor_hidden_dim: int = 128,
                 max_features: int = 1000):
        super(MetaLearningModel, self).__init__()
        
        self.dataset_encoder = DatasetEncoder(
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            max_features=max_features
        )
        
        self.algorithm_predictor = AlgorithmPredictor(
            input_dim=encoder_output_dim,
            num_algorithms=num_algorithms,
            hidden_dim=predictor_hidden_dim
        )
        
        self.num_algorithms = num_algorithms
    
    def forward(self, features: torch.Tensor, feature_mask: torch.Tensor, 
                target_stats: torch.Tensor) -> torch.Tensor:
        """Forward pass of the complete model."""
        dataset_repr = self.dataset_encoder(features, feature_mask, target_stats)
        algorithm_scores = self.algorithm_predictor(dataset_repr)
        return algorithm_scores


class VariableDatasetDataset(Dataset):
    """
    PyTorch Dataset for handling variable-length datasets.
    """
    
    def __init__(self, meta_df: pd.DataFrame, max_features: int = 1000, 
                 feature_samples: int = 100, transform_target: bool = True):
        self.meta_df = meta_df.copy()
        self.max_features = max_features
        self.feature_samples = feature_samples
        self.algorithm_encoder = LabelEncoder()
        
        # Encode algorithms
        self.algorithm_encoder.fit(meta_df['best_algorithm'])
        if transform_target:
            self.meta_df['algorithm_label'] = self.algorithm_encoder.transform(meta_df['best_algorithm'])
        
        self.algorithm_names = self.algorithm_encoder.classes_
        self.num_algorithms = len(self.algorithm_names)
        
        print(f"Dataset created with {len(self.meta_df)} samples")
        print(f"Algorithms: {list(self.algorithm_names)}")
    
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        
        # Load dataset
        try:
            X_train = pd.read_parquet(row['X_train_path'])
            y_train = pd.read_parquet(row['y_train_path']).iloc[:, 0]
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            # Return dummy data
            return self._create_dummy_sample()
        
        # Sample features and prepare tensors
        features, feature_mask = self._process_features(X_train)
        target_stats = self._process_target(y_train)
        
        # Get algorithm rankings from JSON
        rankings = json.loads(row['algorithm_rankings'])
        algorithm_labels = self._process_rankings(rankings)
        
        return {
            'features': features,
            'feature_mask': feature_mask,
            'target_stats': target_stats,
            'algorithm_labels': algorithm_labels,
            'best_algorithm': row['algorithm_label'] if 'algorithm_label' in row else 0
        }
    
    def _process_features(self, X: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process features into fixed-size tensor with mask."""
        # Convert to numeric only
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Sample feature values
        n_features = min(X_numeric.shape[1], self.max_features)
        feature_samples = min(len(X_numeric), self.feature_samples)
        
        # Create feature tensor
        features = torch.zeros(self.max_features, self.feature_samples)
        feature_mask = torch.zeros(self.max_features)
        
        if n_features > 0:
            # Sample rows and columns
            sampled_X = X_numeric.iloc[:feature_samples, :n_features]
            
            # Fill features tensor
            for i in range(n_features):
                col_values = sampled_X.iloc[:, i].fillna(sampled_X.iloc[:, i].mean())
                
                # Pad or truncate to feature_samples
                if len(col_values) >= self.feature_samples:
                    features[i] = torch.tensor(col_values.iloc[:self.feature_samples].values, dtype=torch.float32)
                else:
                    # Pad with mean
                    mean_val = col_values.mean()
                    padded_values = np.concatenate([
                        col_values.values,
                        np.full(self.feature_samples - len(col_values), mean_val)
                    ])
                    features[i] = torch.tensor(padded_values, dtype=torch.float32)
                
                feature_mask[i] = 1.0
        
        return features, feature_mask
    
    def _process_target(self, y: pd.Series) -> torch.Tensor:
        """Process target into statistics tensor."""
        target_stats = torch.tensor([
            float(y.mean()),
            float(y.std()),
            float(y.min()),
            float(y.max())
        ], dtype=torch.float32)
        
        # Handle NaN values
        target_stats = torch.nan_to_num(target_stats, nan=0.0)
        
        return target_stats
    
    def _process_rankings(self, rankings: Dict) -> torch.Tensor:
        """Convert algorithm rankings to labels tensor."""
        # Create ranking scores (higher = better)
        labels = torch.zeros(self.num_algorithms)
        
        max_rank = len(rankings)
        for algo_name, ranking_info in rankings.items():
            if algo_name in self.algorithm_names:
                algo_idx = list(self.algorithm_names).index(algo_name)
                rank = ranking_info.get('rank', max_rank)
                # Convert rank to score (1st rank = highest score)
                score = max_rank - rank + 1
                labels[algo_idx] = score
        
        return labels
    
    def _create_dummy_sample(self):
        """Create dummy sample for error cases."""
        return {
            'features': torch.zeros(self.max_features, self.feature_samples),
            'feature_mask': torch.zeros(self.max_features),
            'target_stats': torch.zeros(4),
            'algorithm_labels': torch.zeros(self.num_algorithms),
            'best_algorithm': 0
        }


class DatasetMetaLearner:
    """
    Main class for training and using the variable-length dataset meta-learning model.
    """
    
    def __init__(self, max_features: int = 500, feature_samples: int = 100,
                 encoder_hidden_dim: int = 128, encoder_output_dim: int = 64,
                 predictor_hidden_dim: int = 128, device: str = None):
        
        self.max_features = max_features
        self.feature_samples = feature_samples
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.algorithm_encoder = None
        self.scaler = StandardScaler()
        
        # Model architecture parameters
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        
        print(f"MetaLearner initialized with device: {self.device}")
    
    def load_data(self, meta_dataset_path: str) -> Tuple[VariableDatasetDataset, VariableDatasetDataset]:
        """Load and split meta-learning dataset."""
        print(f"Loading meta-learning dataset from: {meta_dataset_path}")
        
        meta_df = pd.read_csv(meta_dataset_path)
        print(f"Loaded {len(meta_df)} dataset records")
        
        # Split into train and validation
        train_df, val_df = train_test_split(meta_df, test_size=0.2, random_state=42, 
                                          stratify=meta_df['best_algorithm'])
        
        # Create datasets
        train_dataset = VariableDatasetDataset(
            train_df, max_features=self.max_features, feature_samples=self.feature_samples
        )
        val_dataset = VariableDatasetDataset(
            val_df, max_features=self.max_features, feature_samples=self.feature_samples,
            transform_target=False
        )
        
        # Use the same algorithm encoder for validation
        val_dataset.algorithm_encoder = train_dataset.algorithm_encoder
        val_dataset.algorithm_names = train_dataset.algorithm_names
        val_dataset.num_algorithms = train_dataset.num_algorithms
        val_df['algorithm_label'] = train_dataset.algorithm_encoder.transform(val_df['best_algorithm'])
        val_dataset.meta_df = val_df
        
        self.algorithm_encoder = train_dataset.algorithm_encoder
        
        return train_dataset, val_dataset
    
    def train(self, meta_dataset_path: str, num_epochs: int = 50, batch_size: int = 16,
              learning_rate: float = 0.001, save_path: str = "dataset_meta_model.pth"):
        """Train the meta-learning model."""
        
        # Load data
        train_dataset, val_dataset = self.load_data(meta_dataset_path)
        
        # Create model
        self.model = MetaLearningModel(
            num_algorithms=train_dataset.num_algorithms,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_output_dim=self.encoder_output_dim,
            predictor_hidden_dim=self.predictor_hidden_dim,
            max_features=self.max_features
        ).to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training setup
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Loss functions
        ranking_loss = nn.MSELoss()  # For algorithm ranking scores
        classification_loss = nn.CrossEntropyLoss()  # For best algorithm prediction
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nStarting training with {len(train_dataset)} train samples, {len(val_dataset)} val samples")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                feature_mask = batch['feature_mask'].to(self.device)
                target_stats = batch['target_stats'].to(self.device)
                algorithm_labels = batch['algorithm_labels'].to(self.device)
                best_algorithm = batch['best_algorithm'].to(self.device).long()
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(features, feature_mask, target_stats)
                
                # Combined loss
                loss1 = ranking_loss(predictions, algorithm_labels)
                loss2 = classification_loss(predictions, best_algorithm)
                total_loss = loss1 + 0.5 * loss2
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(total_loss.item())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    feature_mask = batch['feature_mask'].to(self.device)
                    target_stats = batch['target_stats'].to(self.device)
                    algorithm_labels = batch['algorithm_labels'].to(self.device)
                    best_algorithm = batch['best_algorithm'].to(self.device).long()
                    
                    predictions = self.model(features, feature_mask, target_stats)
                    
                    loss1 = ranking_loss(predictions, algorithm_labels)
                    loss2 = classification_loss(predictions, best_algorithm)
                    total_loss = loss1 + 0.5 * loss2
                    val_losses.append(total_loss.item())
                    
                    # Accuracy calculation
                    predicted_best = torch.argmax(predictions, dim=1)
                    correct_predictions += (predicted_best == best_algorithm).sum().item()
                    total_predictions += len(best_algorithm)
            
            # Calculate averages
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Print progress
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss: {avg_val_loss:.4f}")
                print(f"  Val Accuracy: {accuracy:.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.save(save_path)
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training completed. Best model saved to: {save_path}")
        
        # Load best model
        self.load(save_path)
    
    def predict(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Predict best algorithm for a given dataset.
        
        Returns:
            best_algorithm: Name of the best algorithm
            rankings: List of (algorithm_name, score) tuples sorted by score
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            # Process input dataset
            features, feature_mask = self._process_single_dataset_features(X)
            target_stats = self._process_single_target(y)
            
            # Move to device
            features = features.unsqueeze(0).to(self.device)
            feature_mask = feature_mask.unsqueeze(0).to(self.device)
            target_stats = target_stats.unsqueeze(0).to(self.device)
            
            # Predict
            predictions = self.model(features, feature_mask, target_stats)
            scores = torch.softmax(predictions, dim=1).cpu().numpy()[0]
            
            # Create rankings
            algorithm_names = self.algorithm_encoder.classes_
            algorithm_scores = list(zip(algorithm_names, scores))
            algorithm_scores.sort(key=lambda x: x[1], reverse=True)
            
            best_algorithm = algorithm_scores[0][0]
            
            return best_algorithm, algorithm_scores
    
    def _process_single_dataset_features(self, X: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single dataset's features."""
        X_numeric = X.select_dtypes(include=[np.number])
        
        n_features = min(X_numeric.shape[1], self.max_features)
        feature_samples = min(len(X_numeric), self.feature_samples)
        
        features = torch.zeros(self.max_features, self.feature_samples)
        feature_mask = torch.zeros(self.max_features)
        
        if n_features > 0:
            sampled_X = X_numeric.iloc[:feature_samples, :n_features]
            
            for i in range(n_features):
                col_values = sampled_X.iloc[:, i].fillna(sampled_X.iloc[:, i].mean())
                
                if len(col_values) >= self.feature_samples:
                    features[i] = torch.tensor(col_values.iloc[:self.feature_samples].values, dtype=torch.float32)
                else:
                    mean_val = col_values.mean()
                    padded_values = np.concatenate([
                        col_values.values,
                        np.full(self.feature_samples - len(col_values), mean_val)
                    ])
                    features[i] = torch.tensor(padded_values, dtype=torch.float32)
                
                feature_mask[i] = 1.0
        
        return features, feature_mask
    
    def _process_single_target(self, y: pd.Series) -> torch.Tensor:
        """Process a single target variable."""
        target_stats = torch.tensor([
            float(y.mean()),
            float(y.std()),
            float(y.min()),
            float(y.max())
        ], dtype=torch.float32)
        
        return torch.nan_to_num(target_stats, nan=0.0)
    
    def save(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'algorithm_encoder': self.algorithm_encoder,
            'config': {
                'max_features': self.max_features,
                'feature_samples': self.feature_samples,
                'encoder_hidden_dim': self.encoder_hidden_dim,
                'encoder_output_dim': self.encoder_output_dim,
                'predictor_hidden_dim': self.predictor_hidden_dim,
                'num_algorithms': self.model.num_algorithms
            }
        }
        
        torch.save(save_dict, filepath)
    
    def load(self, filepath: str):
        """Load a trained model."""
        save_dict = torch.load(filepath, map_location=self.device, weights_only=False)
        
        config = save_dict['config']
        self.max_features = config['max_features']
        self.feature_samples = config['feature_samples']
        self.encoder_hidden_dim = config['encoder_hidden_dim']
        self.encoder_output_dim = config['encoder_output_dim']
        self.predictor_hidden_dim = config['predictor_hidden_dim']
        
        self.model = MetaLearningModel(
            num_algorithms=config['num_algorithms'],
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_output_dim=self.encoder_output_dim,
            predictor_hidden_dim=self.predictor_hidden_dim,
            max_features=self.max_features
        ).to(self.device)
        
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.algorithm_encoder = save_dict['algorithm_encoder']
        
        print(f"Model loaded from: {filepath}")


def demo_variable_length_meta_learning():
    """Demonstrate the variable-length meta-learning system."""
    print("Variable-Length Dataset Meta-Learning Demo")
    print("=" * 50)
    
    # Check if meta-learning dataset exists
    meta_dataset_path = "src/automl/meta_learning_dataset.csv"
    
    if not Path(meta_dataset_path).exists():
        print(f"Meta-learning dataset not found at: {meta_dataset_path}")
        print("Please run dataset_evaluator.py first to create the meta-learning dataset.")
        return
    
    # Create and train meta-learner
    meta_learner = DatasetMetaLearner(
        max_features=200,
        feature_samples=50,
        encoder_hidden_dim=64,
        encoder_output_dim=32,
        predictor_hidden_dim=64
    )
    
    # Train model
    meta_learner.train(
        meta_dataset_path=meta_dataset_path,
        num_epochs=30,
        batch_size=8,
        learning_rate=0.001,
        save_path="variable_length_meta_model.pth"
    )
    
    print("Training completed!")


if __name__ == "__main__":
    demo_variable_length_meta_learning()
