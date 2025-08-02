"""
Simple Usage Example: Variable-Length Meta-Learning for Algorithm Selection

This example shows how to use the complete meta-learning system after training.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from automl.complete_meta_automl import VariableMetaAutoML
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split


def example_with_synthetic_data():
    """Example using synthetic regression data."""
    print("=" * 60)
    print("EXAMPLE 1: SYNTHETIC REGRESSION DATA")
    print("=" * 60)
    
    # Generate synthetic data
    X, y = make_regression(
        n_samples=1000, 
        n_features=15, 
        n_informative=10,
        noise=0.1, 
        random_state=42
    )
    
    # Convert to pandas
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y, name='target')
    
    print(f"Dataset created:")
    print(f"  Shape: {X.shape}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Target std: {y.std():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Check if trained meta-model exists
    meta_model_path = "variable_length_meta_model.pth"
    if not Path(meta_model_path).exists():
        print(f"\n⚠️  Pre-trained meta-model not found: {meta_model_path}")
        print("The system will work but without meta-learning guidance.")
        meta_model_path = None
    else:
        print(f"✅ Using pre-trained meta-model: {meta_model_path}")
    
    # Create AutoML system
    print(f"\nCreating Variable Meta-AutoML system...")
    automl = VariableMetaAutoML(
        meta_model_path=meta_model_path,
        optimize_hyperparams=True,
        use_ensemble=False,
        n_trials=25,
        random_state=42
    )
    
    # Train the system
    print(f"\nTraining AutoML system...")
    training_results = automl.fit(X_train, y_train, validation_split=0.2)
    
    # Show results
    analysis = training_results['dataset_analysis']
    print(f"\n📊 Dataset Analysis Results:")
    print(f"  Recommended Algorithm: {analysis['recommended_algorithm']}")
    print(f"  Meta-learning Type: {analysis.get('meta_learning', 'None')}")
    print(f"  Training R²: {training_results['train_r2']:.4f}")
    print(f"  Validation R²: {training_results['validation_r2']:.4f}")
    print(f"  Training Time: {training_results['training_time']:.2f} seconds")
    
    # Show algorithm rankings if available
    if analysis.get('algorithm_ranking'):
        print(f"\n🏆 Algorithm Rankings:")
        for i, (alg, score) in enumerate(analysis['algorithm_ranking'][:5], 1):
            print(f"  {i}. {alg}: {score:.4f}")
    
    # Evaluate on test set
    print(f"\n🧪 Test Set Evaluation:")
    test_metrics = automl.evaluate(X_test, y_test)
    print(f"  Test R²: {test_metrics['r2_score']:.4f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test MAE: {test_metrics['mae']:.4f}")
    
    # Make some predictions
    print(f"\n🔮 Sample Predictions:")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    predictions = automl.predict(X_test.iloc[sample_indices])
    
    for i, idx in enumerate(sample_indices):
        actual = y_test.iloc[idx]
        predicted = predictions[i]
        error = abs(actual - predicted)
        print(f"  Sample {i+1}: Actual={actual:.2f}, Predicted={predicted:.2f}, Error={error:.2f}")
    
    return automl


def example_with_california_housing():
    """Example using California housing dataset."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CALIFORNIA HOUSING DATA")
    print("=" * 60)
    
    try:
        # Load California housing data
        california = fetch_california_housing()
        X = pd.DataFrame(california.data, columns=california.feature_names)
        y = pd.Series(california.target, name='house_value')
        
        print(f"California Housing Dataset:")
        print(f"  Shape: {X.shape}")
        print(f"  Features: {list(X.columns)}")
        print(f"  Target (house value): ${y.mean():.0f}k ± ${y.std():.0f}k")
        print(f"  Range: ${y.min():.1f}k - ${y.max():.1f}k")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create AutoML system with ensemble
        print(f"\nCreating AutoML system with ensemble...")
        automl = VariableMetaAutoML(
            meta_model_path="variable_length_meta_model.pth" if Path("variable_length_meta_model.pth").exists() else None,
            optimize_hyperparams=True,
            use_ensemble=True,
            ensemble_top_k=3,
            n_trials=20,
            random_state=42
        )
        
        # Train
        print(f"Training with ensemble approach...")
        training_results = automl.fit(X_train, y_train)
        
        # Results
        analysis = training_results['dataset_analysis']
        print(f"\n📊 Results:")
        print(f"  Model Type: {'Ensemble' if training_results['use_ensemble'] else 'Single'}")
        print(f"  Recommended Algorithm: {analysis['recommended_algorithm']}")
        print(f"  Training R²: {training_results['train_r2']:.4f}")
        print(f"  Validation R²: {training_results['validation_r2']:.4f}")
        
        # Test evaluation
        test_metrics = automl.evaluate(X_test, y_test)
        print(f"  Test R²: {test_metrics['r2_score']:.4f}")
        print(f"  Test RMSE: ${test_metrics['rmse']:.2f}k")
        print(f"  Test MAE: ${test_metrics['mae']:.2f}k")
        
        # Model info
        model_info = automl.get_model_info()
        print(f"\n🔧 Model Info:")
        print(f"  Model Type: {model_info['model_type']}")
        print(f"  Meta-learning: {model_info.get('meta_learning', 'N/A')}")
        
        # Save the trained model
        model_save_path = "trained_california_housing_automl.joblib"
        automl.save(model_save_path)
        print(f"  Saved to: {model_save_path}")
        
        return automl
        
    except Exception as e:
        print(f"Error with California housing example: {e}")
        return None


def example_load_and_use_saved_model():
    """Example showing how to load and use a saved model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: LOADING AND USING SAVED MODEL")
    print("=" * 60)
    
    model_path = "trained_california_housing_automl.joblib"
    
    if not Path(model_path).exists():
        print(f"Saved model not found: {model_path}")
        print("Please run Example 2 first to create a saved model.")
        return
    
    try:
        # Load the saved model
        print(f"Loading saved AutoML model...")
        automl = VariableMetaAutoML.load(model_path)
        
        # Get model information
        model_info = automl.get_model_info()
        print(f"✅ Model loaded successfully!")
        print(f"  Model type: {model_info['model_type']}")
        print(f"  Training R²: {model_info['training_results']['train_r2']:.4f}")
        print(f"  Validation R²: {model_info['training_results']['validation_r2']:.4f}")
        
        # Create some test data (simulate new California housing data)
        print(f"\n🔮 Making predictions on new data...")
        
        # Generate realistic-looking California housing data
        np.random.seed(123)
        new_data = pd.DataFrame({
            'MedInc': np.random.normal(5.0, 2.0, 10),          # Median income
            'HouseAge': np.random.uniform(1, 52, 10),          # House age  
            'AveRooms': np.random.normal(6.0, 1.0, 10),        # Average rooms
            'AveBedrms': np.random.normal(1.0, 0.2, 10),       # Average bedrooms
            'Population': np.random.uniform(500, 5000, 10),     # Population
            'AveOccup': np.random.normal(3.0, 0.5, 10),        # Average occupancy
            'Latitude': np.random.uniform(32, 42, 10),          # Latitude
            'Longitude': np.random.uniform(-125, -114, 10)      # Longitude
        })
        
        # Make predictions
        predictions = automl.predict(new_data)
        
        print(f"Predictions for 10 new houses:")
        for i, pred in enumerate(predictions):
            print(f"  House {i+1}: ${pred:.1f}k")
        
        print(f"\nPrediction statistics:")
        print(f"  Mean: ${predictions.mean():.1f}k")
        print(f"  Range: ${predictions.min():.1f}k - ${predictions.max():.1f}k")
        
    except Exception as e:
        print(f"Error loading/using saved model: {e}")


def print_system_requirements():
    """Print system requirements and setup instructions."""
    print("\n" + "=" * 60)
    print("SYSTEM REQUIREMENTS & SETUP")
    print("=" * 60)
    
    print("📋 Requirements:")
    print("1. Python packages:")
    print("   - pandas, numpy, scikit-learn")
    print("   - torch (PyTorch)")
    print("   - xgboost, lightgbm") 
    print("   - optuna (for hyperparameter optimization)")
    print("   - joblib (for model saving)")
    
    print("\n2. Data structure (for training meta-learner):")
    print("   data/")
    print("   ├── dataset1/")
    print("   │   ├── 1/")
    print("   │   │   ├── X_train.parquet")
    print("   │   │   ├── y_train.parquet")
    print("   │   │   ├── X_test.parquet")
    print("   │   │   └── y_test.parquet")
    print("   │   └── 2/ (more folds...)")
    print("   └── dataset2/ (more datasets...)")
    
    print("\n3. Training the meta-learner:")
    print("   python main_meta_learning.py --step all")
    
    print("\n4. Using the trained system:")
    print("   from src.automl.complete_meta_automl import VariableMetaAutoML")
    print("   automl = VariableMetaAutoML(meta_model_path='variable_length_meta_model.pth')")
    print("   results = automl.fit(X_train, y_train)")
    print("   predictions = automl.predict(X_test)")


def main():
    """Run all examples."""
    print("Variable-Length Meta-Learning for Algorithm Selection")
    print("Based on 'Learning dataset representation for automatic machine learning algorithm selection'")
    
    print_system_requirements()
    
    # Example 1: Synthetic data
    automl1 = example_with_synthetic_data()
    
    # Example 2: California housing
    automl2 = example_with_california_housing()
    
    # Example 3: Load saved model
    example_load_and_use_saved_model()
    
    print("\n" + "🎉" * 20)
    print("ALL EXAMPLES COMPLETED!")
    print("🎉" * 20)
    
    print("\n📝 Summary:")
    print("- The system can handle datasets with variable numbers of features")
    print("- It learns dataset representations using neural networks")
    print("- Algorithm selection is based on actual dataset analysis, not just statistics")
    print("- Supports both single algorithm and ensemble approaches")
    print("- Includes automatic hyperparameter optimization")
    print("- Models can be saved and reloaded for future use")
    
    print("\n🚀 Next steps:")
    print("1. Train the meta-learner on your own datasets using main_meta_learning.py")
    print("2. Use the trained system for automatic algorithm selection")
    print("3. Experiment with different ensemble configurations")
    print("4. Add more algorithms to the portfolio as needed")


if __name__ == "__main__":
    main()


"""
# Load the model
from meta_trainer import load_ranking_meta_model, extract_meta_features, predict_algorithm_rankings
models, checkpoint = load_ranking_meta_model("meta_model.pth")

meta_df = extract_meta_features(X,y)

predictions = predict_algorithm_rankings(models, checkpoint,meta_features)
"""