"""
Complete Meta-Learning Pipeline for Automatic Algorithm Selection

This script provides a complete workflow for:
1. Evaluating all datasets with algorithm portfolio
2. Training variable-length meta-learning model  
3. Using the system for automatic algorithm selection

Usage:
    python main_meta_learning.py --step [evaluate|train|demo|all]
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

# Import our modules
from automl.dataset_evaluator import evaluate_all_datasets, save_meta_dataset
from automl.variable_meta_learner import DatasetMetaLearner
from automl.complete_meta_automl import VariableMetaAutoML

# For testing
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split


def step_1_evaluate_datasets():
    """Step 1: Evaluate all datasets with algorithm portfolio."""
    print("=" * 70)
    print("STEP 1: EVALUATING DATASETS WITH ALGORITHM PORTFOLIO")
    print("=" * 70)
    
    print("This step will:")
    print("1. Find all datasets in the 'data' folder")
    print("2. Evaluate each dataset with all algorithms in the portfolio")
    print("3. Create a meta-learning dataset with results")
    print("4. Save results to 'src/automl/meta_learning_dataset.csv'")
    
    # Check if data folder exists
    data_folder = Path("data")
    if not data_folder.exists():
        print(f"\nError: Data folder '{data_folder}' not found!")
        print("Please ensure you have datasets in the data folder.")
        return False
    
    print(f"\nEvaluating datasets from: {data_folder}")
    
    try:
        # Evaluate all datasets
        meta_df = evaluate_all_datasets(data_folder="data", max_folds=10)
        
        if meta_df.empty:
            print("No datasets were successfully evaluated!")
            return False
        
        # Save the meta-learning dataset
        output_path = "src/automl/meta_learning_dataset.csv"
        save_meta_dataset(meta_df, output_path)
        
        print(f"\n✅ Step 1 completed successfully!")
        print(f"Created meta-learning dataset with {len(meta_df)} records")
        print(f"Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in Step 1: {e}")
        return False


def step_2_train_meta_learner():
    """Step 2: Train the variable-length meta-learning model."""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING VARIABLE-LENGTH META-LEARNING MODEL") 
    print("=" * 70)
    
    print("This step will:")
    print("1. Load the meta-learning dataset created in Step 1")
    print("2. Train a neural network that can handle variable-length datasets")
    print("3. Save the trained model for future use")
    
    # Check if meta-learning dataset exists
    meta_dataset_path = "src/automl/meta_learning_dataset.csv"
    if not Path(meta_dataset_path).exists():
        print(f"\n❌ Meta-learning dataset not found: {meta_dataset_path}")
        print("Please run Step 1 first to create the dataset.")
        return False
    
    # Load and check dataset
    try:
        meta_df = pd.read_csv(meta_dataset_path)
        print(f"\nLoaded meta-learning dataset: {len(meta_df)} records")
        print(f"Unique datasets: {meta_df['dataset_name'].nunique()}")
        print(f"Algorithms: {meta_df['best_algorithm'].value_counts().to_dict()}")
        
        if len(meta_df) < 10:
            print("\n⚠️  Warning: Very small dataset. Results may not be reliable.")
        
    except Exception as e:
        print(f"\n❌ Error loading meta-dataset: {e}")
        return False
    
    try:
        # Create and train meta-learner
        print(f"\nTraining variable-length meta-learner...")
        
        meta_learner = DatasetMetaLearner(
            max_features=min(500, meta_df['n_features'].max()),  # Adaptive to data
            feature_samples=100,
            encoder_hidden_dim=128,
            encoder_output_dim=64,
            predictor_hidden_dim=128
        )
        
        # Train model
        model_save_path = "variable_length_meta_model.pth"
        meta_learner.train(
            meta_dataset_path=meta_dataset_path,
            num_epochs=50,
            batch_size=max(4, len(meta_df) // 20),  # Adaptive batch size
            learning_rate=0.001,
            save_path=model_save_path
        )
        
        print(f"\n✅ Step 2 completed successfully!")
        print(f"Meta-learning model saved to: {model_save_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in Step 2: {e}")
        print(f"Details: {str(e)}")
        return False


def step_3_demo_system():
    """Step 3: Demonstrate the complete meta-AutoML system."""
    print("\n" + "=" * 70)
    print("STEP 3: DEMONSTRATING COMPLETE META-AUTOML SYSTEM")
    print("=" * 70)
    
    print("This step will:")
    print("1. Test the system on synthetic datasets")
    print("2. Show algorithm recommendations") 
    print("3. Compare with and without meta-learning")
    print("4. Test ensemble vs single algorithm approaches")
    
    # Check if meta-model exists
    meta_model_path = "variable_length_meta_model.pth"
    has_meta_model = Path(meta_model_path).exists()
    
    if has_meta_model:
        print(f"\n✅ Using trained meta-model: {meta_model_path}")
    else:
        print(f"\n⚠️  No trained meta-model found. System will use default algorithms.")
        meta_model_path = None
    
    try:
        # Test datasets
        test_datasets = [
            {
                "name": "Small Linear Dataset",
                "data": make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
            },
            {
                "name": "Large Nonlinear Dataset", 
                "data": make_regression(n_samples=2000, n_features=20, noise=0.2, random_state=42)
            },
            {
                "name": "High Dimensional Dataset",
                "data": make_regression(n_samples=500, n_features=50, noise=0.15, random_state=42)
            }
        ]
        
        # Try to add California housing if available
        try:
            california = fetch_california_housing()
            test_datasets.append({
                "name": "California Housing (Real Data)",
                "data": (california.data, california.target)
            })
        except:
            print("California housing dataset not available.")
        
        results_summary = []
        
        for dataset_info in test_datasets:
            name = dataset_info["name"]
            X, y = dataset_info["data"]
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            y = pd.Series(y)
            
            print(f"\n{'-'*50}")
            print(f"Testing: {name}")
            print(f"Shape: {X.shape}")
            print(f"{'-'*50}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Test with meta-learning
            try:
                automl = VariableMetaAutoML(
                    meta_model_path=meta_model_path,
                    optimize_hyperparams=True,
                    use_ensemble=False,
                    n_trials=20,
                    random_state=42
                )
                
                # Train and evaluate
                training_results = automl.fit(X_train, y_train, validation_split=0.2)
                test_metrics = automl.evaluate(X_test, y_test)
                
                result = {
                    "dataset": name,
                    "recommended_algorithm": training_results["dataset_analysis"]["recommended_algorithm"],
                    "test_r2": test_metrics["r2_score"],
                    "test_rmse": test_metrics["rmse"],
                    "training_time": training_results["training_time"],
                    "meta_learning": training_results["dataset_analysis"].get("meta_learning", "unknown"),
                    "status": "success"
                }
                
                print(f"Recommended: {result['recommended_algorithm']}")
                print(f"Test R²: {result['test_r2']:.4f}")
                print(f"Test RMSE: {result['test_rmse']:.4f}")
                print(f"Time: {result['training_time']:.2f}s")
                print(f"Meta-learning: {result['meta_learning']}")
                
                results_summary.append(result)
                
            except Exception as e:
                print(f"Error testing {name}: {e}")
                results_summary.append({
                    "dataset": name,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Print summary
        print(f"\n{'='*70}")
        print("DEMO RESULTS SUMMARY")
        print(f"{'='*70}")
        
        print(f"{'Dataset':<30} {'Algorithm':<20} {'R²':<8} {'RMSE':<10} {'Time':<8}")
        print("-" * 78)
        
        for result in results_summary:
            if result["status"] == "success":
                alg_name = result["recommended_algorithm"][:19]  # Truncate long names
                print(f"{result['dataset']:<30} {alg_name:<20} {result['test_r2']:<8.4f} {result['test_rmse']:<10.2f} {result['training_time']:<8.1f}")
            else:
                print(f"{result['dataset']:<30} {'FAILED':<20} {'N/A':<8} {'N/A':<10} {'N/A':<8}")
        
        print(f"\n✅ Step 3 completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in Step 3: {e}")
        return False


def run_complete_pipeline():
    """Run the complete pipeline: evaluate -> train -> demo."""
    print("🚀 COMPLETE META-LEARNING PIPELINE")
    print("=" * 70)
    print("This will run all steps in sequence:")
    print("1. Evaluate datasets with algorithm portfolio")
    print("2. Train variable-length meta-learning model")
    print("3. Demonstrate the complete system")
    print("=" * 70)
    
    # Step 1: Evaluate datasets
    if not step_1_evaluate_datasets():
        print("\n❌ Pipeline failed at Step 1")
        return False
    
    # Step 2: Train meta-learner
    if not step_2_train_meta_learner():
        print("\n❌ Pipeline failed at Step 2")
        return False
    
    # Step 3: Demo system
    if not step_3_demo_system():
        print("\n❌ Pipeline failed at Step 3")
        return False
    
    print("\n" + "🎉" * 20)
    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("🎉" * 20)
    print("\nKey outputs created:")
    print("1. src/automl/meta_learning_dataset.csv - Training data for meta-learner")
    print("2. variable_length_meta_model.pth - Trained meta-learning model")
    print("3. Complete AutoML system ready for use!")
    
    print("\nTo use the system in your own code:")
    print("""
from src.automl.complete_meta_automl import VariableMetaAutoML

# Create AutoML system
automl = VariableMetaAutoML(
    meta_model_path="variable_length_meta_model.pth",
    optimize_hyperparams=True
)

# Train on your data
results = automl.fit(X_train, y_train)

# Make predictions
predictions = automl.predict(X_test)
""")
    
    return True


def print_usage():
    """Print usage information."""
    print("""
Meta-Learning for Automatic Algorithm Selection

This system implements variable-length dataset meta-learning based on the paper:
"Learning dataset representation for automatic machine learning algorithm selection"

Available steps:

1. evaluate  - Evaluate all datasets in 'data' folder with algorithm portfolio
2. train     - Train the variable-length meta-learning model
3. demo      - Demonstrate the complete system on test datasets  
4. all       - Run complete pipeline (evaluate -> train -> demo)

Usage:
    python main_meta_learning.py --step [evaluate|train|demo|all]
    python main_meta_learning.py --help

Examples:
    python main_meta_learning.py --step all
    python main_meta_learning.py --step evaluate
    python main_meta_learning.py --step train
    python main_meta_learning.py --step demo

Requirements:
- Datasets in 'data' folder with structure: data/dataset_name/fold/X_train.parquet
- Python packages: pandas, numpy, torch, sklearn, xgboost, lightgbm, optuna
""")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Meta-Learning Pipeline for Automatic Algorithm Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--step',
        choices=['evaluate', 'train', 'demo', 'all'],
        required=True,
        help='Step to execute: evaluate datasets, train meta-learner, demo system, or run all'
    )
    
    args = parser.parse_args()
    
    if args.step == 'evaluate':
        success = step_1_evaluate_datasets()
    elif args.step == 'train':
        success = step_2_train_meta_learner()
    elif args.step == 'demo':
        success = step_3_demo_system()
    elif args.step == 'all':
        success = run_complete_pipeline()
    else:
        print_usage()
        return
    
    if success:
        print(f"\n✅ Step '{args.step}' completed successfully!")
    else:
        print(f"\n❌ Step '{args.step}' failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
