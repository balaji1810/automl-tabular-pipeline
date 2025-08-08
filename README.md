# AutoML System - Reproducibility Instructions

This document provides step-by-step instructions to reproduce the AutoML solution for the final test dataset. The AutoML system uses meta-learning to select the best algorithms, followed by hyperparameter optimization and model training.

## Prerequisites

### Environment Setup
1. **Python Environment**: Create and activate a Python virtual environment (Python ≥3.10 required)
   ```cmd
   python -m venv automl-tabular-env
   automl-tabular-env\Scripts\activate
   ```

2. **Install Dependencies**: Install the package and all required dependencies
   ```cmd
   pip install -e .
   ```

3. **Verify Installation**: Test that the installation was successful
   ```cmd
   python -c "import automl"
   ```

### Required Dependencies
The system requires the following main packages (automatically installed via pyproject.toml):
- scikit-learn (machine learning algorithms)
- xgboost, lightgbm (gradient boosting, needs GPU/CUDA support)
- optuna (hyperparameter optimization)
- torch (meta-learning model)
- tabpfn (TabPFN algorithm, needs GPU/CUDA support)
- pandas, numpy (data handling)

## Dataset Preparation

### Practice datasets:
The following datasets can be used or any other datasets you prefer:

* bike_sharing_demand
* brazilian_houses 
* wine_quality
* superconductivity 
* yprop_4_1

You can download the practice data using:
```bash
python download-datasets.py
```

This will by default, download the data to the `/data` folder with the following structure.
The fold numbers `1, ..., n` refer to **outer folds**, meaning each can be treated as a separate dataset for training and validation. You can use the `--fold` argument to specify which fold you would like.

```bash
./data
├── bike_sharing_demand
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 2
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 3
    ...
├── wine_quality 
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
    ...
```

1. **Download Dataset**: Download the dataset
   ```cmd
   python download-dataset.py
   ```

2. **Verify Dataset Structure**: Ensure the dataset is available at `data/brazilian_houses/1/`
   - `X_train.parquet`: Training features
   - `y_train.parquet`: Training targets  
   - `X_test.parquet`: Test features (for predictions)
   - `y_test.parquet`: Test targets (for evaluation)

## Training the AutoML System

### Command 1: Train AutoML Model on Test Dataset (≤24 hours)

Run the following command to train the AutoML system on the test dataset training set:

```cmd
python run.py --task brazilian_houses --fold 1 --seed 42 --timeout 86400 --output-path data\brazilian_houses\predictions.npy
```

* `run.py`: A script that loads in a downloaded dataset, trains an _AutoML-System_ and then generates predictions for
`X_test`, saving those predictions to a file. For the training datasets, you will also have access to `y_test` which
is present in the `./data` folder.

**Command Parameters:**
- `--task brazilian_houses`: Specifies the Brazilian houses dataset
- `--fold 1`: Uses fold 1 (the only fold available for Brazilian houses dataset)
- `--seed 42`: Sets random seed for reproducibility
- `--timeout 86400`: Sets maximum timeout to 24 hours (86400 seconds)
- `--output-path data\brazilian_houses\predictions.npy`: Output path for predictions

**What this command does:**
1. **Meta-Feature Extraction**: Extracts dataset meta-features from training data
2. **Algorithm Selection**: Uses trained meta-learning model (`meta_model.pth`) to recommend top algorithms
3. **Algorithm Training**: Trains multiple algorithms with algorithm-specific preprocessing:
4. **Feature Selection**: Applies feature selection techniques to reduce dimensionality
5. **Hyperparameter Optimization**: Uses Optuna for hyperparameter tuning of each algorithm
6. **Model Selection**: Selects best performing model based on validation R² score
7. **Prediction Generation**: Generates predictions on test set using the best model
8. **Output**: Saves predictions as `predictions.npy` file

**Expected Outputs:**
- **Fully trained model**: The system returns a complete AutoML pipeline with the best-performing algorithm
- **Predictions file**: `data/brazilian_houses/predictions.npy` containing test predictions
- **Logs**: Detailed logging of the training process, algorithm selection, and performance scores

## Alternative Usage Scenarios

### For Custom Timeout (shorter evaluation time):
```cmd
python run.py --task brazilian_houses --fold 1 --seed 42 --timeout 3600 --output-path data\brazilian_houses\predictions.npy
```
This runs with 1-hour timeout for faster evaluation.

### For Different Random Seed:
```cmd
python run.py --task brazilian_houses --fold 1 --seed 123 --timeout 86400 --output-path data\brazilian_houses\predictions.npy
```

## Command 2: Generate Predictions (if model is already trained)

Since our AutoML system is end-to-end, **Command 1 above already generates the final predictions**. The `predictions.npy` file will be created automatically after training completes.

## Output Verification

After running the command, verify the output:

1. **Check predictions file exists**:
   ```cmd
   dir data\brazilian_houses\predictions.npy
   ```

2. **Verify predictions format** (Python):
   ```python
   import numpy as np
   preds = np.load('data/brazilian_houses/predictions.npy')
   print(f"Predictions shape: {preds.shape}")
   print(f"Predictions type: {preds.dtype}")
   ```

The predictions array should contain numerical regression values for each test sample.

## Reference performance

| Dataset | Test performance |
| -- | -- |
| bike_sharing_demand | 0.9457 |
| brazilian_houses | 0.9896 |
| superconductivity | 0.9311 |
| wine_quality | 0.4410 |
| yprop_4_1 | 0.0778 |
| exam_dataset | 0.9290 |

The scores listed are the R² values calculated using scikit-learn's `metrics.r2_score`.

## Troubleshooting

### Common Issues:
1. **Import Error**: Ensure package is installed with `pip install -e .`
2. **Dataset Not Found**: Run `python download-dataset.py` first
3. **Memory Issues**: Reduce timeout or use quiet mode
4. **Timeout**: Increase `--timeout` parameter for complex datasets

### Performance Expectations:
- **Training Time**: Up to 24 hours for full optimization
- **Output Format**: NumPy array saved as `.npy` file

## Reproducibility Notes

- **Random Seed**: Fixed at 10 for all random operations
- **Cross-validation**: Uses fixed train/validation split (80/20)
- **Algorithm Order**: Deterministic algorithm evaluation order
- **Hyperparameter Search**: Optuna with fixed seed ensures reproducible hyperparameter optimization

This ensures that running the same command multiple times will produce identical results.