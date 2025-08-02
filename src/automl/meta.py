from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

from sklearn.neural_network import MLPRegressor
import torch

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from tabpfn import TabPFNRegressor

import openml


algorithms = [
    LGBMRegressor(n_jobs=-1),
    XGBRegressor(enable_categorical = True, n_jobs = -1),
    RandomForestRegressor(n_jobs=-1),
    DecisionTreeRegressor(),
    HistGradientBoostingRegressor(),
    LinearRegression(),
    BayesianRidge(),
    GradientBoostingRegressor(),
    MLPRegressor(),
    SVR(),
    # TabPFNRegressor(n_jobs=-1, ignore_pretraining_limits=True),
    # device="cuda" if torch.cuda.is_available() else "cpu",
]

algorithms_dict = {algo.__class__.__name__: algo for algo in algorithms}


datasets = [
    "bike_sharing_demand",
    "brazilian_houses",
    "superconductivity",
    "wine_quality",
    "yprop_4_1"
]

def detect_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Detect numerical and categorical columns in a DataFrame.
    Returns lists of column names.
    """
    # A simple heuristic: dtype kind
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # Also treat low-cardinality ints as categorical
    for col in X.select_dtypes(include=["int"]):
        if X[col].nunique() < 20 and col not in cat_cols:
            cat_cols.append(col)
            num_cols.remove(col)
    return num_cols, cat_cols


# def load_dataset(name: str, fold: int = 1):
#     """Load X_train, y_train, X_test, y_test for a given dataset and fold."""
#     base = Path(__file__).resolve().parents[2] / "data" / name / str(fold)
#     X_train = pd.read_parquet(base / "X_train.parquet")
#     y_train = pd.read_parquet(base / "y_train.parquet").iloc[:, 0]
#     X_test  = pd.read_parquet(base / "X_test.parquet")
#     y_test  = pd.read_parquet(base / "y_test.parquet").iloc[:, 0]
#     return X_train, y_train, X_test, y_test


def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing:
      - numeric: impute + optional scaling
      - categorical: impute + one-hot encode
    """
    num_cols, cat_cols = detect_column_types(X)

    transformers = []
    if num_cols:
        num_steps = []
        num_steps.append(("imputer", SimpleImputer(strategy="mean")))
        # num_steps.append(("scaler", StandardScaler()))
        num_steps.append(("robust_scaler", RobustScaler()))
        transformers.append(("numerical", Pipeline(steps=num_steps), num_cols))
    if cat_cols:
        cat_steps = []
        cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        transformers.append(("categorical", Pipeline(steps=cat_steps), cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    preprocessor.set_output(transform="pandas")
    return preprocessor


def extract_meta_features(X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute meta-features for regression datasets."""
    # Basic sizes
    n, d = X.shape
    meta_features = {
        "n_samples": n,
        "n_features": d,
        "log_n_samples": np.log(n + 1) if n > 0 else 0.0,
        "log_n_features": np.log(d + 1) if d > 0 else 0.0,
        "feature_ratio": d / n if n else 0.0,
        "target_mean": float(y.mean()) if len(y) else 0.0,
        "target_std": float(y.std()) if len(y) else 0.0,
        "target_skew": float(skew(y)) if len(y) else 0.0,
        "target_kurtosis": float(kurtosis(y)) if len(y) else 0.0,
    }

    # Restrict to numeric cols
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0: 
        # no numeric features → fill zeros
        meta_features.update({
            "mean_feature_skew": 0.0,
            "mean_feature_kurtosis": 0.0,
            # "zero_var_pct": 1.0,
            "mean_abs_corr": 0.0,
            "max_abs_corr": 0.0,
        })
        
    else:
        # 1) Per-feature mean skew/kurtosis
        # Compute skew/kurtosis across columns means (column-wise)
        means = X_num.mean(axis=0)
        meta_features["mean_feature_skew"]     = float(skew(means))
        meta_features["mean_feature_kurtosis"] = float(kurtosis(means))

        # 2) Zero-variance percentage
        # zero_var = (X_num.var(axis=0) == 0).sum()
        # meta_features["zero_var_pct"] = float(zero_var / X_num.shape[1])

        # 3) Pairwise correlations (only if >1 numeric column)
        if X_num.shape[1] > 1:
            corr = X_num.corr().abs()
            # take upper triangle without diagonal
            iu = np.triu_indices(corr.shape[0], k=1)
            tri_vals = corr.values[iu]
            meta_features["mean_abs_corr"] = float(np.nanmean(tri_vals))
            meta_features["max_abs_corr"]  = float(np.nanmax(tri_vals))
        else:
            meta_features["mean_abs_corr"] = 0.0
            meta_features["max_abs_corr"]  = 0.0

    # 4) Probing Features
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Preprocess (impute + encode) X_train and X_test
    preprocessor = build_preprocessor(pd.concat([X_train, X_test], ignore_index=True))
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    sample_fraction = 0.01  # 1% sample for probing
    # 4.1) Mean value predictor performance (baseline)
    dummy_mean = DummyRegressor(strategy='mean')
    dummy_mean.fit(X_train, y_train)
    y_pred_mean = dummy_mean.predict(X_test)
    
    meta_features['mean_predictor_r2'] = r2_score(y_test, y_pred_mean)
    
    # 4.2) Decision stump performance (depth=1 tree)
    stump = DecisionTreeRegressor(max_depth=1, random_state=42)
    stump.fit(X_train, y_train)
    y_pred_stump = stump.predict(X_test)
    
    meta_features['decision_stump_r2'] = r2_score(y_test, y_pred_stump)
    
    # Relative improvement over mean predictor
    meta_features['stump_vs_mean_r2_ratio'] = (
        meta_features['decision_stump_r2'] / max(meta_features['mean_predictor_r2'], 1e-10)
    )
    
    # 4.3) Simple rule model performance (linear regression)
    simple_rule = LinearRegression()
    simple_rule.fit(np.array(X_train), np.array(y_train))
    y_pred_rule = simple_rule.predict(np.array(X_test))

    meta_features['simple_rule_r2'] = r2_score(y_test, y_pred_rule)
        
    # Relative improvement over mean predictor
    meta_features['rule_vs_mean_r2_ratio'] = (
        meta_features['simple_rule_r2'] / max(meta_features['mean_predictor_r2'], 1e-10)
    )
    
    # 4.4) Performance of algorithms on 1% of data
    if len(X_train) > 100:  # Only if we have enough data
        # Sample 1% of training data
        sample_size = max(int(len(X_train) * sample_fraction), 10)
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[sample_indices]
        y_sample = y_train.iloc[sample_indices]

        for algo_name, algorithm in algorithms_dict.items():
            try:
                # Clone the algorithm to avoid fitting issues
                from sklearn.base import clone
                algo_clone = clone(algorithm)
                
                # Fit on 1% sample
                algo_clone.fit(X_sample, y_sample)
                
                # Predict on test set
                y_pred_algo = algo_clone.predict(X_test)
                
                # Store performance
                algo_r2 = r2_score(y_test, y_pred_algo)
                
                meta_features[f'{algo_name}_1pct_r2'] = algo_r2
                
                # Relative performance vs baselines
                meta_features[f'{algo_name}_vs_mean_r2_ratio'] = (
                    algo_r2 / max(meta_features['mean_predictor_r2'], 1e-10)
                )
                meta_features[f'{algo_name}_vs_stump_r2_ratio'] = (
                    algo_r2 / max(meta_features['decision_stump_r2'], 1e-10)
                )
                
            except Exception as e:
                print(f"Error evaluating {algo_name} on 1% data: {e}")
                meta_features[f'{algo_name}_1pct_r2'] = -1.0
                meta_features[f'{algo_name}_1pct_rmse'] = float('inf')
                meta_features[f'{algo_name}_vs_mean_r2_ratio'] = 0.0
                meta_features[f'{algo_name}_vs_stump_r2_ratio'] = 0.0
    
    # 4.5) Additional derived meta-features
    meta_features['baseline_difficulty'] = 1 - meta_features['mean_predictor_r2']
    # meta_features['linear_separability'] = meta_features['simple_rule_r2']
    meta_features['tree_advantage'] = (
        meta_features['decision_stump_r2'] - meta_features['simple_rule_r2']
    )

    # 5) Algorithm suitability indicators
    meta_features['tabpfn_suitable'] = 1 if (n < 1000 and d < 100 and n*d < 10000) else 0
    meta_features['svm_suitable'] = 1 if (100 <= n <= 10000 and d <= 1000) else 0
    meta_features['linear_favorable'] = 1 if (meta_features['simple_rule_r2'] > 0.7 and 
                                            meta_features['target_skew'] < 2) else 0

    return meta_features


def train_meta_model(X: pd.DataFrame,y: pd.Series, model_type: str = "nn"):
    """
    Train a meta-model on the extracted meta-features and best selected algorithm.
    """
    """
    # Encode target algorithms
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Training meta-model with {X.shape[0]} datasets and {X.shape[1]} meta-features")
    print(f"Unique algorithms: {list(label_encoder.classes_)}")
    
    # Split data if we have enough samples
    if len(X) > 1000:
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train, y_test = X.values, X.values, y_encoded, y_encoded
        print("Warning: Using all data for both training and testing due to small sample size")
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Define model directly
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    hidden_sizes = [128, 64, 32]
    
    layers = []
    prev_size = input_size
    
    for hidden_size in hidden_sizes:
        layers.extend([
            nn.Linear(prev_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        ])
        prev_size = hidden_size
    
    layers.append(nn.Linear(prev_size, num_classes))
    model = nn.Sequential(*layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 50
    
    for epoch in range(500):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        
        y_test_cpu = y_test_tensor.cpu().numpy()
        predicted_cpu = predicted.cpu().numpy()
        
        accuracy = accuracy_score(y_test_cpu, predicted_cpu)
        
        print(f"Meta-model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_cpu, predicted_cpu, target_names=label_encoder.classes_))
    
    # Save model and metadata
    model_package = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'num_classes': num_classes,
        'hidden_sizes': hidden_sizes,
        'label_encoder': label_encoder,
        'feature_names': list(X.columns),
        'accuracy': accuracy,
        'model_type': model_type
    }
    
    torch.save(model_package, save_path)
    print(f"Meta-model saved to {save_path}")
    
    return model_package
    """
      
       
def load_openml_dataset(
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
    
       
def algorithms_evaluation(algorithms: list, datasets: list):
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

        # Build preprocessor on all data (to avoid leakage, you can fit only on train)
        preprocessor = build_preprocessor(pd.concat([X_train, X_test], ignore_index=True))

        # 1) extract meta-features
        meta = extract_meta_features(X_train, y_train)

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
        algo = sorted_names[0]

        # 4) assemble record
        record = {
            "dataset_index": i,
            "dataset_id": ds["dataset_id"],
            **meta,
            "Algorithm": algo,
            **{f"{n}_r2": scores[n] for n in scores},
        }
        records.append(record)

    # 5) save to CSV
    df = pd.DataFrame(records)
    df.to_csv("meta_records.csv", index=False)
    print("\nSaved meta-dataset to meta_records.csv")
    
    return records
    

def main():
    
    openml_datasets = load_openml_dataset(
        NumberOfFeatures=(1, 10000),
        NumberOfInstances=(1, 10000),
        NumberOfInstancesWithMissingValues=(0, 10000),
        NumberOfNumericFeatures=(1, 10000),
        NumberOfSymbolicFeatures=(1, 10000),
        max_datasets=2
    )

    records = algorithms_evaluation(algorithms=algorithms, datasets=openml_datasets)


if __name__ == "__main__":
    main()
