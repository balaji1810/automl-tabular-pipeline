from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from tabpfn import TabPFNRegressor
import torch


# GPU-optimized algorithms configuration
def get_gpu_optimized_algorithms():
    """Get algorithms configured for GPU usage if available."""
    use_gpu = torch.cuda.is_available()
    
    algorithms = [
        # GPU-enabled algorithms
        LGBMRegressor(
            n_jobs=-1,
            device='gpu' if use_gpu else 'cpu',
            gpu_platform_id=0 if use_gpu else None,
            gpu_device_id=0 if use_gpu else None,
            verbose=-1
        ),
        XGBRegressor(
            enable_categorical=True, 
            n_jobs=-1,
            tree_method='gpu_hist' if use_gpu else 'hist',
            gpu_id=0 if use_gpu else None,
            verbosity=0
        ),
        
        # CPU-only algorithms (will run on CPU regardless)
        RandomForestRegressor(n_jobs=-1),
        DecisionTreeRegressor(),
        HistGradientBoostingRegressor(),
        GradientBoostingRegressor(),
        MLPRegressor(max_iter=500),  # Increased iterations for better convergence
        BayesianRidge(),
        LinearRegression(),
        SVR(),
        
        # TabPFN (GPU-enabled when available)
        TabPFNRegressor(
            n_jobs=4,  # Limit parallelism to avoid CPU overload
            device="cuda" if use_gpu else "cpu"  # Explicitly set GPU device
        )
    ]
    
    return algorithms

# Default algorithms (backward compatibility)
algorithms = get_gpu_optimized_algorithms()

algorithms_dict = {algo.__class__.__name__: algo for algo in algorithms}
