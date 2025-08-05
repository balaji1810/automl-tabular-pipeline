import pandas as pd
from sklearn.compose import ColumnTransformer
import logging

from .UniversalPreprocessor import build_universal_preprocessor

logger = logging.getLogger(__name__)


# Algorithm categories for preprocessing strategy selection
ALGORITHM_CATEGORIES = {
    # Tree-based algorithms (minimal preprocessing needed)
    'tree_based': [
        'RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor', 
        'DecisionTreeRegressor', 'GradientBoostingRegressor', 
        'HistGradientBoostingRegressor'
    ],
    
    # Neural networks (need heavy preprocessing)
    'neural': [
        'MLPRegressor', 'TabPFNRegressor'
    ],
    
    # Distance-based algorithms (very sensitive to scales)
    'distance_based': [
        'SVR', 'KNeighborsRegressor'
    ],
    
    # Linear models (moderate preprocessing)
    'linear': [
        'LinearRegression', 'BayesianRidge', 'SGDRegressor'
    ]
}

def determine_preprocessing_strategy(algorithm_name: str) -> str:
    """
    Determine the best preprocessing strategy based on the algorithms being used.
    
    Args:
        algorithm_names: List of algorithm class names
        
    Returns:
        Preprocessing strategy: 'conservative', 'balanced', or 'aggressive'
    """
    if not algorithm_name:
        return 'balanced'
    algo_name = algorithm_name
    if algo_name in ALGORITHM_CATEGORIES['tree_based']:
        return 'conservative'
    elif algo_name in ALGORITHM_CATEGORIES['neural']:
        return 'aggressive'
    elif algo_name in ALGORITHM_CATEGORIES['distance_based']:
        return 'aggressive'
    elif algo_name in ALGORITHM_CATEGORIES['linear']:
        return 'balanced'
    else:
        return 'balanced'


def build_algorithm_aware_preprocessor(
    X: pd.DataFrame, 
    algorithm_name: str,
) -> ColumnTransformer:
    """
    Build a preprocessor that's optimized for the specific algorithms being used.
    
    Args:
        X: Input DataFrame
        algorithm_name: List of algorithm class names that will be used
        custom_strategy: Override automatic strategy selection
        
    Returns:
        ColumnTransformer optimized for the given algorithms
    """
    strategy = determine_preprocessing_strategy(algorithm_name)
    logger.info(f"Auto-selected preprocessing strategy: {strategy} for algorithms: {algorithm_name}")
    
    # Use the universal preprocessor with the determined strategy
    preprocessor = build_universal_preprocessor(X, preprocessing_strategy=strategy)
    
    return preprocessor