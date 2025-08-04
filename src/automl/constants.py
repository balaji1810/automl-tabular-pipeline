from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from tabpfn import TabPFNRegressor


algorithms = [
    LGBMRegressor(n_jobs=-1),
    XGBRegressor(enable_categorical = True, n_jobs = -1),
    RandomForestRegressor(n_jobs=-1),
    DecisionTreeRegressor(),
    # HistGradientBoostingRegressor(),
    GradientBoostingRegressor(),
    MLPRegressor(),
    BayesianRidge(),
    LinearRegression(),
    SVR(),
    TabPFNRegressor(n_jobs=-1, ignore_pretraining_limits=True, device="auto")
    # device="cuda" if torch.cuda.is_available() else "cpu",
]

algorithms_dict = {algo.__class__.__name__: algo for algo in algorithms}
