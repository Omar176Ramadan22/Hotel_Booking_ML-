from sklearn.linear_model import Ridge
from scipy.stats import randint, loguniform, uniform
from xgboost import XGBRegressor
from src.preprocessing.pipeline import build_model_pipeline

MODELS = {"ridge": {"make": Ridge, "scale": True}}
MODELS["xgboost"] = {
    "make": lambda: XGBRegressor(tree_method="hist", n_jobs=-1),
    "scale": False,
    "search": {
        "model__n_estimators": randint(100, 501),
        "model__learning_rate": loguniform(0.02, 0.3),
        "model__max_depth": randint(3, 9),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.6, 0.4),
    },
    "search_description": {
        "model__n_estimators": "integers 100 to 500",
        "model__learning_rate": "loguniform 0.02 to 0.3",
        "model__max_depth": "integers 3 to 8",
        "model__subsample": "uniform 0.7 to 1.0",
        "model__colsample_bytree": "uniform 0.6 to 1.0",
    },
}


def get_pipeline(name, params=None):
    if name not in MODELS:
        raise ValueError(f"Unknown regression model: {name}")
    settings = MODELS[name]
    pipeline = build_model_pipeline(
        "regression", settings["make"](), scale=settings["scale"]
    )
    if params:
        pipeline.set_params(**params)
    return pipeline
