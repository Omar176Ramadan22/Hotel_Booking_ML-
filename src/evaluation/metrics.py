from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_scores(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": sqrt(mse),
        "R2": float(r2_score(y_true, y_pred)),
    }
