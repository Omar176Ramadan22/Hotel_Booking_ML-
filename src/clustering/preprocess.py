# preprocess.py
# Turns the clustering columns into numbers that the algorithms can use.

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

from src.clustering.features import NUMERIC_COLUMNS, FLAG_COLUMNS, CATEGORICAL_COLUMNS


def build_cluster_preprocessor():
    # numbers: log first (long tails), then scale
    numeric_steps = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_steps, NUMERIC_COLUMNS),
        ("flags", "passthrough", FLAG_COLUMNS),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                      sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    return preprocessor