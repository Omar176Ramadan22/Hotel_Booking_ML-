"""Reusable, leakage-safe sklearn preprocessing pipeline.

    raw booking(s)
      -> BookingFeatureBuilder   (HasAgent/HasCompany, Children->0, Country->'Unknown')
      -> ColumnTransformer       (selects ONLY the task's allowed columns)
           numeric     : median imputer -> StandardScaler (optional)
           categorical : 'Unknown' imputer -> OneHotEncoder (rare/unseen -> 'infrequent')

Everything that learns from data (median, scaler mean/std, category lists, top
agents) is fitted when `.fit` is called on the TRAINING split, and nothing else.
Wrap the preprocessor and a model in one Pipeline so cross-validation re-fits
all of it inside every fold:

    pipe = build_model_pipeline("regression", RandomForestRegressor(random_state=42))
    pipe.fit(X_train, y_train)          # train split only
    joblib.dump(pipe, "artifacts/regression_pipeline.joblib")
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.booking_features import BookingFeatureBuilder
from src.features.columns import feature_columns


DEFAULT_TOP_N_AGENTS = 20


def build_preprocessor(
    task: str,
    top_n_agents: int = DEFAULT_TOP_N_AGENTS,
    scale: bool = True,
    include_borderline: bool = False,
    min_category_freq: int = 50,
) -> Pipeline:
    """Return an UNFITTED preprocessing pipeline for `task`.

    scale=True for distance/gradient models (Linear/Ridge, KNN, SVM, MLP);
    scale=False for tree models, which do not need it.
    top_n_agents: keep the N most frequent agents (learned on training data only),
    pool the rest as 'Other'; default 20. Use 0 for the flag-only baseline.
    min_category_freq: categories seen fewer times than this in training (e.g.
    rare countries) are pooled into one 'infrequent' column; categories never
    seen in training are pooled there too instead of crashing at inference.
    """
    numeric, categorical = feature_columns(task, top_n_agents, include_borderline)

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_category_freq,
            sparse_output=False,
        )),
    ]

    encoder = ColumnTransformer(
        [("num", Pipeline(numeric_steps), numeric),
         ("cat", Pipeline(categorical_steps), categorical)],
        remainder="drop",  # anything not listed (leaky or new columns) is discarded
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("features", BookingFeatureBuilder(top_n_agents=top_n_agents)),
        ("encode", encoder),
    ])


def build_model_pipeline(task: str, model, **preprocessor_options) -> Pipeline:
    """Preprocessor + estimator as ONE object (what gets tuned, saved and loaded)."""
    return Pipeline([
        ("preprocess", build_preprocessor(task, **preprocessor_options)),
        ("model", model),
    ])
