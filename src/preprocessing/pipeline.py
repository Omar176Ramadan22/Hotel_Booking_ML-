from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from src.features.booking_features import BookingFeatureBuilder
from src.features.columns import feature_columns

DEFAULT_TOP_N_AGENTS = 20


def build_preprocessor(
    task,
    top_n_agents=DEFAULT_TOP_N_AGENTS,
    scale=True,
    include_borderline=False,
    min_category_freq=50,
    numeric_degree=1,
):
    numeric, categorical = feature_columns(task, top_n_agents, include_borderline)
    # Expand numeric inputs after imputation and before scaling.
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("polynomial", PolynomialFeatures(degree=numeric_degree, include_bias=False)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        (
            "encode",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=min_category_freq,
                sparse_output=False,
            ),
        ),
    ]
    encoder = ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline(categorical_steps), categorical),
        ],
        # Only explicitly allowed features reach the estimator.
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", BookingFeatureBuilder(top_n_agents=top_n_agents)),
            ("encode", encoder),
        ]
    )


# Keeping preprocessing and estimation together refits both inside each CV fold.
def build_model_pipeline(task, model, **preprocessor_options):
    return Pipeline(
        [
            ("preprocess", build_preprocessor(task, **preprocessor_options)),
            ("model", model),
        ]
    )
