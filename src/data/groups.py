from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import pandas as pd
from src.features.booking_features import BookingFeatureBuilder
from src.features.columns import feature_columns


def regression_group_keys(df: pd.DataFrame) -> pd.Series:
    # Ignore learned top-agent categories so related bookings stay in one group.
    numeric, categorical = feature_columns("regression", top_n_agents=0)
    features = BookingFeatureBuilder(top_n_agents=0).fit_transform(df)
    missing = set(numeric + categorical) - set(features.columns)
    if missing:
        raise ValueError(f"Missing regression group columns: {sorted(missing)}")

    def number(value):
        if pd.isna(value):
            return None
        try:
            value = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(f"Invalid numeric group value: {value!r}") from exc
        if not value.is_finite():
            raise ValueError("Numeric group values must be finite or missing")
        # Preserve distinct long numeric values when building canonical keys.
        with localcontext() as context:
            context.prec = max(28, len(value.as_tuple().digits))
            return str(value.normalize()) if value else "0"

    values = features[numeric + categorical].copy()
    for col in numeric:
        values[col] = values[col].map(number)
    for col in categorical:
        values[col] = values[col].map(lambda v: None if pd.isna(v) else str(v))
    prefix = ["regression-flag-policy-v1", numeric, categorical]
    return pd.Series(
        [
            json.dumps([prefix, list(row)], ensure_ascii=True, separators=(",", ":"))
            for row in values.itertuples(index=False, name=None)
        ],
        index=df.index,
        name="regression_group_key",
        dtype="str",
    )


def regression_group_ids(df: pd.DataFrame) -> pd.Series:
    keys = regression_group_keys(df)
    mapping = {
        key: hashlib.sha256(key.encode("utf-8")).hexdigest() for key in keys.unique()
    }
    return keys.map(mapping).rename("regression_group")
