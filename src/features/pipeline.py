"""Leak-free feature selection and preprocessing for every task.

PREDICTION POINT
    All three tasks assume we know the booking as it looks when it is MADE
    (or shortly after), before the guest arrives. A column is only allowed as
    an input if its value is known at that moment.

LEAKAGE RULES (checked on H2 after cleaning)
    ReservationStatus      'Canceled'/'No-Show' exactly when IsCanceled = 1:
                           it IS the answer. Never an input.
    ReservationStatusDate  set when the booking ends (cancel or check-out),
                           i.e. after the outcome. Never an input.
    AssignedRoomType       given at check-in. Bookings whose assigned room
                           differs from the reserved one cancel only 5% of the
                           time vs 33% otherwise, because a cancelled guest is
                           rarely moved to another room. Indirect leak.
    BookingChanges         counted "before check-in or cancellation", so the
                           stored value is the final count, not the count at
                           prediction time. Excluded to stay safe.
    IsCanceled             the future outcome; not an input for ADR or
                           clustering either.

PREPROCESSING (everything that learns from data lives here)
    The stateless `BookingFeatures` step adds a few derived columns. The
    `ColumnTransformer` then imputes, encodes and scales. Because it is a
    scikit-learn object, `fit` is called on the training rows only and the
    learned values (medians, category lists, means) are reused unchanged on
    the test set and on new bookings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that must never reach any model (see the module docstring).
LEAKAGE_COLUMNS = [
    "ReservationStatus", "ReservationStatusDate", "AssignedRoomType", "BookingChanges",
]
TARGETS = {"regression": "ADR", "classification": "IsCanceled"}

MONTHS = {
    m: i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], start=1)
}

# ---- Feature groups (after BookingFeatures has run) ------------------------
NUMERIC_FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalMonthNumber", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
    "TotalNights", "Adults", "Children", "Babies", "TotalGuests",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "DaysInWaitingList", "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
    "HasAgent", "HasCompany",
]
# Few categories: fill the rare missing value with the most common category.
CATEGORICAL_FEATURES = [
    "Meal", "MarketSegment", "DistributionChannel", "ReservedRoomType",
    "DepositType", "CustomerType",
]
# Many categories: Country (166), Agent (223), Company (203). Rare values are
# grouped into one "infrequent" column so the model is not flooded with
# columns that appear a handful of times.
HIGH_CARDINALITY_FEATURES = ["Country"]
# For IDs, missing means "booked without an agent/company", which is real
# information, so it becomes its own category "none" instead of being guessed.
ID_FEATURES = ["Agent", "Company"]

MIN_CATEGORY_COUNT = 30  # a category needs this many training rows to get its own column


def task_features(task: str) -> dict[str, list[str]]:
    """Return the feature groups allowed as inputs for `task`.

    Every task uses the same booking-time features. Classification and
    clustering may also use ADR, because the price is agreed when the room is
    booked. Regression cannot, because ADR is its target.
    """
    if task not in ("regression", "classification", "clustering"):
        raise ValueError(f"Unknown task {task!r}")
    numeric = list(NUMERIC_FEATURES)
    if task in ("classification", "clustering"):
        numeric.append("ADR")
    groups = {
        "numeric": numeric,
        "categorical": list(CATEGORICAL_FEATURES),
        "high_cardinality": list(HIGH_CARDINALITY_FEATURES),
        "id": list(ID_FEATURES),
    }
    _check_no_leakage(task, groups)
    return groups


def _check_no_leakage(task: str, groups: dict[str, list[str]]) -> None:
    forbidden = set(LEAKAGE_COLUMNS) | {"IsCanceled"}
    if task == "regression":
        forbidden.add("ADR")
    used = {c for cols in groups.values() for c in cols}
    leaked = sorted(used & forbidden)
    if leaked:
        raise ValueError(f"Leakage: {leaked} cannot be inputs for {task}")


def split_xy(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate inputs from the target. The input table keeps every other column;
    the preprocessor selects the allowed ones and drops the rest."""
    target = TARGETS[task]
    return df.drop(columns=[target]), df[target]


class BookingFeatures(BaseEstimator, TransformerMixin):
    """Derived columns computed from one booking at a time.

    Nothing is learned in `fit`, so this step behaves the same on train,
    test and a single new booking.
    """

    NEW_COLUMNS = ["ArrivalMonthNumber", "TotalNights", "TotalGuests", "HasAgent", "HasCompany"]

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + self.NEW_COLUMNS, dtype=object)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["ArrivalMonthNumber"] = X["ArrivalDateMonth"].map(MONTHS)
        X["TotalNights"] = X["StaysInWeekendNights"] + X["StaysInWeekNights"]
        X["TotalGuests"] = X[["Adults", "Children", "Babies"]].fillna(0).sum(axis=1)
        X["HasAgent"] = X["Agent"].notna().astype(int)
        X["HasCompany"] = X["Company"].notna().astype(int)
        # scikit-learn imputers expect np.nan in plain object columns, not
        # pandas' pd.NA / string dtype.
        text_cols = X.select_dtypes(include=["object", "string"]).columns
        X[text_cols] = X[text_cols].astype(object).where(X[text_cols].notna(), np.nan)
        return X


def build_preprocessor(task: str, scale: bool = True,
                       min_category_count: int = MIN_CATEGORY_COUNT) -> Pipeline:
    """Unfitted preprocessing pipeline for `task`.

    Use it as the first step of a model pipeline, e.g.
    ``Pipeline([("prep", build_preprocessor("classification")), ("model", clf)])``
    so cross-validation refits it on each training fold automatically.
    `scale=False` skips standardisation (tree models do not need it).
    """
    groups = task_features(task)

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    def one_hot(**kw):
        return OneHotEncoder(sparse_output=False, **kw)

    columns = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), groups["numeric"]),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", one_hot(handle_unknown="ignore")),
            ]), groups["categorical"]),
            ("high_cardinality", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", one_hot(handle_unknown="infrequent_if_exist",
                                   min_frequency=min_category_count)),
            ]), groups["high_cardinality"]),
            ("id", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="none")),
                ("encode", one_hot(handle_unknown="infrequent_if_exist",
                                   min_frequency=min_category_count)),
            ]), groups["id"]),
        ],
        remainder="drop",  # anything not listed above (leaky columns, target) is dropped
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", BookingFeatures()), ("columns", columns)])
