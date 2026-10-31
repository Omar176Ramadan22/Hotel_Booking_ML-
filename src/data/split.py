"""One reproducible train/test split used by every experiment.

The test set is created ONCE here and must not be used for choosing
preprocessing options or hyperparameters (use cross-validation on the training
part for that). Cleaning in src/data/load.py is stateless, so it is safe to run
before this split.

Regression population
---------------------
ADR = 0 rows (~1.8% of H2, mostly complimentary stays) are priced by policy, not
by the market, so the regression task uses PAID bookings only (ADR > 0). The
filter is applied BEFORE the split so training, validation and test all describe
the same population; filtering only the training rows would make test scores
worse for a reason unrelated to model quality. Classification keeps every
booking.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.columns import TARGETS


def regression_population(df: pd.DataFrame) -> pd.DataFrame:
    """Keep paid bookings only (ADR > 0). Use this for regression EDA/plots too."""
    return df[df["ADR"] > 0].reset_index(drop=True)


def make_split(
    df: pd.DataFrame,
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
    paid_bookings_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test for `task`.

    X keeps every column except the target; the pipeline itself decides which
    columns are allowed (see src/features/columns.py). Classification is stratified
    so both parts keep the same cancellation rate. For regression,
    `paid_bookings_only=True` (default) removes ADR <= 0 rows first.
    """
    if task not in TARGETS:
        raise ValueError(f"task must be one of {sorted(TARGETS)}, got '{task}'")
    if task == "regression" and paid_bookings_only:
        df = regression_population(df)
    target = TARGETS[task]
    X, y = df.drop(columns=[target]), df[target]
    stratify = y if task == "classification" else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state,
                            stratify=stratify)
