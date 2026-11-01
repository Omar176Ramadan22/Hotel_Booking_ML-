"""One reproducible train/test split used by every experiment.

The test set is created ONCE here and must not be used for choosing
preprocessing options or hyperparameters (use cross-validation on the training
part for that). Cleaning in src/data/load.py is stateless, so it is safe to run
before this split.

Regression population
---------------------
The team restricts regression to PAID bookings (ADR > 0), excluding 722 of the
53,042 cleaned H2 rows (1.361%). Complimentary stays are only one possible
explanation for zero ADR, not a verified label in the file. The
filter is applied BEFORE the split so training, validation and test all describe
the same population; filtering only the training rows would make test scores
worse for a reason unrelated to model quality. Classification keeps every
booking.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from src.features.columns import TARGETS
from src.data.groups import regression_group_ids


def regression_population(df: pd.DataFrame) -> pd.DataFrame:
    """Keep paid bookings only (ADR > 0). Use this for regression EDA/plots too."""
    return df.loc[df["ADR"] > 0].copy()


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
    `paid_bookings_only=True` (default) removes ADR <= 0 rows first, then uses
    GroupShuffleSplit on stateless allowed-feature groups. Original indexes stay
    aligned. Use prepare_regression_data for central configuration and CV folds.
    """
    if task not in TARGETS:
        raise ValueError(f"task must be one of {sorted(TARGETS)}, got '{task}'")
    if task == "regression" and paid_bookings_only:
        df = regression_population(df)
    target = TARGETS[task]
    X, y = df.drop(columns=[target]), df[target]
    if task == "regression":
        groups = regression_group_ids(X)
        train, test = next(GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state,
        ).split(X, y, groups))
        return X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    stratify = y
    return train_test_split(X, y, test_size=test_size, random_state=random_state,
                            stratify=stratify)


def grouped_cv_folds(X_train, groups_train, n_splits=5):
    """Materialize reusable positional folds using ONLY outer training rows.

    Pass the returned tuple as cv=... to future searches/scores. Keep the full
    preprocessor inside each model pipeline so it is refitted within each fold.
    """
    if not X_train.index.equals(groups_train.index):
        raise ValueError("Training features and groups must have identical row alignment")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if groups_train.isna().any() or groups_train.nunique() < n_splits:
        raise ValueError("CV requires nonmissing groups and at least n_splits groups")
    return tuple(GroupKFold(n_splits=n_splits).split(X_train, groups=groups_train))
