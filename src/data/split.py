from __future__ import annotations
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from src.features.columns import TARGETS
from src.data.groups import regression_group_ids


def regression_population(df: pd.DataFrame) -> pd.DataFrame:
    # Restrict regression to positive-rate bookings before splitting.
    return df.loc[df["ADR"] > 0].copy()


def make_split(
    df: pd.DataFrame,
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
    paid_bookings_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if task not in TARGETS:
        raise ValueError(f"task must be one of {sorted(TARGETS)}, got '{task}'")
    if task == "regression" and paid_bookings_only:
        df = regression_population(df)
    target = TARGETS[task]
    X, y = (df.drop(columns=[target]), df[target])
    if task == "regression":
        # Keep each booking-feature group entirely on one side of the holdout.
        groups = regression_group_ids(X)
        train, test = next(
            GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            ).split(X, y, groups)
        )
        return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])
    stratify = y
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def grouped_cv_folds(X_train, groups_train, n_splits=5):
    if not X_train.index.equals(groups_train.index):
        raise ValueError(
            "Training features and groups must have identical row alignment"
        )
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if groups_train.isna().any() or groups_train.nunique() < n_splits:
        raise ValueError("CV requires nonmissing groups and at least n_splits groups")
    return tuple(GroupKFold(n_splits=n_splits).split(X_train, groups=groups_train))
