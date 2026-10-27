"""One train/test split shared by every task.

The test set is cut ONCE, right after cleaning, and then left untouched until
the final evaluation. Model selection and tuning use cross-validation on the
training set only (the spec forbids tuning on the final test set).

Why stratified on IsCanceled: about 30% of the clean bookings are cancelled.
Stratifying keeps that same 30% in both parts, so the test set is a fair
sample and the classification scores are comparable between runs.

Why one split for all tasks: regression, classification and clustering then
share the same held-out bookings, so no task ever trains on rows another task
is tested on, and the final application can be evaluated on one test set.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_COLUMN = "IsCanceled"


def split_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify_column: str | None = STRATIFY_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a CLEAN booking table into (train, test).

    The same inputs always give the same split because the seed is fixed.
    Pass `stratify_column=None` for a plain random split.
    """
    stratify = df[stratify_column] if stratify_column else None
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train.copy(), test.copy()


def cv_folds(
    task: str, n_splits: int = 5, random_state: int = RANDOM_STATE
) -> KFold | StratifiedKFold:
    """Cross-validation folds to use INSIDE the training set for model selection.

    Classification folds are stratified like the main split. Regression folds
    are plain shuffled folds, because ADR is a number, not a class.
    """
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if task == "regression":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    raise ValueError(f"No supervised folds for task {task!r}")
