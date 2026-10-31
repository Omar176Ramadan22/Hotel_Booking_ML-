"""One reproducible train/test split used by every experiment.

The test set is created ONCE here and must not be used for choosing
preprocessing options or hyperparameters (use cross-validation on the training
part for that). Cleaning in src/data/load.py is stateless, so it is safe to run
before this split.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.columns import TARGETS


def make_split(
    df: pd.DataFrame,
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test for `task`.

    X keeps every column except the target; the pipeline itself decides which
    columns are allowed (see src/features/columns.py). Classification is stratified
    so both parts keep the same cancellation rate.
    """
    target = TARGETS[task]
    X, y = df.drop(columns=[target]), df[target]
    stratify = y if task == "classification" else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state,
                            stratify=stratify)
