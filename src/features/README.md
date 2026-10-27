# Features

Train/test split and leak-free preprocessing, shared by all three tasks.

- `split.py`: `split_train_test(clean_df)` makes the one 80/20 split, stratified on
  `IsCanceled` with seed 42. `cv_folds(task)` gives the folds for model selection
  inside the training set. The test set stays untouched until final evaluation.
- `pipeline.py`: `build_preprocessor(task)` returns an unfitted scikit-learn pipeline
  that adds derived columns, keeps only booking-time features, imputes missing values,
  one-hot encodes categories and scales numbers. Leaky columns are listed in
  `LEAKAGE_COLUMNS` and can never be selected.

Typical use:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data.load import load_clean
from src.features.split import split_train_test, cv_folds
from src.features.pipeline import build_preprocessor, split_xy

df, _ = load_clean("data/H2.csv")
train, test = split_train_test(df)
X_train, y_train = split_xy(train, "classification")

model = Pipeline([
    ("prep", build_preprocessor("classification")),
    ("model", LogisticRegression(max_iter=2000)),
])
model.fit(X_train, y_train)   # preprocessing is fitted on the training rows only
```
