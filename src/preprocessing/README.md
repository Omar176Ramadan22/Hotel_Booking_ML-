# Preprocessing

`pipeline.py` builds the reusable, leakage-safe sklearn pipeline used by every
supervised experiment and by inference.

```python
from src.data.load import load_clean
from src.data.split import make_split
from src.preprocessing import build_model_pipeline

df, _ = load_clean("data/H2.csv")                     # stateless cleaning (safe before the split)
X_train, X_test, y_train, y_test = make_split(df, "regression")   # seed 42

pipe = build_model_pipeline("regression", model, scale=False)     # scale=True for KNN/SVM/MLP/Ridge
pipe.fit(X_train, y_train)                            # fitted on TRAIN only
```

## Two layers

| Layer | Where | Learns from data? | Runs |
|---|---|---|---|
| Cleaning (strip text, NULL -> NaN, duplicates, ADR > 1000 or < 0, zero guests, zero nights) | `src/data/load.py` | No | Before the split |
| Fitted preprocessing (imputers, scaler, encoder, top agents) | `pipeline.py` | **Yes** | Inside the pipeline, fitted on training data only |

`src/preprocessing/cleaning.py` (YAML-driven, used by the notebooks) applies the same
rules; `tests/test_load.py` checks both paths return identical data (53,042 rows).

## Regression population

Regression uses **paid bookings only (ADR > 0)**. The ~1.8% of bookings with `ADR = 0`
(mostly complimentary stays) are priced by policy, not by the market. `make_split`
applies the filter *before* splitting, so train and test describe the same population.
Classification keeps every booking. Switch off with `make_split(..., paid_bookings_only=False)`.

## Missing-value strategy

| Column | Missing (after cleaning) | Decision |
|---|---|---|
| `Children` | 4 rows | fill with 0 (modeling assumption: missing = no children) |
| `Country` | 3 rows | fill with `"Unknown"`; rare / unseen countries pooled by the encoder (`min_frequency=50`) |
| `Agent` | ~10% | create `HasAgent` **first**, then drop raw ID; `top_n_agents=20` (default) also keeps the 20 most frequent agents + `Other` + `None` |
| `Company` | ~95% | create `HasCompany`, drop raw ID |

Numeric leftovers: median imputer. Categorical leftovers: `"Unknown"`.
Unseen categories at inference never crash (`handle_unknown="infrequent_if_exist"`).
Required fields missing from a new booking raise a clear error on purpose.

## Options

* `scale` - StandardScaler on numeric columns (not needed for tree models).
* `top_n_agents` - default **20** (best result in validation checks); 0 = flag-only baseline for comparison.
* `include_borderline` - adds `RequiredCarParkingSpaces`, `TotalOfSpecialRequests` for the leakage ablation.
* `min_category_freq` - rare category threshold for one-hot encoding.
