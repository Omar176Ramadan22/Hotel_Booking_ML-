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
| Cleaning (strip text, NULL -> NaN, duplicates, ADR > 1000, zero guests) | `src/data/load.py` | No | Before the split |
| Fitted preprocessing (imputers, scaler, encoder, top agents) | `pipeline.py` | **Yes** | Inside the pipeline, fitted on training data only |

## Missing-value strategy

| Column | Missing (after cleaning) | Decision |
|---|---|---|
| `Children` | 4 rows | fill with 0 (modeling assumption: missing = no children) |
| `Country` | 3 rows | fill with `"Unknown"`; rare / unseen countries pooled by the encoder (`min_frequency=50`) |
| `Agent` | ~10% | create `HasAgent` **first**, then drop raw ID; optional `top_n_agents=N` keeps the N most frequent agents + `Other` + `None` |
| `Company` | ~95% | create `HasCompany`, drop raw ID |

Numeric leftovers: median imputer. Categorical leftovers: `"Unknown"`.
Unseen categories at inference never crash (`handle_unknown="infrequent_if_exist"`).
Required fields missing from a new booking raise a clear error on purpose.

## Options

* `scale` - StandardScaler on numeric columns (not needed for tree models).
* `top_n_agents` - 0 = baseline (flag only); e.g. 20 for the agent experiment.
* `include_borderline` - adds `RequiredCarParkingSpaces`, `TotalOfSpecialRequests` for the leakage ablation.
* `min_category_freq` - rare category threshold for one-hot encoding.
