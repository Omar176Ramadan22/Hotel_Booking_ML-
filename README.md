# Hotel Booking Demand — End-to-End Machine Learning Project

CSE382 — Introduction to Machine Learning

This repository contains the team's end-to-end machine learning project using the **City Hotel (H2)** Hotel Booking Demand dataset.

## Repository and submission revision

GitHub repository: https://github.com/Omar176Ramadan22/Hotel_Booking_ML-

Branch for review: `ridge-regression`.

Frozen implementation commit: [`4e9581b27aaabd47b93fed257dd5759a6ac06a60`](https://github.com/Omar176Ramadan22/Hotel_Booking_ML-/commit/4e9581b27aaabd47b93fed257dd5759a6ac06a60). The following documentation commit records this revision without changing code or saved results.

Submitted ZIP commit (expanded by `git archive`): `$Format:%H$`

Create the submission from the reviewed commit with Git so only tracked source, documentation, and evaluation outputs are included:

```powershell
git archive --format=zip --output=../Hotel_Booking_submission.zip HEAD
```

The tracked `.gitattributes` rule substitutes the exact archived commit SHA into this README in that ZIP. Trained artifacts and `work/` are untracked and excluded. Keep regenerated artifacts available locally for the live discussion.

## Required tasks

- ADR regression
- Booking cancellation classification (`IsCanceled`)
- Unsupervised clustering
- Dimensionality reduction where justified
- Model comparison
- Hyperparameter tuning
- Reproducible preprocessing and inference
- Final usable CLI or GUI/web/desktop application

## Dataset

- Dataset: City Hotel — H2
- Rows: 79,330 bookings
- Source variables: 31
- Required project dataset: `data/H2.csv`

The project specification requires H2 only and does not allow merging H1 and H2 for the required experiments.

## Repository structure

```text
hotel-booking-ml/
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── .gitignore
├── data/
│   ├── H2.csv
│   └── README.md
├── notebooks/
│   └── 01_data_understanding_cleaning_fixed.ipynb
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── regression/
│   ├── classification/
│   ├── clustering/
│   ├── tuning/
│   ├── evaluation/
│   └── inference/
├── artifacts/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── results/
└── tests/
```

## Current status

Ridge is finished and frozen: baseline/tuned models, CV/test tables, diagnostics, input validation, saved-model prediction, and recipe-based refitting. The test suite passes 136 tests. Existing XGBoost baseline/tuned experiments are also recorded in the regression CSVs.

Read [the Ridge report](outputs/reports/ridge_model_report.md) for the baseline/tuned comparison, leakage audit, evaluation history, limitations, and team proposals. The test set was previously inspected; it is not a blind holdout. Do not change Ridge code, settings, artifacts, or results to improve its score.

PCA is deferred. Its later separate experiment must fit PCA inside training-fold pipelines, use the frozen CV folds, preserve existing files, and avoid test-set scoring. Classification/LDA, clustering, a third regression model, the complete comparison/selection process, and the final report/presentation remain team work.

## Environment setup

Recommended Python version: Python 3.10+.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Initial EDA

Open the repository in VS Code and launch:

```text
notebooks/01_data_understanding_cleaning_fixed.ipynb
```

The notebook is for exploration. The final implementation will live under `src/`; the project specification does not accept a notebook-only solution.

## Main application

From the repository root, check the dataset and tests:

```powershell
python main.py analyze
python -m pytest -q
```

On a fresh clone, regenerate both frozen Ridge pipelines and their matching bounds files without a search or evaluation:

```powershell
python main.py refit --task regression --model ridge
python main.py predict --task regression --model ridge --input data/example_booking.json
```

For interactive entry, omit `--input`. Predictions load saved artifacts; they do not retrain. Inputs outside training bounds are rejected, as are unseen categories. The interface normalizes the supported categorical fields and validates arrival dates/weeks. Years after 2017 are currently rejected; alternatives in the report are proposals, not implemented policies.

`train` and `tune` reject frozen Ridge; batch `--model all` skips it. Use `refit` to reproduce its artifacts. Existing Ridge CV/test outputs and diagnostic figures are checked in; do not rerun evaluation or diagnostics to overwrite these frozen results.

XGBoost training/tuning/evaluation is implemented separately. Classification and clustering CLI tasks remain scaffolds even where teammates have exploratory notebooks/configuration.

Every later regression model should reuse the shared preparation:

```python
from src.data.preparation import prepare_regression_data

prepared = prepare_regression_data("config.yaml")
# Use X_train/y_train and the frozen grouped folds for fitting and selection.
# baseline_cv has 5 folds; tuning_cv has 3 folds for future unfrozen searches.
```

The historical tuned Ridge used five folds for both search and reporting; it did not use the later 3/5 protocol. Groups exclude targets/outcomes and use permitted inputs before learned transforms. Learned medians, category lists, agent lists, and scaling stay inside the modeling pipeline.

The earlier [data-preparation audit](outputs/reports/data_preparation_audit.md) is retained as historical context. Current model status and evaluation limitations are documented in [the Ridge report](outputs/reports/ridge_model_report.md).

## Team contributions

Each team member should work from their own GitHub account and make meaningful incremental commits.

| Member | Responsibility | GitHub username |
|---|---|---|
| Member 1 | Data understanding & EDA | TBD |
| Member 2 | Preprocessing & feature engineering | TBD |
| Member 3 | Regression | TBD |
| Member 4 | Classification | TBD |
| Member 5 | Clustering / integration | TBD |

> Update this table after the team agrees on responsibilities.

## Important project rules

- Use H2.csv (City Hotel) for the required experiments.
- Do not merge H1 and H2 for the main experiments.
- Do not submit a notebook-only solution.
- Evaluate at least three models/algorithms for each required task.
- Compare baseline/untuned and tuned/optimized versions.
- Do not assume every dataset column is safe to use as a predictor.
- Audit for direct and indirect data leakage.
- Keep preprocessing and inference reproducible.
- Do not hard-code predictions.
- Keep meaningful Git commits throughout development.

## References

Hotel Booking Demand dataset:
https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/

Required H2 CSV mirror:
https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv
