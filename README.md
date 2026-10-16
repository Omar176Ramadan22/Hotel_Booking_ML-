# Hotel Booking Demand — End-to-End Machine Learning Project

CSE382 — Introduction to Machine Learning

This repository contains the team's end-to-end machine learning project using the **City Hotel (H2)** Hotel Booking Demand dataset.

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
│   └── 01_eda.ipynb
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

### Milestone 1 — Data preparation and initial baseline experiments
- [x] Repository structure created
- [x] H2 dataset added
- [x] Initial EDA notebook created
- [ ] Data-quality analysis completed
- [ ] Preprocessing strategy finalized
- [ ] Feature engineering finalized
- [ ] Baseline experiments completed

### Milestone 2 — Complete final project
- [ ] Regression models
- [ ] Classification models
- [ ] Clustering
- [ ] Dimensionality reduction where justified
- [ ] Hyperparameter tuning
- [ ] Model comparison
- [ ] Leakage audit
- [ ] Saved inference artifacts
- [ ] Final application
- [ ] Tests
- [ ] Final report and presentation

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
notebooks/01_eda.ipynb
```

The notebook is for exploration. The final implementation will live under `src/`; the project specification does not accept a notebook-only solution.

## Main application

The final project will expose functionality through `main.py`. Exact commands will be added as the implementation develops.

Planned examples:

```bash
python main.py analyze
python main.py train --task regression
python main.py train --task classification
python main.py tune --task regression
python main.py evaluate
python main.py cluster
python main.py predict --task regression
python main.py predict --task classification
```

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
