from pathlib import Path
from src.data.load import load_raw
from src.data.preparation import load_config, prepare_regression_data


def run_analyze():
    cfg = load_config()
    raw = load_raw(Path(cfg["dataset"]["path"]))
    prepared = prepare_regression_data()
    print(f"Raw rows: {len(raw):,}; columns: {len(raw.columns)}")
    for step in prepared.cleaning_log:
        print(f"{step['step']}: {step['rows_removed']:,} removed")
    print(
        f"Training rows: {len(prepared.X_train):,}; test rows: {len(prepared.X_test):,}"
    )
    print(
        f"Overlapping groups: {len(set(prepared.groups_train) & set(prepared.groups_test))}"
    )
    print(raw.isna().sum().to_string())
