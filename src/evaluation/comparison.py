from pathlib import Path
import pandas as pd


def selected_result(path="outputs/tables/regression_model_comparison.csv"):
    table = pd.read_csv(path)
    selected = table.loc[table["SELECTED"].astype(str).str.lower() == "true"]
    if len(selected) != 1:
        raise ValueError("Comparison CSV must identify exactly one selected model.")
    return selected.iloc[0].to_dict()


def show_results():
    path = Path("outputs/tables/regression_model_comparison.csv")
    print(pd.read_csv(path).to_string(index=False))
    print(f"Selected model: {selected_result(path)['model']}; test metrics are historical records.")


def run_compare(directory="outputs"):
    directory = Path(directory)
    cv = pd.read_csv(directory / "tables/regression_cv_results.csv")
    recorded = pd.read_csv(directory / "tables/regression_test_results.csv")
    columns = ["model", "cv_rmse_mean", "cv_mae_mean", "cv_r2_mean", "fit_time_s"]
    tuned = cv.loc[cv["stage"] == "tuned", columns]
    if tuned.empty or tuned["model"].duplicated().any():
        raise ValueError("Expected one tuned CV row per model.")
    recorded = recorded.loc[recorded["stage"] == "tuned", ["model", "test_rmse"]]
    table = tuned.merge(recorded, on="model", how="left", validate="one_to_one")
    # Held-out metrics are displayed but never used in the ordering.
    table = table.sort_values(["cv_rmse_mean", "cv_mae_mean", "fit_time_s"], kind="stable").reset_index(drop=True)
    table["SELECTED"] = table.index == 0
    table.to_csv(directory / "tables/regression_model_comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"Selected model: {table.loc[0, 'model']} (lowest tuned CV RMSE, then CV MAE, then fit time)")
    return table
