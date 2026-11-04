import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data.training import prepare_training_data
from src.evaluation.metrics import regression_scores
from src.regression.models import get_pipeline

VARIANTS = ("original", "standardized", "pca_95")
MODELS = ("ridge", "xgboost")


def experiment_pipeline(name, parameters, variant, seed=42):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown PCA experiment variant: {variant}")
    pipeline = get_pipeline(name, {**parameters, "model__random_state": seed})
    # The scaling-only control separates PCA from scaling effects.
    if variant != "original":
        pipeline.steps.insert(1, ("pca_scale", StandardScaler()))
    if variant == "pca_95":
        pipeline.steps.insert(2, ("pca", PCA(n_components=0.95, svd_solver="full")))
    return pipeline


def estimate_resources(prepared):
    estimates = []
    for name in MODELS:
        pipeline = joblib.load(f"artifacts/regression_{name}_tuned.joblib")
        probe = pipeline.named_steps["preprocess"].transform(prepared.X_train.iloc[:128])
        features = probe.shape[1]
        largest_fold = max(len(fit) for fit, _ in prepared.baseline_cv)
        matrix_mib = largest_fold * features * 8 / 1024**2
        estimates.append({"model": name, "features_estimate": features,
                          "dense_fit_matrix_MiB": matrix_mib,
                          "rough_working_arrays_MiB": 8 * matrix_mib})
    print(json.dumps(estimates, indent=2), flush=True)
    print("Sequential experiment: 2 models x 3 variants x 5 folds = 30 fits. "
          "Allow several minutes; full-SVD PCA and rotated XGBoost inputs can cost more than the original fits. "
          "The first completed fold gives a measured runtime check. Estimates exclude interpreter/tree overhead.", flush=True)
    return estimates


def compare_folds(prepared, recipes, directory, artifact_directory, seed=42):
    directory, artifact_directory = Path(directory), Path(artifact_directory)
    if (directory / "fold_results.csv").exists():
        raise FileExistsError("Experiment results already exist; choose a new --output-dir.")
    directory.mkdir(parents=True, exist_ok=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    rows, variance, fold_records = [], [], []
    for fold, (fit, validation) in enumerate(prepared.baseline_cv, 1):
        groups = prepared.groups_train
        if not set(groups.iloc[fit]).isdisjoint(groups.iloc[validation]):
            raise ValueError("PCA experiment groups overlap.")
        membership = {"fit": prepared.X_train.index[fit].tolist(),
                      "validation": prepared.X_train.index[validation].tolist()}
        fold_records.append({"fold": fold, "fit_rows": len(fit), "validation_rows": len(validation),
                             "membership_sha256": hashlib.sha256(json.dumps(membership).encode()).hexdigest()})
    protocol = {"models": recipes, "seed": seed, "training_rows": len(prepared.X_train),
                "folds": fold_records, "variance_threshold": 0.95, "solver": "full",
                "representation": "existing dense encoding; experimental full-column scaling in both paired controls",
                "selection": "predefined 95% variance; no target-based component search",
                "artifacts": str(artifact_directory), "heldout_used": False}
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    for name, recipe in recipes.items():
        for fold, (fit, validation) in enumerate(prepared.baseline_cv, 1):
            for variant in VARIANTS:
                pipeline = experiment_pipeline(name, recipe["best_params"], variant, seed)
                start = perf_counter()
                pipeline.fit(prepared.X_train.iloc[fit], prepared.y_train.iloc[fit])
                fit_time = perf_counter() - start
                start = perf_counter()
                predicted = pipeline.predict(prepared.X_train.iloc[validation])
                predict_time = perf_counter() - start
                before = len(pipeline.named_steps["preprocess"].named_steps["encode"].get_feature_names_out())
                after, retained = before, 1.0
                if variant == "pca_95":
                    pca = pipeline.named_steps["pca"]
                    after = int(pca.n_components_)
                    retained = float(pca.explained_variance_ratio_.sum())
                    for component, cumulative in enumerate(np.cumsum(pca.explained_variance_ratio_), 1):
                        variance.append({"model": name, "fold": fold, "component": component,
                                         "cumulative_variance": float(cumulative)})
                    joblib.dump(pipeline, artifact_directory / f"{name}_pca_fold_{fold}.joblib")
                metrics = {key.lower(): value for key, value in regression_scores(prepared.y_train.iloc[validation], predicted).items()}
                row = {"model": name, "variant": variant, "fold": fold, **metrics,
                       "features_before": before, "features_after": after, "variance_retained": retained,
                       "fit_time_s": fit_time, "predict_time_s": predict_time}
                rows.append(row)
                print(f"{name} {variant} fold {fold}: RMSE={row['rmse']:.6f}; "
                      f"features {before}->{after}; fit={fit_time:.2f}s; predict={predict_time:.3f}s", flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(directory / "fold_results.csv", index=False)
    summary = folds.groupby(["model", "variant"], sort=False).agg(
        cv_rmse=("rmse", "mean"), cv_rmse_std=("rmse", "std"),
        cv_mae=("mae", "mean"), cv_r2=("r2", "mean"),
        features_before_min=("features_before", "min"), features_before_max=("features_before", "max"),
        features_after_min=("features_after", "min"), features_after_max=("features_after", "max"),
        variance_retained_mean=("variance_retained", "mean"),
        fit_time_s_mean=("fit_time_s", "mean"), predict_time_s_mean=("predict_time_s", "mean"),
    ).reset_index()
    summary.to_csv(directory / "summary.csv", index=False)
    pd.DataFrame(variance).to_csv(directory / "explained_variance.csv", index=False)
    plot_results(summary, pd.DataFrame(variance), directory)
    print(summary.to_string(index=False), flush=True)
    return summary


def plot_results(summary, variance, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, summary["model"].unique()):
        for fold, values in variance.loc[variance["model"] == name].groupby("fold"):
            ax.plot(values["component"], values["cumulative_variance"], label=f"Fold {fold}")
        ax.axhline(0.95, color="black", linestyle="--", label="95% threshold")
        ax.set(title=name, xlabel="Retained principal components", ylabel="Cumulative training-fold variance", ylim=(0, 1.01))
        ax.legend(fontsize=7)
    fig.suptitle("PCA: retained portion of each training-fold variance curve")
    fig.tight_layout()
    fig.savefig(directory / "explained_variance.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, summary["model"].unique()):
        values = summary.loc[summary["model"] == name]
        ax.bar(values["variant"], values["cv_rmse"], yerr=values["cv_rmse_std"], capsize=4)
        ax.set(title=name, ylabel="Mean grouped CV RMSE", xlabel="Experimental representation")
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Training-only comparison; bars show fold mean and sample SD")
    fig.tight_layout()
    fig.savefig(directory / "cv_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Separate training-only PCA comparison")
    parser.add_argument("--output-dir", default="outputs/experiments/pca")
    parser.add_argument("--artifact-dir", default="artifacts/experiments/pca")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    prepared = prepare_training_data()
    estimates = estimate_resources(prepared)
    if args.estimate_only:
        return
    if max(row["rough_working_arrays_MiB"] for row in estimates) > 2048:
        raise ValueError("Estimated working arrays exceed 2 GiB; review resources before running.")
    recipes = {name: json.loads(Path(f"outputs/results/regression_{name}_best_params.json").read_text()) for name in MODELS}
    compare_folds(prepared, recipes, args.output_dir, args.artifact_dir)


if __name__ == "__main__":
    main()
