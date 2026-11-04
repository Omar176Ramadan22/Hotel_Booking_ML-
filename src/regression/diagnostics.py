import json
from pathlib import Path
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from src.data.preparation import load_config
from src.data.training import prepare_training_data as prepare_regression_data
import pandas as pd
from src.evaluation.metrics import regression_scores
from src.regression.artifacts import reproduction_hint

ARTIFACT_DIR = Path("artifacts")


def save_plots(actual, predicted, directory, model_name="ridge", source="cv"):
    actual, predicted = (np.asarray(actual), np.asarray(predicted))
    if actual.ndim != 1 or actual.shape != predicted.shape or (not len(actual)):
        raise ValueError(
            "Actual and predicted values must be matching non-empty vectors."
        )
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Plot values must be finite.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = np.linspace(0, len(actual) - 1, min(len(actual), 6000), dtype=int)
    a, p = (actual[rows], predicted[rows])
    caption = f"Source: H2 training-only grouped CV; {len(rows):,} of {len(actual):,} bookings shown.\nParameters selected using training CV; these are diagnostic, not independent test scores."
    if source == "train":
        caption = f"Source: fitted training data; {len(rows):,} of {len(actual):,} bookings shown.\nIn-sample diagnostic only; not cross-validation or held-out performance."
    prediction_label = "CV predicted" if source == "cv" else "Training predicted"
    paths = []
    label = "XGBoost" if model_name == "xgboost" else model_name.replace("_", " ").title()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(a, p, s=9, alpha=0.25, color="steelblue", rasterized=True)
    low, high = (min(a.min(), p.min()), max(a.max(), p.max()))
    ax.plot(
        [low, high], [low, high], color="black", linewidth=1, label="Perfect prediction"
    )
    ax.set(
        xlabel="Actual ADR (ADR units)",
        ylabel=f"{prediction_label} ADR (ADR units)",
        title=f"{label}: actual versus {prediction_label.lower()} ADR",
    )
    ax.legend()
    fig.text(0.12, 0.02, caption, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = directory / f"{model_name}_{source}_actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(p, a - p, s=9, alpha=0.25, color="steelblue", rasterized=True)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        xlabel=f"{prediction_label} ADR (ADR units)",
        ylabel="Actual minus predicted ADR (ADR units)",
        title=f"{label}: {source} residuals versus predicted ADR",
    )
    fig.text(0.12, 0.02, caption, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = directory / f"{model_name}_{source}_residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def save_feature_importance(pipeline, directory, model_name):
    names = pipeline.named_steps["preprocess"].named_steps["encode"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    if len(names) != len(values):
        raise ValueError("Feature names and importance values must match.")
    top = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False).head(15)
    result = directory / f"results/regression_{model_name}_feature_importance.csv"
    top.to_csv(result, index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ordered = top.iloc[::-1]
    ax.barh(ordered["feature"], ordered["importance"])
    label = "XGBoost" if model_name == "xgboost" else model_name.replace("_", " ").title()
    measure = "Normalized gain importance" if model_name == "xgboost" else "Normalized impurity decrease"
    ax.set(title=f"{label}: top 15 encoded features", xlabel=measure, ylabel="Encoded feature")
    fig.tight_layout()
    figure = directory / f"figures/{model_name}_feature_importance.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    print(result)
    print(figure)


def run_diagnostics(model_name="ridge", stage="tuned"):
    cfg = load_config()
    artifact = ARTIFACT_DIR / f"regression_{model_name}_{stage}.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing saved model: {artifact}")
    pipeline = joblib.load(artifact)
    prepared = prepare_regression_data()
    # These training-fold predictions are diagnostics after parameter selection.
    predictions = cross_val_predict(
        clone(pipeline), prepared.X_train, prepared.y_train, cv=prepared.baseline_cv, n_jobs=1
    )
    directory = Path(cfg["outputs"]["dir"])
    paths = save_plots(
        prepared.y_train.to_numpy(), predictions, directory / "figures", model_name
    )
    results = {
        "model": model_name,
        "stage": stage,
        "rows": len(predictions),
        "pooled_training_cv_metrics": regression_scores(prepared.y_train, predictions),
        "negative_predictions": int((predictions < 0).sum()),
    }
    (directory / "results").mkdir(parents=True, exist_ok=True)
    (directory / f"results/regression_{model_name}_diagnostics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    for path in paths:
        print(path)
    if model_name == "xgboost":
        save_feature_importance(pipeline, directory, model_name)
    print(json.dumps(results, indent=2))
    return results


def run_training_diagnostics(model_name="random_forest", stage="tuned"):
    artifact = ARTIFACT_DIR / f"regression_{model_name}_{stage}.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing {artifact}; {reproduction_hint(model_name)}")
    pipeline = joblib.load(artifact)
    prepared = prepare_regression_data()
    predictions = pipeline.predict(prepared.X_train)
    directory = Path(load_config()["outputs"]["dir"])
    paths = save_plots(prepared.y_train, predictions, directory / "figures", model_name, source="train")
    results = {
        "model": model_name, "stage": stage, "rows": len(predictions),
        "source": "in_sample_training_not_cv_or_holdout",
        "training_metrics": regression_scores(prepared.y_train, predictions),
        "negative_predictions": int((predictions < 0).sum()),
    }
    (directory / "results").mkdir(parents=True, exist_ok=True)
    path = directory / f"results/regression_{model_name}_training_diagnostics.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for figure in paths:
        print(figure)
    if hasattr(pipeline.named_steps["model"], "feature_importances_"):
        save_feature_importance(pipeline, directory, model_name)
    print(json.dumps(results, indent=2))
    return results
