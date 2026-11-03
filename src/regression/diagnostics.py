import json
from pathlib import Path
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict
from src.data.preparation import load_config, prepare_regression_data
from src.evaluation.metrics import regression_scores

ARTIFACT_DIR = Path("artifacts")


def save_plots(actual, predicted, directory, model_name="ridge"):
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
    paths = []
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(a, p, s=9, alpha=0.25, color="steelblue", rasterized=True)
    low, high = (min(a.min(), p.min()), max(a.max(), p.max()))
    ax.plot(
        [low, high], [low, high], color="black", linewidth=1, label="Perfect prediction"
    )
    ax.set(
        xlabel="Actual ADR (ADR units)",
        ylabel="CV predicted ADR (ADR units)",
        title="Ridge: actual versus CV predicted ADR",
    )
    ax.legend()
    fig.text(0.12, 0.02, caption, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = directory / f"{model_name}_cv_actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(p, a - p, s=9, alpha=0.25, color="steelblue", rasterized=True)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        xlabel="CV predicted ADR (ADR units)",
        ylabel="Actual minus predicted ADR (ADR units)",
        title="Ridge: CV residuals versus predicted ADR",
    )
    fig.text(0.12, 0.02, caption, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = directory / f"{model_name}_cv_residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def run_diagnostics(model_name="ridge", stage="tuned"):
    cfg = load_config()
    artifact = ARTIFACT_DIR / f"regression_{model_name}_{stage}.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing saved model: {artifact}")
    pipeline = joblib.load(artifact)
    prepared = prepare_regression_data()
    # These training-fold predictions are diagnostics after parameter selection.
    predictions = cross_val_predict(
        pipeline, prepared.X_train, prepared.y_train, cv=prepared.baseline_cv, n_jobs=1
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
    return results
