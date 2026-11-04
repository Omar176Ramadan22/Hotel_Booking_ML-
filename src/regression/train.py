import json
from pathlib import Path
from time import perf_counter
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from src.data.preparation import load_config, prepare_regression_data as prepare_evaluation_data
from src.data.training import prepare_training_data as prepare_regression_data
from src.evaluation.metrics import regression_scores
from src.regression.models import get_pipeline
from src.regression.models import MODELS
from src.data.rules import write_training_limits
from src.features.columns import feature_columns
from src.regression.artifacts import best_params_path, reproduction_hint, write_metadata, plain_parameters

ARTIFACT_DIR = Path("artifacts")
SCORING = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def _check_frozen(model_name, cfg):
    if model_name in cfg["regression"].get("frozen_models", []):
        raise ValueError(
            f"{model_name} is frozen. Training and tuning are disabled for the finalized experiment; {reproduction_hint(model_name)}"
        )


def _write_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    read_options = {"dtype": str, "keep_default_na": False}
    rows = pd.read_csv(path, **read_options) if path.exists() else pd.DataFrame()
    if not rows.empty:
        same = (rows["model"] == row["model"]) & (rows["stage"] == row["stage"])
        rows = rows.loc[~same]
    pd.concat([rows, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def _save_model(name, stage, pipeline, prep, cfg, scores, cv_use, experiment_time):
    # Fit only outer training rows after CV.
    start = perf_counter()
    pipeline.fit(prep.X_train, prep.y_train)
    fit_time = perf_counter() - start
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / f"regression_{name}_{stage}.joblib"
    joblib.dump(pipeline, artifact)
    write_training_limits(artifact, prep.X_train, feature_columns("regression")[0])
    write_metadata(artifact, name, stage, pipeline, cfg["random_state"], len(prep.X_train))
    settings = pipeline.get_params()
    if name == "ridge":
        parameters = {
            "alpha": settings["model__alpha"],
            "numeric_degree": settings["preprocess__encode__num__polynomial__degree"],
        }
    else:
        keys = list(MODELS[name]["search"]) + [key for key in (
            "model__tree_method", "model__n_jobs", "model__random_state"
        ) if key in settings]
        parameters = {key.removeprefix("model__"): settings[key] for key in keys}
        parameters["numeric_degree"] = settings["preprocess__encode__num__polynomial__degree"]
    parameters = plain_parameters(parameters)
    row = {
        "model": name,
        "stage": stage,
        "cv_use": cv_use,
        "parameters": json.dumps(parameters),
        "numeric_degree": settings["preprocess__encode__num__polynomial__degree"],
        "comparison_cv_folds": len(prep.baseline_cv),
        "search_cv_folds": len(prep.tuning_cv) if stage == "tuned" else 0,
        "top_n_agents": settings["preprocess__features__top_n_agents"],
        "min_category_frequency": settings[
            "preprocess__encode__cat__encode__min_frequency"
        ],
        "cv_mae_mean": float(-scores["mae"].mean()),
        "cv_mse_mean": float(-scores["mse"].mean()),
        "cv_rmse_mean": float(-scores["rmse"].mean()),
        "cv_rmse_std": float(scores["rmse"].std()),
        "cv_r2_mean": float(scores["r2"].mean()),
        "fit_time_s": fit_time,
        "experiment_time_s": experiment_time + fit_time,
    }
    row.update(
        {
            f"train_{name.lower()}": value
            for name, value in regression_scores(
                prep.y_train, pipeline.predict(prep.X_train)
            ).items()
        }
    )
    _write_row(Path(cfg["outputs"]["dir"]) / "tables/regression_cv_results.csv", row)
    print(f"{name} {stage}: CV RMSE={row['cv_rmse_mean']:.4f}; saved pipeline")
    return row


def run_train(model_name):
    cfg = load_config()
    _check_frozen(model_name, cfg)
    if model_name == "random_forest":
        print("Random Forest baseline uses 100 unrestricted trees; its artifact may be about 260 MB.", flush=True)
    prep = prepare_regression_data()
    pipeline = get_pipeline(model_name, {"model__random_state": cfg["random_state"]})
    start = perf_counter()
    cv = cross_validate(
        pipeline,
        prep.X_train,
        prep.y_train,
        cv=prep.baseline_cv,
        scoring=SCORING,
        error_score="raise",
    )
    scores = {name: cv[f"test_{name}"] for name in SCORING}
    return _save_model(
        model_name,
        "baseline",
        pipeline,
        prep,
        cfg,
        scores,
        "baseline",
        perf_counter() - start,
    )


def run_tune(model_name):
    cfg = load_config()
    _check_frozen(model_name, cfg)
    prep = prepare_regression_data()
    pipeline = get_pipeline(model_name, {"model__random_state": cfg["random_state"]})
    settings = MODELS[model_name]
    grid = settings.get("search", cfg["tuning"].get(f"{model_name}_grid"))
    start = perf_counter()
    search_options = {}
    search_class = GridSearchCV
    if "search" in settings:
        search_class = RandomizedSearchCV
        search_options = {
            "n_iter": cfg["tuning"]["n_iter"][model_name],
            "random_state": cfg["random_state"],
        }
        if model_name == "random_forest":
            search_options["verbose"] = 2
    search = search_class(
        pipeline,
        grid,
        cv=prep.tuning_cv,
        scoring=SCORING,
        refit="rmse",
        n_jobs=1,
        error_score="raise",
        **search_options,
    )
    search.fit(prep.X_train, prep.y_train)
    selected = get_pipeline(
        model_name, {**search.best_params_, "model__random_state": cfg["random_state"]}
    )
    comparison = cross_validate(
        selected,
        prep.X_train,
        prep.y_train,
        cv=prep.baseline_cv,
        scoring=SCORING,
        error_score="raise",
    )
    elapsed = perf_counter() - start
    scores = {name: comparison[f"test_{name}"] for name in SCORING}
    directory = Path(cfg["outputs"]["dir"])
    (directory / "results").mkdir(parents=True, exist_ok=True)
    details = {
        "model": model_name,
        "best_params": plain_parameters(search.best_params_),
        "search_space": settings.get("search_text", settings.get("search_description", grid)),
        "optimization_metric": "RMSE",
        "cv_folds": len(prep.tuning_cv),
        "comparison_cv_folds": len(prep.baseline_cv),
        "comparison_score_use": "post_selection_comparison_not_independent",
        "best_cv_rmse": float(-search.best_score_),
        "n_iter": len(search.cv_results_["params"]),
        "seed": cfg["random_state"],
        "search_boundaries": search_boundaries(search.best_params_, grid),
        "preprocessing": {
            "numeric": "median imputation, polynomial features, scaling",
            "categorical": "Unknown imputation, one-hot encoding",
        },
        "configuration": cfg,
    }
    if not settings["scale"]:
        details["preprocessing"]["numeric"] = "median imputation, polynomial features"
    details["any_best_value_at_edge"] = any(item["at_edge"] is True for item in details["search_boundaries"].values())
    (directory / f"results/regression_{model_name}_best_params.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    row = _save_model(
        model_name,
        "tuned",
        selected,
        prep,
        cfg,
        scores,
        "post_selection_comparison",
        elapsed,
    )
    print("Best parameters:", search.best_params_)
    return row


def search_boundaries(parameters, space):
    boundaries = {}
    for name, value in plain_parameters(parameters).items():
        choices = space[name]
        if isinstance(choices, list):
            limits = [min(choices), max(choices)] if all(isinstance(item, (int, float)) for item in choices) else None
        else:
            lower, upper = choices.support()
            limits = [float(lower), float(upper)]
        boundaries[name] = {
            "value": value, "limits": limits,
            "at_edge": bool(value in limits) if limits is not None else None,
        }
    return boundaries


def run_refit(model_name, output_dir=None):
    if model_name not in MODELS:
        raise ValueError(f"Unknown regression model: {model_name}")
    recipe_path = best_params_path(model_name)
    if not recipe_path.exists():
        raise FileNotFoundError(f"Missing best-parameters recipe: {recipe_path}; {reproduction_hint(model_name)}")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("model") != model_name:
        raise ValueError(f"Expected a {model_name} recipe.")
    if model_name == "ridge" and recipe.get("frozen") is not True:
        raise ValueError("Expected a frozen Ridge recipe.")
    cfg = recipe["configuration"]
    # Reuse the recorded cleaning rules and split.
    prep = prepare_regression_data(config=cfg)
    directory = ARTIFACT_DIR if output_dir is None else Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for stage in ("baseline", "tuned"):
        params = {"model__random_state": cfg["random_state"]}
        if stage == "tuned":
            params.update(recipe["best_params"])
        pipeline = get_pipeline(model_name, params)
        pipeline.fit(prep.X_train, prep.y_train)
        artifact = directory / f"regression_{model_name}_{stage}.joblib"
        joblib.dump(pipeline, artifact)
        write_training_limits(artifact, prep.X_train, feature_columns("regression")[0])
        write_metadata(artifact, model_name, stage, pipeline, cfg["random_state"], len(prep.X_train))
        paths.append(artifact)
        print(f"{model_name} {stage}: fresh refit on {len(prep.X_train):,} training rows; saved {artifact}, bounds and metadata")
    return paths


def run_evaluate(model_name):
    get_pipeline(model_name)
    paths = {
        stage: ARTIFACT_DIR / f"regression_{model_name}_{stage}.joblib"
        for stage in ("baseline", "tuned")
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing saved pipelines: {', '.join(missing)}; {reproduction_hint(model_name)}"
        )
    cfg = load_config()
    if model_name in cfg["regression"].get("frozen_models", []):
        raise ValueError(
            f"{model_name} is finalized; held-out evaluation is disabled. "
            "Use: python main.py results --task regression for recorded results."
        )
    prep = prepare_evaluation_data()
    rows = []
    for stage, path in paths.items():
        pipeline = joblib.load(path)
        metrics = regression_scores(prep.y_test, pipeline.predict(prep.X_test))
        row = {
            "model": model_name,
            "stage": stage,
            **{f"test_{key.lower()}": value for key, value in metrics.items()},
        }
        _write_row(
            Path(cfg["outputs"]["dir"]) / "tables/regression_test_results.csv", row
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return result
