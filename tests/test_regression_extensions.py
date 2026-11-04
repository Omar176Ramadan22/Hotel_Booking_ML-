import ast
import inspect
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import main
from src.data import training
from src.data.preparation import load_config
from src.regression import train
from src.regression.artifacts import reproduction_hint
from src.regression.models import MODELS, get_pipeline
from src.data.groups import regression_group_ids
from src.data.split import grouped_cv_folds
from test_preprocessing import make_bookings


def test_refit_source_never_loads_a_fitted_model():
    tree = ast.parse(inspect.getsource(train.run_refit))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "load" for node in ast.walk(tree))


@pytest.mark.parametrize("name,parameters", [
    ("ridge", {"model__alpha": 0.0001, "preprocess__encode__num__polynomial__degree": 2}),
    ("xgboost", {"model__n_estimators": 3, "model__max_depth": 2}),
    ("random_forest", {"model__n_estimators": 3, "model__max_depth": 2}),
])
def test_refit_creates_fresh_unfitted_components_for_each_stage(tmp_path, monkeypatch, name, parameters):
    cfg = load_config()
    data = make_bookings()
    prep = SimpleNamespace(X_train=data.drop(columns="ADR"), y_train=data.ADR)
    monkeypatch.chdir(tmp_path)
    recipe = train.best_params_path(name)
    recipe.parent.mkdir(parents=True)
    recipe.write_text(json.dumps({"model": name, "frozen": True, "configuration": cfg, "best_params": parameters}))
    monkeypatch.setattr(train, "prepare_regression_data", lambda config: prep)
    factory = train.get_pipeline
    original_load = joblib.load
    created, previous_components = [], []

    def fresh_factory(model_name, params):
        pipeline = factory(model_name, params)
        components = [pipeline] + [value for value in pipeline.get_params().values() if isinstance(value, BaseEstimator)]
        for component in components:
            assert all(component is not previous for previous in previous_components)
            with pytest.raises(NotFittedError):
                check_is_fitted(component)
        created.append(pipeline)
        previous_components.extend(components)
        return pipeline

    monkeypatch.setattr(train, "get_pipeline", fresh_factory)
    monkeypatch.setattr(train.joblib, "load", lambda *a, **k: pytest.fail("Refit loaded a saved model"))
    output = tmp_path / "new_artifacts"
    paths = train.run_refit(name, output_dir=output)
    assert len(created) == len(paths) == 2
    for stage, path in zip(("baseline", "tuned"), paths):
        pipeline = original_load(path)
        assert path.parent == output
        assert np.isfinite(pipeline.predict(prep.X_train)).all()
        if stage == "tuned":
            for key, value in parameters.items():
                assert pipeline.get_params()[key] == value
        meta = json.loads(train.Path(str(path) + ".meta.json").read_text())
        assert meta["model"] == name and meta["stage"] == stage
        assert meta["seed"] == 42 and meta["training_rows"] == len(data)
        assert set(meta["versions"]) == {"python", "numpy", "pandas", "scikit-learn", "xgboost"}
    assert not (tmp_path / "outputs/tables").exists()


def test_refit_missing_recipe_and_unknown_name_fail_clearly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "prepare_regression_data", lambda **k: pytest.fail("Missing recipe prepared data"))
    with pytest.raises(FileNotFoundError, match="Missing best-parameters recipe.*run train/tune"):
        train.run_refit("random_forest", output_dir=tmp_path / "artifacts")
    with pytest.raises(ValueError, match="Unknown regression model: unknown"):
        train.run_refit("unknown", output_dir=tmp_path / "artifacts")


def test_cli_refit_all_skips_models_without_a_recipe(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for name in ("ridge", "xgboost"):
        path = train.best_params_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    called = []
    monkeypatch.setattr(train, "run_refit", lambda name, output_dir: called.append((name, output_dir)))
    output = str(tmp_path / "artifacts")
    monkeypatch.setattr("sys.argv", ["main.py", "refit", "--task", "regression", "--model", "all", "--output-dir", output])
    main.main()
    assert called == [("ridge", output), ("xgboost", output)]
    assert "random_forest: no best-parameters JSON; skipped." in capsys.readouterr().out


def test_missing_artifact_hint_depends_on_recipe_existence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "train/tune" in reproduction_hint("random_forest")
    path = train.best_params_path("random_forest")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    assert reproduction_hint("random_forest") == "run: python main.py refit --task regression --model random_forest"


def test_training_preparation_has_no_heldout_data(tmp_path, monkeypatch):
    cfg = load_config()
    data = make_bookings()
    monkeypatch.setattr(training, "load_config", lambda path: cfg)
    monkeypatch.setattr(training, "load_clean", lambda *a, **k: (data.copy(), []))
    prep = training.prepare_training_data()
    assert not any(hasattr(prep, name) for name in ("X_test", "y_test", "groups_test"))
    assert prep.X_train.index.equals(prep.y_train.index)
    assert prep.X_train.index.equals(prep.groups_train.index)
    assert len(prep.baseline_cv) == 5 and len(prep.tuning_cv) == 3
    for fit, validation in prep.baseline_cv + prep.tuning_cv:
        assert set(prep.groups_train.iloc[fit]).isdisjoint(prep.groups_train.iloc[validation])


@pytest.mark.parametrize("function", [train.run_train, train.run_tune])
def test_frozen_xgboost_blocks_training_and_tuning(function, monkeypatch):
    monkeypatch.setattr(train, "prepare_regression_data", lambda: pytest.fail("Frozen model prepared data"))
    with pytest.raises(ValueError, match="xgboost is frozen.*refit --task regression --model xgboost"):
        function("xgboost")


def test_random_forest_registry_defaults_and_search_documentation():
    settings = MODELS["random_forest"]
    pipeline = get_pipeline("random_forest")
    model = pipeline.named_steps["model"]
    assert settings["scale"] is False
    assert model.n_estimators == 100 and model.max_depth is None
    assert model.min_samples_leaf == 1 and model.n_jobs == -1
    assert set(settings["search_text"]) == set(settings["search"])
    assert all(isinstance(value, str) for value in settings["search_text"].values())
    assert settings["search"]["model__n_estimators"].support() == (100, 300)
    numeric = pipeline.named_steps["preprocess"].named_steps["encode"].transformers[0][1]
    assert "scale" not in numeric.named_steps
    assert load_config()["tuning"]["n_iter"]["random_forest"] == 15


def test_random_forest_tuning_two_tiny_candidates(tmp_path, monkeypatch):
    from scipy.stats import randint

    data = make_bookings()
    X, y = data.drop(columns="ADR"), data.ADR
    groups = regression_group_ids(X)
    prep = SimpleNamespace(
        X_train=X, y_train=y,
        baseline_cv=grouped_cv_folds(X, groups, 5),
        tuning_cv=grouped_cv_folds(X, groups, 3),
    )
    cfg = load_config()
    cfg["regression"]["frozen_models"].remove("random_forest")
    cfg["tuning"]["n_iter"]["random_forest"] = 2
    cfg["outputs"]["dir"] = str(tmp_path / "outputs")
    settings = MODELS["random_forest"]
    tiny = {"model__n_estimators": randint(2, 4), "model__max_depth": [2],
            "model__min_samples_leaf": [2], "model__max_features": ["sqrt"]}
    monkeypatch.setitem(settings, "search", tiny)
    monkeypatch.setitem(settings, "search_text", {key: str(value) for key, value in tiny.items()})
    monkeypatch.setattr(train, "load_config", lambda: cfg)
    monkeypatch.setattr(train, "prepare_regression_data", lambda: prep)
    monkeypatch.setattr(train, "ARTIFACT_DIR", tmp_path / "artifacts")
    row = train.run_tune("random_forest")
    details = json.loads((tmp_path / "outputs/results/regression_random_forest_best_params.json").read_text())
    pipeline = joblib.load(tmp_path / "artifacts/regression_random_forest_tuned.joblib")
    assert details["n_iter"] == 2 and details["seed"] == 42
    assert details["cv_folds"] == 3 and details["comparison_cv_folds"] == 5
    assert set(details["search_boundaries"]) == set(tiny)
    for key, value in details["best_params"].items():
        assert pipeline.get_params()[key] == value
    assert pipeline.named_steps["model"].n_estimators <= 3
    assert np.isfinite(pipeline.predict(X)).all()
    assert row["search_cv_folds"] == 3 and row["comparison_cv_folds"] == 5
    assert {"cv_rmse_mean", "train_rmse", "numeric_degree"} <= row.keys()


def test_existing_csv_field_text_is_preserved_when_adding_a_model(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text('model,stage,score\nridge,tuned,19.778548957460520\nxgboost,tuned,14.400593016645137\n')
    train._write_row(path, {"model": "random_forest", "stage": "tuned", "score": 15.0})
    text = path.read_text()
    assert "ridge,tuned,19.778548957460520" in text
    assert "xgboost,tuned,14.400593016645137" in text


@pytest.mark.parametrize("forest_rmse,selected", [(9.0, "random_forest"), (10.0, "xgboost")])
def test_comparison_selects_without_using_heldout_metrics(tmp_path, forest_rmse, selected):
    from src.evaluation.comparison import run_compare

    tables = tmp_path / "tables"
    tables.mkdir()
    cv = pd.DataFrame([
        ["ridge", "tuned", 10., 8., .5, 5.],
        ["xgboost", "tuned", 10., 8., .5, 4.],
        ["random_forest", "tuned", forest_rmse, 9., .5, 1.],
        ["ridge", "baseline", 1., 1., .9, 1.],
    ], columns=["model", "stage", "cv_rmse_mean", "cv_mae_mean", "cv_r2_mean", "fit_time_s"])
    recorded = pd.DataFrame([
        ["ridge", "tuned", 100.], ["xgboost", "tuned", 200.], ["random_forest", "tuned", 1.],
    ], columns=["model", "stage", "test_rmse"])
    cv_path, recorded_path = tables / "regression_cv_results.csv", tables / "regression_test_results.csv"
    cv.to_csv(cv_path, index=False)
    recorded.to_csv(recorded_path, index=False)
    before = cv_path.read_bytes(), recorded_path.read_bytes()
    result = run_compare(tmp_path)
    assert result.loc[result["SELECTED"], "model"].tolist() == [selected]
    assert len(result) == 3 and int(result["SELECTED"].sum()) == 1
    assert (cv_path.read_bytes(), recorded_path.read_bytes()) == before
    assert (tables / "regression_model_comparison.csv").exists()
