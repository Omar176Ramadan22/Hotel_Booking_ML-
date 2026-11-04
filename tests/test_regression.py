from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
import main
from src.data.preparation import load_config, prepare_regression_data
from src.evaluation.metrics import regression_scores
from src.regression import train
from src.regression.models import get_pipeline


def test_regression_scores():
    scores = regression_scores([1, 2, 3], [1, 2, 4])
    assert scores == pytest.approx(
        {"MAE": 1 / 3, "MSE": 1 / 3, "RMSE": np.sqrt(1 / 3), "R2": 0.5}
    )


def test_ridge_pipeline_and_optional_parameters():
    pipeline = get_pipeline("ridge", {"model__alpha": 4})
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    assert isinstance(pipeline.named_steps["model"], Ridge)
    assert pipeline.named_steps["model"].alpha == 4
    numeric = (
        pipeline.named_steps["preprocess"].named_steps["encode"].transformers[0][1]
    )
    assert "scale" in numeric.named_steps
    with pytest.raises(ValueError, match="Unknown regression model"):
        get_pipeline("missing")


def test_train_real_data_writes_pipeline_and_cv_row(tmp_path, monkeypatch):
    prepared = prepare_regression_data()
    training_only = SimpleNamespace(
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        baseline_cv=prepared.baseline_cv,
    )
    monkeypatch.setattr(train, "prepare_regression_data", lambda: training_only)
    cfg = load_config()
    cfg["regression"]["frozen_models"] = []
    cfg["outputs"]["dir"] = str(tmp_path / "outputs")
    monkeypatch.setattr(train, "load_config", lambda: cfg)
    monkeypatch.setattr(train, "ARTIFACT_DIR", tmp_path / "artifacts")
    train.run_train("ridge")
    saved = joblib.load(tmp_path / "artifacts/regression_ridge_baseline.joblib")
    assert saved.named_steps["model"].random_state == cfg["random_state"]
    assert saved.named_steps["model"].n_features_in_ > 0
    rows = pd.read_csv(tmp_path / "outputs/tables/regression_cv_results.csv")
    assert len(rows) == 1
    assert rows.loc[0, "model"] == "ridge" and rows.loc[0, "stage"] == "baseline"
    assert rows.loc[0, "cv_rmse_mean"] > 0
    assert rows.loc[0, "fit_time_s"] > 0
    assert {
        "model",
        "stage",
        "parameters",
        "cv_mae_mean",
        "cv_mse_mean",
        "cv_rmse_mean",
        "cv_r2_mean",
        "fit_time_s",
    } <= set(rows)


def test_evaluate_missing_pipelines_fails_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ARTIFACT_DIR", tmp_path)

    def unexpected_preparation():
        pytest.fail("Missing artifacts should be checked before preparing test data")

    monkeypatch.setattr(train, "prepare_regression_data", unexpected_preparation)
    with pytest.raises(
        FileNotFoundError, match=r"Missing saved pipelines.*run: python main\.py refit --task regression --model ridge"
    ):
        train.run_evaluate("ridge")


def test_csv_replaces_matching_row_and_preserves_other_stages(tmp_path):
    path = tmp_path / "results.csv"
    train._write_row(path, {"model": "ridge", "stage": "baseline", "score": 10})
    train._write_row(path, {"model": "ridge", "stage": "tuned", "score": 8})
    train._write_row(path, {"model": "ridge", "stage": "baseline", "score": 9})
    rows = pd.read_csv(path)
    assert len(rows) == 2
    assert rows.set_index("stage")["score"].to_dict() == {"baseline": 9, "tuned": 8}


@pytest.mark.parametrize("command", ["train", "tune", "evaluate"])
@pytest.mark.parametrize("task", ["classification", "clustering"])
def test_unsupported_cli_tasks_exit_cleanly(command, task, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["main.py", command, "--task", task])
    main.main()
    assert "not implemented yet" in capsys.readouterr().out


def test_tuning_saves_settings_without_accessing_test_data(tmp_path, monkeypatch):
    import json
    from test_preprocessing import make_bookings
    from src.data.groups import regression_group_ids
    from src.data.split import grouped_cv_folds

    data = make_bookings()
    X, y = (data.drop(columns="ADR"), data.ADR)
    prep = SimpleNamespace(
        X_train=X,
        y_train=y,
        baseline_cv=grouped_cv_folds(X, regression_group_ids(X), 3),
        tuning_cv=grouped_cv_folds(X, regression_group_ids(X), 2),
    )
    cfg = load_config()
    cfg["regression"]["frozen_models"] = []
    cfg["outputs"]["dir"] = str(tmp_path)
    cfg["tuning"]["ridge_grid"] = {"model__alpha": [1.0, 10.0]}
    monkeypatch.setattr(train, "load_config", lambda: cfg)
    monkeypatch.setattr(train, "prepare_regression_data", lambda: prep)
    monkeypatch.setattr(train, "ARTIFACT_DIR", tmp_path / "artifacts")
    row = train.run_tune("ridge")
    settings = json.loads(
        (tmp_path / "results/regression_ridge_best_params.json").read_text()
    )
    saved = joblib.load(tmp_path / "artifacts/regression_ridge_tuned.joblib")
    assert saved.named_steps["model"].alpha == settings["best_params"]["model__alpha"]
    assert row["cv_use"] == "post_selection_comparison"
    assert np.isfinite(row["cv_rmse_mean"])
    assert settings["cv_folds"] == 2
    assert settings["comparison_cv_folds"] == 3
    assert row["comparison_cv_folds"] == 3
    assert np.isfinite(saved.predict(X.iloc[:3])).all()


@pytest.mark.parametrize("degree", [1, 2])
def test_polynomial_ridge_handles_new_categories_after_reload(degree, tmp_path):
    from test_preprocessing import make_bookings

    data = make_bookings()
    X, y = (data.drop(columns="ADR"), data.ADR)
    pipeline = get_pipeline(
        "ridge", {"preprocess__encode__num__polynomial__degree": degree}
    )
    pipeline.fit(X, y)
    new = X.iloc[:3].assign(Country="UNSEEN")
    path = tmp_path / "ridge.joblib"
    joblib.dump(pipeline, path)
    np.testing.assert_allclose(pipeline.predict(new), joblib.load(path).predict(new))


def test_xgboost_pipeline():
    from xgboost import XGBRegressor

    pipeline = get_pipeline("xgboost")
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    model = pipeline.named_steps["model"]
    assert isinstance(model, XGBRegressor)
    assert model.tree_method == "hist" and model.n_jobs == -1
    numeric = pipeline.named_steps["preprocess"].named_steps["encode"].transformers[0][1]
    assert "scale" not in numeric.named_steps


def test_model_search_documentation_matches_ranges():
    from src.regression.models import MODELS

    cfg = load_config()
    assert cfg["tuning"]["n_iter"]["xgboost"] == 40
    for name, settings in MODELS.items():
        space = settings.get("search", cfg["tuning"].get(f"{name}_grid"))
        description = settings.get("search_description", space)
        assert set(space) == set(description)
        for key, distribution in space.items():
            if isinstance(distribution, list):
                assert description[key] == distribution
                continue
            lower, upper = distribution.support()
            kind = {"randint": "integers", "uniform": "uniform", "loguniform": "loguniform"}
            expected = f"{kind[distribution.dist.name]} {lower:g} to {upper:g}"
            assert description[key].replace("1.0", "1") == expected


def test_xgboost_tuning_two_candidates(tmp_path, monkeypatch):
    import json
    from test_preprocessing import make_bookings
    from src.data.groups import regression_group_ids
    from src.data.split import grouped_cv_folds

    data = make_bookings()
    X, y = data.drop(columns="ADR"), data.ADR
    groups = regression_group_ids(X)
    prep = SimpleNamespace(
        X_train=X, y_train=y,
        baseline_cv=grouped_cv_folds(X, groups, 5),
        tuning_cv=grouped_cv_folds(X, groups, 3),
    )
    cfg = load_config()
    cfg["regression"]["frozen_models"].remove("xgboost")
    cfg["tuning"]["n_iter"]["xgboost"] = 2
    cfg["outputs"]["dir"] = str(tmp_path)
    monkeypatch.setattr(train, "load_config", lambda: cfg)
    monkeypatch.setattr(train, "prepare_regression_data", lambda: prep)
    monkeypatch.setattr(train, "ARTIFACT_DIR", tmp_path / "artifacts")
    row = train.run_tune("xgboost")
    details = json.loads((tmp_path / "results/regression_xgboost_best_params.json").read_text())
    saved = joblib.load(tmp_path / "artifacts/regression_xgboost_tuned.joblib")
    assert details["configuration"]["tuning"]["n_iter"]["xgboost"] == 2
    assert details["cv_folds"] == 3 and row["comparison_cv_folds"] == 5
    assert saved.named_steps["model"].random_state == cfg["random_state"]
    assert saved.named_steps["model"].get_params()["n_estimators"] == details["best_params"]["model__n_estimators"]
    assert np.isfinite(saved.predict(X.iloc[:3])).all()
    assert {"train_mae", "train_mse", "train_rmse", "train_r2", "numeric_degree"} <= row.keys()


def test_refit_uses_frozen_recipe_training_only_without_search(tmp_path, monkeypatch):
    import json
    from test_preprocessing import make_bookings

    recipe = json.loads(
        (train.Path("outputs/results/regression_ridge_best_params.json")).read_text()
    )
    data = make_bookings()
    training_only = SimpleNamespace(X_train=data.drop(columns="ADR"), y_train=data.ADR)

    def prepare(config):
        assert config == recipe["configuration"]
        return training_only

    monkeypatch.setattr(train, "prepare_regression_data", prepare)
    monkeypatch.setattr(train, "load_config", lambda: pytest.fail("Current settings used"))
    for name in ("GridSearchCV", "RandomizedSearchCV", "cross_validate", "_write_row"):
        monkeypatch.setattr(train, name, lambda *a, **k: pytest.fail("Search or results changed"))
    monkeypatch.chdir(tmp_path)
    recipe_path = tmp_path / "outputs/results/regression_ridge_best_params.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(json.dumps(recipe))
    before = recipe_path.read_bytes()
    paths = train.run_refit("ridge")
    assert len(paths) == 2
    for stage, path in zip(("baseline", "tuned"), paths):
        saved = joblib.load(path)
        params = {"model__random_state": recipe["configuration"]["random_state"]}
        if stage == "tuned":
            params.update(recipe["best_params"])
        expected = get_pipeline("ridge", params).fit(training_only.X_train, training_only.y_train)
        np.testing.assert_array_equal(saved.named_steps["model"].coef_, expected.named_steps["model"].coef_)
        limits = json.loads(path.with_suffix(".bounds.json").read_text())
        assert limits["training_rows"] == len(data)
        assert limits["numeric"]["LeadTime"]["max"] == data.LeadTime.max()
    assert recipe_path.read_bytes() == before
    assert not (tmp_path / "outputs/tables").exists()


def test_cli_refit_dispatch(monkeypatch):
    called = []
    monkeypatch.setattr(train, "run_refit", lambda name: called.append(name))
    monkeypatch.setattr("sys.argv", ["main.py", "refit", "--task", "regression", "--model", "ridge"])
    main.main()
    assert called == ["ridge"]


def test_refit_rejects_unsupported_model(tmp_path, monkeypatch):
    import json
    from test_preprocessing import make_bookings

    recipe = json.loads(train.Path("outputs/results/regression_xgboost_best_params.json").read_text())
    data = make_bookings()
    training_only = SimpleNamespace(X_train=data.drop(columns="ADR"), y_train=data.ADR)
    monkeypatch.setattr(train, "prepare_regression_data", lambda config: training_only)
    output = tmp_path / "empty_artifacts"
    monkeypatch.setattr("sys.argv", [
        "main.py", "refit", "--task", "regression", "--model", "xgboost",
        "--output-dir", str(output),
    ])
    main.main()
    for stage in ("baseline", "tuned"):
        artifact = output / f"regression_xgboost_{stage}.joblib"
        pipeline = joblib.load(artifact)
        assert pipeline.named_steps["model"].random_state == recipe["configuration"]["random_state"]
        if stage == "tuned":
            for name, value in recipe["best_params"].items():
                assert pipeline.get_params()[name] == value
        assert json.loads(artifact.with_suffix(".bounds.json").read_text())["training_rows"] == len(data)
        assert json.loads(train.Path(str(artifact) + ".meta.json").read_text())["model"] == "xgboost"
