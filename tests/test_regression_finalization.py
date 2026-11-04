import ast
import inspect
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from src.data.groups import regression_group_ids
from src.data.rules import write_training_limits
from src.data.split import grouped_cv_folds
from src.evaluation.comparison import selected_result
from src.features.columns import feature_columns
from src.inference import predict
from src.regression import diagnostics, pca_experiment, train
from src.regression.models import get_pipeline
from test_preprocessing import make_bookings


@pytest.mark.parametrize("name", ["ridge", "xgboost", "random_forest"])
@pytest.mark.parametrize("command", ["train", "tune", "evaluate"])
def test_finalized_commands_stop_before_data_or_model_loading(name, command, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ARTIFACT_DIR", tmp_path)
    for stage in ("baseline", "tuned"):
        (tmp_path / f"regression_{name}_{stage}.joblib").touch()
    monkeypatch.setattr(train, "prepare_regression_data", lambda: pytest.fail("Prepared training data"))
    monkeypatch.setattr(train, "prepare_evaluation_data", lambda: pytest.fail("Prepared held-out data"))
    monkeypatch.setattr(train.joblib, "load", lambda *a: pytest.fail("Loaded a model"))
    message = "held-out evaluation is disabled" if command == "evaluate" else "frozen.*refit"
    with pytest.raises(ValueError, match=message):
        getattr(train, f"run_{command}")(name)


def test_selected_model_is_read_from_recorded_comparison(tmp_path, monkeypatch):
    path = tmp_path / "comparison.csv"
    path.write_text("model,SELECTED,cv_rmse_mean,test_rmse\nridge,False,19.78,20.15\nxgboost,True,14.40,14.61\n")
    monkeypatch.setattr(predict, "selected_result", lambda: selected_result(path))
    assert predict.default_model() == "xgboost"
    path.write_text("model,SELECTED\nridge,True\nxgboost,True\n")
    with pytest.raises(ValueError, match="exactly one"):
        selected_result(path)


@pytest.mark.parametrize("name", ["ridge", "xgboost", "random_forest"])
def test_prediction_uses_saved_pipeline_without_fit_or_search(name, tmp_path, monkeypatch):
    data = make_bookings()
    parameters = {} if name == "ridge" else {"model__n_estimators": 3, "model__max_depth": 2}
    pipeline = get_pipeline(name, parameters).fit(data.drop(columns="ADR"), data.ADR)
    path = tmp_path / f"regression_{name}_tuned.joblib"
    joblib.dump(pipeline, path)
    write_training_limits(path, data, feature_columns("regression")[0])
    monkeypatch.setattr(predict, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(predict, "default_model", lambda: name)
    monkeypatch.setattr(Pipeline, "fit", lambda *a, **k: pytest.fail("Prediction fitted a pipeline"))
    monkeypatch.setattr(type(pipeline.named_steps["model"]), "fit", lambda *a, **k: pytest.fail("Prediction fitted an estimator"))
    monkeypatch.setattr(train, "run_refit", lambda *a, **k: pytest.fail("Prediction triggered refit"))
    monkeypatch.setattr(train, "run_tune", lambda *a, **k: pytest.fail("Prediction triggered search"))
    assert np.isfinite(predict.predict_bookings(predict.EXAMPLE_BOOKING)).all()
    assert np.isfinite(predict.predict_bookings(predict.EXAMPLE_BOOKING, name)).all()
    with pytest.raises(ValueError, match="training range"):
        predict.predict_bookings({**predict.EXAMPLE_BOOKING, "Adults": 10}, name)
    monkeypatch.setattr(predict, "ARTIFACT_DIR", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match=f"refit --task regression --model {name}"):
        predict.predict_bookings(predict.EXAMPLE_BOOKING, name)


def test_inference_and_pca_source_have_no_training_or_holdout_access():
    inference = ast.parse(inspect.getsource(predict.predict_bookings))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {"fit", "fit_transform"} for node in ast.walk(inference))
    source = inspect.getsource(pca_experiment)
    assert "X_test" not in source and "y_test" not in source
    assert "RandomizedSearchCV" not in source and "GridSearchCV" not in source


def test_rf_training_diagnostics_do_not_fit(tmp_path, monkeypatch):
    data = make_bookings()
    X, y = data.drop(columns="ADR"), data.ADR
    pipeline = get_pipeline("random_forest", {"model__n_estimators": 3}).fit(X, y)
    joblib.dump(pipeline, tmp_path / "regression_random_forest_tuned.joblib")
    ridge = get_pipeline("ridge").fit(X, y)
    joblib.dump(ridge, tmp_path / "regression_ridge_tuned.joblib")
    cfg = train.load_config()
    cfg["outputs"]["dir"] = str(tmp_path / "outputs")
    monkeypatch.setattr(diagnostics, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(diagnostics, "load_config", lambda: cfg)
    monkeypatch.setattr(diagnostics, "prepare_regression_data", lambda: SimpleNamespace(X_train=X, y_train=y))
    monkeypatch.setattr(Pipeline, "fit", lambda *a, **k: pytest.fail("Diagnostic fitted a model"))
    monkeypatch.setattr(diagnostics, "cross_val_predict", lambda *a, **k: pytest.fail("Diagnostic refitted CV models"))
    result = diagnostics.run_training_diagnostics()
    assert result["source"] == "in_sample_training_not_cv_or_holdout"
    assert result["rows"] == len(X)
    assert len(list((tmp_path / "outputs/figures").glob("*.png"))) == 3
    assert len(pd.read_csv(tmp_path / "outputs/results/regression_random_forest_feature_importance.csv")) <= 15
    ridge_result = diagnostics.run_training_diagnostics("ridge")
    assert ridge_result["source"] == "in_sample_training_not_cv_or_holdout"
    assert len(list((tmp_path / "outputs/figures").glob("*.png"))) == 5


def test_pca_and_preprocessing_are_fitted_only_on_each_training_fold(tmp_path, monkeypatch):
    data = make_bookings()
    X, y = data.drop(columns="ADR"), data.ADR
    groups = regression_group_ids(X)
    folds = grouped_cv_folds(X, groups, 2)
    prepared = SimpleNamespace(X_train=X, y_train=y, groups_train=groups, baseline_cv=folds)
    recipes = {
        "ridge": {"best_params": {"model__alpha": 0.0001, "preprocess__encode__num__polynomial__degree": 2}},
        "xgboost": {"best_params": {"model__n_estimators": 3, "model__max_depth": 2}},
    }
    original_fit, original_pca = Pipeline.fit, PCA.fit_transform
    fitted_indices, pca_rows = [], []

    def fit(pipeline, features, target=None, **kwargs):
        if "model" in pipeline.named_steps:
            assert not hasattr(pipeline.named_steps["model"], "n_features_in_")
            fitted_indices.append(set(features.index))
            assert set(features.index) in [set(X.index[indices]) for indices, _ in folds]
        return original_fit(pipeline, features, target, **kwargs)

    def pca_fit(pca, features, target=None, **kwargs):
        assert not hasattr(pca, "components_")
        assert len(features) in [len(indices) for indices, _ in folds]
        pca_rows.append(len(features))
        return original_pca(pca, features, target, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(PCA, "fit_transform", pca_fit)
    output, artifacts = tmp_path / "results", tmp_path / "experimental_artifacts"
    summary = pca_experiment.compare_folds(prepared, recipes, output, artifacts)
    assert len(fitted_indices) == 12 and len(pca_rows) == 4
    assert len(summary) == 6
    assert set(summary["variant"]) == {"original", "standardized", "pca_95"}
    assert (summary["variance_retained_mean"] >= 0.95).all()
    assert (summary["features_after_max"] <= summary["features_before_max"]).all()
    assert len(list(artifacts.glob("*.joblib"))) == 4
    for path in artifacts.glob("*.joblib"):
        pipeline = joblib.load(path)
        assert list(pipeline.named_steps) == ["preprocess", "pca_scale", "pca", "model"]
        assert pipeline.named_steps["pca"].n_components == 0.95
        if path.name.startswith("ridge"):
            assert pipeline.named_steps["model"].alpha == 0.0001
    assert json.loads((output / "protocol.json").read_text())["heldout_used"] is False
    with pytest.raises(FileExistsError):
        pca_experiment.compare_folds(prepared, recipes, output, artifacts)
