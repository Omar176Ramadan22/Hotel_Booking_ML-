import ast
import copy
import csv
import io
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

import main
from src.data import analysis
from src.data.load import clean
from src.data.rules import (
    arrival_week,
    positive_total,
    GUEST_COLUMNS,
    NIGHT_COLUMNS,
    write_training_limits,
)
from src.data.preparation import load_config
from src.inference import predict
from src.inference.predict import EXAMPLE_BOOKING, validate_bookings
from src.features.columns import feature_columns
from src.regression import train
from src.regression.models import get_pipeline
from src.preprocessing.cleaning import invalid_guest_count, zero_night_stays
from test_preprocessing import make_bookings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_model(tmp_path, monkeypatch):
    data = make_bookings()
    artifact = tmp_path / "regression_ridge_tuned.joblib"
    pipeline = get_pipeline("ridge").fit(data.drop(columns="ADR"), data.ADR)
    joblib.dump(pipeline, artifact)
    write_training_limits(artifact, data, feature_columns("regression")[0])
    monkeypatch.setattr(predict, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(predict, "default_model", lambda: "ridge")
    return artifact


def test_only_evaluation_consumer_reads_y_test():
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                attribute = (
                    isinstance(node, ast.Attribute)
                    and node.attr == "y_test"
                    and isinstance(node.ctx, ast.Load)
                )
                subscript = (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value == "y_test"
                    and isinstance(node.ctx, ast.Load)
                )
                if attribute or subscript:
                    assert (
                        path.name == "train.py" and function.name == "run_evaluate"
                    ), (path, function.name)


def test_analyze_never_reads_test_targets(monkeypatch, capsys):
    prepared = SimpleNamespace(
        X_train=pd.DataFrame({"x": [1]}),
        X_test=pd.DataFrame({"x": [2]}),
        groups_train=pd.Series([1]),
        groups_test=pd.Series([2]),
        cleaning_log=[],
    )
    monkeypatch.setattr(analysis, "prepare_regression_data", lambda: prepared)
    monkeypatch.setattr(analysis, "load_raw", lambda path: pd.DataFrame({"x": [1, 2]}))
    analysis.run_analyze()
    assert "Overlapping groups: 0" in capsys.readouterr().out


def test_saved_results_keep_degree_and_verifier_columns():
    settings = json.loads(
        (ROOT / "outputs/results/regression_ridge_best_params.json").read_text()
    )
    degree = settings["best_params"]["preprocess__encode__num__polynomial__degree"]
    with (ROOT / "outputs/tables/regression_cv_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    tuned = next(row for row in rows if row["stage"] == "tuned")
    assert int(tuned["numeric_degree"]) == degree
    assert json.loads(tuned["parameters"])["numeric_degree"] == degree
    assert float(tuned["cv_rmse_mean"]) > 0
    with (ROOT / "outputs/tables/regression_test_results.csv").open() as stream:
        assert all(float(row["test_rmse"]) > 0 for row in csv.DictReader(stream))
    assert settings["frozen"] is True
    assert settings["recorded_protocol"]["search_folds"] == 5


@pytest.mark.parametrize(
    "columns,remove",
    [(GUEST_COLUMNS, invalid_guest_count), (NIGHT_COLUMNS, zero_night_stays)],
)
def test_shared_zero_total_rule_matches_notebook_cleaning(columns, remove):
    data = pd.DataFrame({column: [0, 1, None] for column in columns})
    expected = data.loc[positive_total(data, columns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(remove(data, columns=columns), expected)
    cleaned, _ = clean(data, drop_duplicates=False)
    pd.testing.assert_frame_equal(cleaned, expected)


@pytest.mark.parametrize(
    "task,model",
    [("classification", "xgboost"), ("classification", "all"), ("regression", "all")],
)
def test_parser_accepts_task_specific_names_and_all(task, model):
    args = main.build_parser().parse_args(["train", "--task", task, "--model", model])
    assert args.model == model


def test_all_regression_dispatch(monkeypatch):
    from src.data import preparation

    cfg = load_config()
    cfg["regression"]["frozen_models"] = []
    monkeypatch.setattr(preparation, "load_config", lambda: cfg)
    called = []
    monkeypatch.setattr(train, "run_train", lambda name: called.append(name))
    monkeypatch.setattr(
        "sys.argv", ["main.py", "train", "--task", "regression", "--model", "all"]
    )
    main.main()
    assert called == list(main.MODELS)


@pytest.mark.parametrize("function", [train.run_train, train.run_tune])
def test_frozen_ridge_refuses_training_before_data_is_prepared(function, monkeypatch):
    monkeypatch.setattr(
        train,
        "prepare_regression_data",
        lambda: pytest.fail("Frozen model prepared training data"),
    )
    with pytest.raises(ValueError, match="frozen"):
        function("ridge")


def test_range_limits_reject_extreme_inputs_before_prediction(saved_model, monkeypatch):
    limits = json.loads(saved_model.with_suffix(".bounds.json").read_text())
    monkeypatch.setattr(
        predict.joblib,
        "load",
        lambda path: pytest.fail("Out-of-range input reached prediction"),
    )
    booking = copy.deepcopy(EXAMPLE_BOOKING)
    booking["Adults"] = 10
    with pytest.raises(ValueError, match="Adults.*training range"):
        predict.predict_bookings(booking)
    booking = copy.deepcopy(EXAMPLE_BOOKING)
    booking["StaysInWeekNights"] = 60
    with pytest.raises(ValueError, match="training range"):
        predict.predict_bookings(booking)
    for year in (2018, 2020, 2026):
        booking = copy.deepcopy(EXAMPLE_BOOKING)
        booking["ArrivalDateYear"] = year
        booking["ArrivalDateWeekNumber"] = arrival_week(date(year, 7, 15))
        with pytest.raises(ValueError, match="ArrivalDateYear.*training range"):
            predict.predict_bookings(booking)
    assert limits["numeric"]["Adults"]["max"] == 2


def test_combined_nights_limit_is_checked(saved_model, monkeypatch):
    limits = json.loads(saved_model.with_suffix(".bounds.json").read_text())
    limits["totals"]["nights"]["max"] = 3
    saved_model.with_suffix(".bounds.json").write_text(json.dumps(limits))
    with pytest.raises(ValueError, match="Total nights"):
        predict.predict_bookings(EXAMPLE_BOOKING)


@pytest.mark.parametrize(
    "year,month,day,week", [(2015, 7, 5, 28), (2016, 7, 15, 29), (2017, 1, 1, 1)]
)
def test_h2_week_convention(year, month, day, week):
    assert arrival_week(date(year, month, day)) == week


def test_inconsistent_week_and_month_is_rejected():
    booking = copy.deepcopy(EXAMPLE_BOOKING)
    booking["ArrivalDateWeekNumber"] = 2
    with pytest.raises(ValueError, match="must be 29"):
        validate_bookings(booking)


def test_clipping_warning_is_visible_in_cli_json(
    saved_model, monkeypatch, tmp_path, capsys
):
    pipeline = joblib.load(saved_model)
    monkeypatch.setattr(pipeline, "predict", lambda data: np.array([-2.0]))
    monkeypatch.setattr(
        predict.joblib,
        "load",
        lambda path: pipeline,
    )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(EXAMPLE_BOOKING))
    result = predict.run_predict(input_path=path)
    assert result["predictions"] == [{"ADR": 0.0}]
    assert "clipped to 0" in result["warnings"][0]
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("frozen_models", [["ridge"], list(main.MODELS)])
def test_all_training_skips_frozen_models(monkeypatch, capsys, frozen_models):
    from src.data import preparation

    cfg = load_config()
    cfg["regression"]["frozen_models"] = frozen_models
    monkeypatch.setattr(preparation, "load_config", lambda: cfg)
    called = []
    monkeypatch.setattr(train, "run_train", lambda name: called.append(name))
    monkeypatch.setattr(
        "sys.argv", ["main.py", "train", "--task", "regression", "--model", "all"]
    )
    main.main()
    assert called == [name for name in main.MODELS if name not in frozen_models]
    output = capsys.readouterr().out
    for name in frozen_models:
        assert f"{name}: frozen; skipped" in output
