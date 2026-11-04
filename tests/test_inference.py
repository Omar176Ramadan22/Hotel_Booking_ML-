import copy
import json
from types import SimpleNamespace
import joblib
import numpy as np
import pytest
from sklearn.linear_model import Ridge
import main
from src.inference import predict
from src.inference.predict import EXAMPLE_BOOKING, validate_bookings
from src.data.rules import write_training_limits
from src.features.columns import feature_columns
from src.data.preparation import load_config
from src.data.split import grouped_cv_folds
from src.data.groups import regression_group_ids
from src.regression.models import get_pipeline
from src.regression import diagnostics
from test_preprocessing import make_bookings


@pytest.fixture
def saved_model(tmp_path, monkeypatch):
    data = make_bookings()
    pipeline = get_pipeline("ridge").fit(data.drop(columns="ADR"), data.ADR)
    joblib.dump(pipeline, tmp_path / "regression_ridge_tuned.joblib")
    write_training_limits(
        tmp_path / "regression_ridge_tuned.joblib",
        data,
        feature_columns("regression")[0],
    )
    monkeypatch.setattr(predict, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(predict, "default_model", lambda: "ridge")
    return pipeline


def test_saved_prediction_does_not_retrain(saved_model, monkeypatch):
    expected = saved_model.predict(predict.normalize_categories(validate_bookings(EXAMPLE_BOOKING), saved_model))
    monkeypatch.setattr(
        Ridge, "fit", lambda *args, **kwargs: pytest.fail("Inference retrained")
    )
    actual = predict.predict_bookings(EXAMPLE_BOOKING)
    np.testing.assert_allclose(actual, expected)


def test_batch_json_and_interactive_prediction(
    saved_model, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "bookings.json"
    path.write_text(json.dumps([EXAMPLE_BOOKING, EXAMPLE_BOOKING]))
    result = predict.run_predict(input_path=path)
    assert len(result["predictions"]) == 2
    capsys.readouterr()
    monkeypatch.setattr("builtins.input", lambda label: "")
    result = predict.run_predict()
    assert len(result["predictions"]) == 1
    assert json.loads(capsys.readouterr().out)["predictions"] == result["predictions"]


def test_cli_predict_loads_json(saved_model, tmp_path, monkeypatch, capsys):
    path = tmp_path / "booking.json"
    path.write_text(json.dumps(EXAMPLE_BOOKING))
    monkeypatch.setattr(
        "sys.argv", ["main.py", "predict", "--task", "regression", "--input", str(path)]
    )
    main.main()
    output = json.loads(capsys.readouterr().out)
    assert np.isfinite(output["predictions"][0]["ADR"])


@pytest.mark.parametrize(
    "updates",
    [
        {"LeadTime": -1},
        {"Adults": 1.5},
        {"LeadTime": "not a number"},
        {"LeadTime": float("inf")},
        {"ArrivalDateMonth": "NotAMonth"},
        {"ArrivalDateMonth": "February", "ArrivalDateDayOfMonth": 31},
        {"ArrivalDateWeekNumber": 54},
        {"IsRepeatedGuest": 2},
        {"Adults": 0, "Children": 0, "Babies": 0},
        {"StaysInWeekendNights": 0, "StaysInWeekNights": 0},
        {"Meal": ["BB"]},
        {"Meal": 123},
    ],
)
def test_invalid_bookings_are_rejected(updates):
    booking = copy.deepcopy(EXAMPLE_BOOKING)
    booking.update(updates)
    with pytest.raises(ValueError):
        validate_bookings(booking)


def test_missing_required_and_optional_fields(saved_model):
    booking = copy.deepcopy(EXAMPLE_BOOKING)
    del booking["LeadTime"]
    with pytest.raises(ValueError, match="LeadTime"):
        validate_bookings(booking)
    booking = {
        name: value
        for name, value in EXAMPLE_BOOKING.items()
        if name not in {"Agent", "Company", "Country", "Children"}
    }
    assert np.isfinite(predict.predict_bookings(booking)).all()


def test_invalid_json_and_missing_artifact(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text("{")
    with pytest.raises(ValueError, match="Invalid JSON"):
        predict.run_predict(input_path=path)
    monkeypatch.setattr(predict, "ARTIFACT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"run: python main\.py refit --task regression --model ridge"):
        predict.predict_bookings(EXAMPLE_BOOKING, model_name="ridge")


def test_diagnostics_use_only_training_rows(tmp_path, monkeypatch):
    data = make_bookings()
    X, y = (data.drop(columns="ADR"), data.ADR)
    pipeline = get_pipeline("ridge").fit(X, y)
    joblib.dump(pipeline, tmp_path / "regression_ridge_tuned.joblib")
    prepared = SimpleNamespace(
        X_train=X,
        y_train=y,
        baseline_cv=grouped_cv_folds(X, regression_group_ids(X), 3),
    )
    cfg = load_config()
    cfg["outputs"]["dir"] = str(tmp_path / "outputs")
    monkeypatch.setattr(diagnostics, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(diagnostics, "load_config", lambda: cfg)
    monkeypatch.setattr(diagnostics, "prepare_regression_data", lambda: prepared)
    result = diagnostics.run_diagnostics()
    assert result["rows"] == len(X)
    for path in (tmp_path / "outputs/figures").glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG")
    assert len(list((tmp_path / "outputs/figures").glob("*.png"))) == 2


def test_negative_model_output_is_clipped_with_warning(saved_model, monkeypatch):
    monkeypatch.setattr(saved_model, "predict", lambda data: np.array([-1.0]))
    monkeypatch.setattr(
        predict.joblib,
        "load",
        lambda path: saved_model,
    )
    with pytest.warns(UserWarning, match="clipped to 0"):
        values = predict.predict_bookings(EXAMPLE_BOOKING)
    assert values.tolist() == [0.0]


def test_nonfinite_model_output_is_rejected(saved_model, monkeypatch):
    monkeypatch.setattr(saved_model, "predict", lambda data: np.array([float("inf")]))
    monkeypatch.setattr(
        predict.joblib,
        "load",
        lambda path: saved_model,
    )
    with pytest.raises(ValueError, match="non-finite"):
        predict.predict_bookings(EXAMPLE_BOOKING)


@pytest.mark.parametrize("field", predict.CATEGORY_FIELDS)
def test_category_case_and_padding_do_not_change_prediction(saved_model, field):
    booking = {**EXAMPLE_BOOKING, field: "  " + EXAMPLE_BOOKING[field].lower() + "  "}
    normalized = validate_bookings(booking)
    assert normalized.loc[0, field] == EXAMPLE_BOOKING[field].upper()
    expected = saved_model.predict(predict.normalize_categories(validate_bookings(EXAMPLE_BOOKING), saved_model))
    np.testing.assert_array_equal(predict.predict_bookings(booking), expected)
    np.testing.assert_array_equal(predict.predict_bookings(booking), predict.predict_bookings(EXAMPLE_BOOKING))


@pytest.mark.parametrize("field", predict.CATEGORY_FIELDS)
def test_unknown_category_lists_training_options(saved_model, field):
    booking = {**EXAMPLE_BOOKING, field: "NEVER_SEEN"}
    with pytest.raises(ValueError, match=f"Unknown {field}.*Valid options:"):
        predict.predict_bookings(booking)


def test_unknown_room_lists_all_valid_rooms(saved_model):
    with pytest.raises(ValueError, match="Unknown ReservedRoomType: 'Z'. Valid options: A, D"):
        predict.predict_bookings({**EXAMPLE_BOOKING, "ReservedRoomType": "Z"})
