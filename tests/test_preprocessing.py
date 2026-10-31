"""Tests for the feature builder and the reusable preprocessing pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from src.data.load import load_clean
from src.data.split import make_split
from src.features import BookingFeatureBuilder, feature_columns, forbidden_columns
from src.preprocessing import build_model_pipeline, build_preprocessor

DATA = Path(__file__).resolve().parents[1] / "data" / "H2.csv"


def make_bookings(n: int = 60) -> pd.DataFrame:
    """Small fake table with every column the pipeline may see (incl. leaky ones)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "IsCanceled": rng.integers(0, 2, n),
        "LeadTime": rng.integers(0, 300, n),
        "ArrivalDateYear": 2016,
        "ArrivalDateMonth": rng.choice(["July", "August"], n),
        "ArrivalDateWeekNumber": rng.integers(1, 53, n),
        "ArrivalDateDayOfMonth": rng.integers(1, 28, n),
        "StaysInWeekendNights": rng.integers(0, 3, n),
        "StaysInWeekNights": rng.integers(1, 6, n),
        "Adults": 2,
        "Children": [np.nan] * 3 + [0.0] * (n - 3),
        "Babies": 0,
        "Meal": rng.choice(["BB", "HB"], n),
        "Country": ["PRT"] * 25 + ["FRA"] * 25 + [pd.NA] * 5 + ["XYZ"] * 5,
        "MarketSegment": rng.choice(["Online TA", "Direct"], n),
        "DistributionChannel": "TA/TO",
        "IsRepeatedGuest": 0,
        "PreviousCancellations": 0,
        "PreviousBookingsNotCanceled": 0,
        "ReservedRoomType": rng.choice(["A", "D"], n),
        "AssignedRoomType": rng.choice(["A", "D"], n),
        "BookingChanges": rng.integers(0, 3, n),
        "DepositType": "No Deposit",
        "Agent": ["9"] * 20 + ["7"] * 15 + ["3"] * 5 + [pd.NA] * 20,
        "Company": [pd.NA] * 55 + ["40"] * 5,
        "DaysInWaitingList": 0,
        "CustomerType": "Transient",
        "ADR": rng.uniform(50, 200, n),
        "RequiredCarParkingSpaces": 0,
        "TotalOfSpecialRequests": rng.integers(0, 3, n),
        "ReservationStatus": "Check-Out",
        "ReservationStatusDate": "2016-08-01",
    })


# ---- feature builder --------------------------------------------------------
def test_flags_are_created_before_raw_ids_are_dropped():
    df = make_bookings()
    out = BookingFeatureBuilder().fit(df).transform(df)
    assert out["HasAgent"].sum() == 40            # 20 + 15 + 5 agents recorded
    assert out["HasCompany"].sum() == 5           # not all zeros
    assert "Agent" not in out and "Company" not in out


def test_fixed_fills():
    out = BookingFeatureBuilder().fit(make_bookings()).transform(make_bookings())
    assert out["Children"].isna().sum() == 0
    assert out["Children"].iloc[:3].tolist() == [0, 0, 0]
    assert out["Country"].isna().sum() == 0
    assert (out["Country"] == "Unknown").sum() == 5


def test_top_n_agents_learned_from_fit_data_only():
    train = make_bookings()
    builder = BookingFeatureBuilder(top_n_agents=2).fit(train)
    assert builder.top_agents_ == ["9", "7"]
    new = pd.DataFrame({"Agent": ["9", "3", "555", pd.NA]})
    out = builder.transform(new)
    assert out["AgentTop"].tolist() == ["9", "Other", "Other", "None"]


def test_top_n_zero_adds_no_agent_column():
    df = make_bookings()
    assert "AgentTop" not in BookingFeatureBuilder().fit(df).transform(df)


def test_input_is_not_modified():
    df = make_bookings()
    before = df.copy()
    BookingFeatureBuilder().fit(df).transform(df)
    pd.testing.assert_frame_equal(df, before)


# ---- leakage rules -----------------------------------------------------------
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_no_forbidden_column_is_selected(task):
    numeric, categorical = feature_columns(task, top_n_agents=10, include_borderline=True)
    assert not (set(numeric + categorical) & forbidden_columns(task))


def test_adr_is_not_a_regression_feature_but_is_allowed_for_classification():
    assert "ADR" not in sum(feature_columns("regression"), [])
    assert "ADR" in feature_columns("classification")[0]


def test_borderline_columns_are_off_by_default_and_switchable():
    default = sum(feature_columns("regression"), [])
    ablation = sum(feature_columns("regression", include_borderline=True), [])
    assert "TotalOfSpecialRequests" not in default
    assert "TotalOfSpecialRequests" in ablation


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_leaky_values_cannot_change_the_prediction(task):
    """If a leaky column really is ignored, scrambling it changes nothing."""
    df = make_bookings()
    X, y = df.drop(columns=["ADR" if task == "regression" else "IsCanceled"]), \
        df["ADR" if task == "regression" else "IsCanceled"]
    prep = build_preprocessor(task).fit(X)
    scrambled = X.copy()
    scrambled["ReservationStatus"] = "Canceled"
    scrambled["AssignedRoomType"] = "Z"
    scrambled["BookingChanges"] = 99
    scrambled["DaysInWaitingList"] = 99
    np.testing.assert_array_equal(prep.transform(X), prep.transform(scrambled))


# ---- pipeline behaviour ------------------------------------------------------
def test_pipeline_handles_unseen_category_and_optional_fields():
    """Agent/Company/Children/Country are optional (NULL in the real data)."""
    df = make_bookings()
    X, y = df.drop(columns=["ADR"]), df["ADR"]
    pipe = build_model_pipeline("regression", RandomForestRegressor(n_estimators=5, random_state=0))
    pipe.fit(X, y)
    leaky = ["IsCanceled", "AssignedRoomType", "BookingChanges", "DaysInWaitingList",
             "ReservationStatus", "ReservationStatusDate"]
    new = X.iloc[[0]].drop(columns=leaky + ["Agent", "Company", "Children"])
    new["Country"] = "ZZZ"                                  # never seen in training
    pred = pipe.predict(new)
    assert pred.shape == (1,) and np.isfinite(pred).all()


def test_missing_required_field_fails_loudly():
    df = make_bookings()
    pipe = build_model_pipeline("regression", RandomForestRegressor(n_estimators=5, random_state=0))
    pipe.fit(df.drop(columns=["ADR"]), df["ADR"])
    incomplete = df.drop(columns=["ADR", "LeadTime"]).iloc[[0]]
    with pytest.raises(ValueError, match="LeadTime"):
        pipe.predict(incomplete)


def test_scaling_can_be_switched_off_and_output_has_no_nan():
    df = make_bookings()
    X = df.drop(columns=["ADR"])
    for scale in (True, False):
        out = build_preprocessor("regression", scale=scale).fit_transform(X)
        assert not np.isnan(out).any()


def test_median_is_learned_from_training_rows_only():
    train = make_bookings()
    test = make_bookings().assign(LeadTime=10_000)
    prep = build_preprocessor("regression").fit(train.drop(columns=["ADR"]))
    imputer = prep.named_steps["encode"].named_transformers_["num"].named_steps["impute"]
    lead_idx = prep.named_steps["encode"].transformers_[0][2].index("LeadTime")
    assert imputer.statistics_[lead_idx] == train["LeadTime"].median()
    prep.transform(test.drop(columns=["ADR"]))  # transforming must not refit
    assert imputer.statistics_[lead_idx] == train["LeadTime"].median()


def test_saved_pipeline_reloads_and_predicts_identically(tmp_path):
    df = make_bookings()
    X, y = df.drop(columns=["IsCanceled"]), df["IsCanceled"]
    pipe = build_model_pipeline("classification", RandomForestClassifier(n_estimators=5, random_state=0))
    pipe.fit(X, y)
    joblib.dump(pipe, tmp_path / "p.joblib")
    reloaded = joblib.load(tmp_path / "p.joblib")
    np.testing.assert_array_equal(pipe.predict(X), reloaded.predict(X))


def test_cross_validation_runs_with_full_pipeline():
    df = make_bookings()
    X, y = df.drop(columns=["ADR"]), df["ADR"]
    pipe = build_model_pipeline("regression", RandomForestRegressor(n_estimators=5, random_state=0))
    scores = cross_val_score(pipe, X, y, cv=3, scoring="neg_root_mean_squared_error")
    assert len(scores) == 3


# ---- split ------------------------------------------------------------------
def test_split_is_reproducible_stratified_and_keeps_target_out_of_X():
    df = pd.concat([make_bookings()] * 4, ignore_index=True)
    a = make_split(df, "classification", random_state=42)
    b = make_split(df, "classification", random_state=42)
    pd.testing.assert_frame_equal(a[0], b[0])
    assert "IsCanceled" not in a[0]
    assert abs(a[2].mean() - a[3].mean()) < 0.05


# ---- real H2 file -----------------------------------------------------------
@pytest.mark.skipif(not DATA.exists(), reason="data/H2.csv not available")
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_real_h2_end_to_end(task):
    df, _ = load_clean(DATA)
    X_train, X_test, y_train, y_test = make_split(df, task)
    model = (RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
             if task == "regression"
             else RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1))
    pipe = build_model_pipeline(task, model, scale=False).fit(X_train, y_train)
    assert np.isfinite(pipe.predict(X_test)).all()
